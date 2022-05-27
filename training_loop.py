# Copyright 2022 Google.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic JAX training loop for experiments."""

import functools
import os

from typing import (Any, Callable, Dict, Optional, Sequence, Tuple)

from absl import logging

from clu import metric_writers
# from clu import periodic_actions

import flax
from flax import jax_utils
from flax import linen as nn
from flax import struct
from flax.training import checkpoints

import gin
import jax
import jax.numpy as jnp
import numpy as np

import tensorflow.compat.v2 as tf

import  metrics_summary
import  optimizer_config as opt_config
import  training_task


PRNGKeys = training_task.PRNGKeys
TrainState = training_task.TrainState
TrainingTask = training_task.TrainingTask
StepFunction = training_task.StepFunction
Metrics = training_task.Metrics
MetricWriter = metric_writers.MetricWriter
MetricsSummary = metrics_summary.MetricsSummary


gfile = tf.io.gfile
unfreeze = flax.core.unfreeze
flatten_dict = flax.traverse_util.flatten_dict
should_run = training_task.should_run


# TODO(cstaats): Use a Protocol to specify that it must be possible to call
# the function with parameters (step: int, mode: str). This won't be feasible
# until we start using Python 3.8 or later.
StepModeCallable = Callable[..., None]


# This variable should *only* be set from register_interstep_callbacks.
_interstep_callbacks: Optional[Tuple[StepModeCallable, ...]] = None


@gin.configurable
def register_interstep_callbacks(**kwargs: StepModeCallable) -> None:
  """Populates _interstep_callbacks from gin.

  This function should be called exactly ONCE and that call should happen AFTER
  flag initialization (and more specifically, after gin parsing). And the caller
  should NOT specify any arguments.

  In gin configurations, a callback can be specified with an arbitrary name
  like so:

      register_interstep_callbacks.my_callback_name = @my_callback_function

  Multiple callbacks can be registered without overriding each other as long as
  they all have different names. Conversely, if you *want* to override a
  callback, you need to give that callback the same name.

  Args:
    **kwargs: Specified by gin. Each argument should be a function (callable)
      that can be called as my_function(step, mode), where step is an int and
      mode is a str.

  Raises:
    ValueError: Raised on the second (and any subsequent) function call.
  """
  global _interstep_callbacks
  logging.info("registering functions: %s", kwargs.keys())
  if _interstep_callbacks is not None:
    raise ValueError("register_interstep_callbacks may only be called once.")
  _interstep_callbacks = tuple(kwargs.values())


def clear_interstep_callbacks():
  """Clear all registered callbacks, so that new ones can be registered."""
  global _interstep_callbacks
  _interstep_callbacks = None


def run_interstep_callbacks(mode: str, step: int, sub_step: int = 0):
  """Run the registered callbacks.

  Args:
    mode: mode of the task to execute callbacks for.
    step: training step number.
    sub_step: For tasks that execute multiple iterations within a step.
      E.g. a test cycle that runs multiple testing steps.
  """
  for func in _interstep_callbacks:
    func(sub_step or step, mode)


@gin.configurable
@struct.dataclass
class Trainer:
  """Implements a JAX training loop."""

  # Returns a Flax module for the model.
  # Takes a single argument mode, which can be "test", "train", or "generate".
  model_definition: Any = gin.REQUIRED

  # Iterator over trainining data.
  get_training_dataset_iterator: Callable[[], Any] = gin.REQUIRED

  # Iterator over test data.
  get_test_dataset_iterator: Optional[Callable[[], Any]] = None

  workdir: str = ""                    # Working directory for checkpoints.
  load_dir: str = ""                   # Optional directory to load model.
  num_steps: int = 100000              # Number of steps to train.
  status_every_steps: int = 10         # Log step number every N steps.
  log_every_steps: int = 100           # Log scalar data every N steps.
  test_every_steps: int = 10           # Test model every N steps.
  num_test_steps: int = 1              # Number of iterations to test.
  generate_every_steps: int = 1000     # Generate examples every N steps.
  print_input_every_steps: int = 1000  # Print example data every N steps.

  save_checkpoints: bool = True        # Save training checkpoints
  checkpoint_every_steps: int = 5000   # Save checkpoints every N steps.
  restore_checkpoints: bool = True     # Restore from previous checkpoint.
  restore_state_variables: bool = True  # Restore TrainState.state from chkpt.

  # Record metrics for "train", "test", etc. in separate directories.
  # Otherwise they will be saved with separate prefixes.
  use_separate_metric_directories: bool = True

  # Optimizer options.
  optimizer_factory: opt_config.OptimizerConfig = gin.REQUIRED
  learning_rate_schedule: Callable[[jnp.ndarray, int], jnp.ndarray] = (
      opt_config.lr_cosine_decay)

  # Maximum steps for the LR schedule.  Zero means use num_steps.
  max_scheduled_steps: int = 0
  warmup_steps: int = 1000               # Number of warmup steps.
  learning_rate_multiplier: float = 1.0  # Used to scale the learning rate.

  random_seed: int = 42                  # Initial random seed.

  # Names of random number generators used by the model.
  rng_key_names: Optional[Sequence[str]] = ("dropout",)

  # Debug options.
  replicate_mode: bool = True     # pmap over multiple replicas.
  trace_debug_mode: bool = False  # Run in eager mode to trace results.
  print_variables: bool = False   # Dump parameters/variables to stdout.

  # Function to compute additional summary information.
  # Takes a MetricsSummary object and a mode string (e.g. "test") as arguments,
  # returns a MetricsSummary object.
  process_summaries_function: Optional[Callable[[Any, str], Any]] = None

  # Function to pretty print the input for each training step.
  pretty_print_input_function: Optional[Callable[[Any], Any]] = None

  # Classes to use for summarizing metrics.
  metrics_summary_factory: Any = metrics_summary.MetricsSummary
  extra_summaries_fn: training_task.ExtraSummariesFunction = (
      lambda mode, step: dict())

  post_save_checkpoint_fn: Callable[[str, int], None] = lambda mode, step: None
  post_load_checkpoint_fn: Callable[[str, int], None] = lambda mode, step: None

  def learning_rate_schedule_fn(self, step):
    """Returns the learning rate for the given step."""

    # There are four components to the learning rate.
    #
    # The base_lrate is defined by the optimizer, and different optimizers have
    # different relative rates, e.g. Adafactor requires a higher LR than Adam.
    # By default, the base_lrate is 1.0 for Adafactor.
    #
    # The base_lrate is then multiplied by the learning rate decay schedule,
    # which typically starts at a maximum value and decays over time.
    # Each schedule can be individually configured, e.g. from 0.01 to 0.001.
    # The max_scheduled_steps parameter controls the decay rate of the schedule.
    #
    # Finally, the LR is scaled by the learning_rate_multiplier, which provides
    # an easy way to scale the LR for hyperparameter tuning in a way that is
    # independent of the choice of schedule or optimizer.  The default is 1.0.
    #
    # During the warmp period, the learning rate ramps up linearly from zero.

    step = jnp.asarray(step, dtype=jnp.float32)
    if self.max_scheduled_steps == 0:
      max_steps = self.num_steps
    else:
      max_steps = self.max_scheduled_steps

    base_lrate = float(self.optimizer_factory.learning_rate)
    lr_multiplier = float(self.learning_rate_multiplier)

    # Linear increase in learning rate up to warmup_steps.
    warmup_steps = float(self.warmup_steps)
    lr_warmup_ramp = jnp.minimum(step, warmup_steps) / warmup_steps

    # Hold step at a constant value during the warmup period.
    # Required for some schedules, like rsqrt_decay.
    step = jnp.maximum(step, warmup_steps)

    # Get the scheduled learning rate.
    lrate = self.learning_rate_schedule(step, max_steps)

    # Multiply lrate by the base, warmup and multiplier factors.
    lrate = lrate * base_lrate * lr_warmup_ramp * lr_multiplier
    return jnp.asarray(lrate, dtype=jnp.float32)

  def _init_rngs(self, rngs: PRNGKeys, step: int) -> PRNGKeys:
    # Get a new random number generator for each step
    rngs = jax.random.fold_in(rngs, step)
    rngs = jax.random.split(rngs, len(self.rng_key_names))
    rngs = {key: rngs[i] for i, key in enumerate(self.rng_key_names)}
    return rngs

  def train_step(self, model: nn.Module, tstate: TrainState, x: Any,
                 rngs: PRNGKeys) -> Tuple[TrainState, Metrics]:
    """Perform a training step, pmapped over multiple devices.

    Args:
      model:  The model to use for the step function.
      tstate: Values for state variables, and the optimizer.
      x:      A batch of inputs to train on.
      rngs:   PRNGKey (possibly replicated).

    Returns:
      Tuple of (new_tstate, metrics: dictionary of scalar values)
    """

    mutable_keys = [k for (k, _) in tstate.state.items()]
    step = tstate.optimizer.state.step
    rngs = self._init_rngs(rngs, step)

    # Refactor the model as a loss function from trainable params to loss, so
    # that we can differentiate with jax and get {d}loss/{d}params.
    # Inputs and non-trainable params are bound within the closure.
    # model:: x, { state_params } -> (loss, metrics), { new_state_params }
    # loss_fn:: params -> (loss, (metrics, new_state))
    def loss_fn(params):
      """Loss function."""
      (loss, mets), nstate = model.apply({"params": params, **tstate.state},
                                         x,
                                         rngs=rngs,
                                         mutable=mutable_keys)
      return loss, (mets, nstate)

    # grad_fn:: params -> ((loss, (aux, nstate)), param_gradients)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    # Run forward and backward pass.
    (loss, (metrics, new_state)), param_grads = grad_fn(tstate.optimizer.target)
    del loss  # loss is only recorded if it is part of the metrics
    if self.replicate_mode:
      param_grads = jax.lax.pmean(param_grads, axis_name="batch")
    lrate = self.learning_rate_schedule_fn(step)
    new_optimizer = tstate.optimizer.apply_gradient(
        param_grads, learning_rate=lrate)

    # Metrics are summary values that will be logged.
    if self.replicate_mode:
      # Merge metrics (take mean/sum etc.) over replicas on-device.
      summary_class = self.metrics_summary_factory
      metrics = summary_class.merge_replicated_metrics(
          metrics, model.metrics_summary_operations(aggregate_over="devices"))

    metrics["learning_rate"] = lrate
    return (TrainState(new_optimizer, new_state), metrics)

  def other_step(self, model: nn.Module, tstate: TrainState, x: Any,
                 rngs: PRNGKeys) -> Tuple[TrainState, Metrics]:
    """Perform a test or generate step, pmapped over multiple devices.

    Args:
      model:  The model to use for the step function.
      tstate: Values for state variables, and the optimizer.
      x:      A batch of inputs to train on.
      rngs:   PRNGKey (possibly replicated).

    Returns:
      Tuple of (new_tstate, metrics: dictionary of scalar values)
    """

    mutable_keys = [k for (k, _) in tstate.state.items()]
    step = tstate.optimizer.state.step
    rngs = self._init_rngs(rngs, step)

    params = tstate.optimizer.target
    (loss, metrics), new_state = model.apply({"params": params, **tstate.state},
                                             x,
                                             rngs=rngs,
                                             mutable=mutable_keys)
    del loss  # loss is only recorded if it is part of the metrics

    # Metrics are summary values that will be logged.
    if self.replicate_mode:
      # Merge metrics (take mean/sum etc.) over replicas on-device.
      summary_class = self.metrics_summary_factory
      metrics = summary_class.merge_replicated_metrics(
          metrics, model.metrics_summary_operations(aggregate_over="devices"))

    return (TrainState(tstate.optimizer, new_state), metrics)

  def initialize_model(self) -> Tuple[TrainState, int, nn.Module, PRNGKeys]:
    """Initialize the model and/or load it from a checkpoint.

    Returns:
      (tstate: TrainState,  -- The parameters and state for the the model.
       start_step: int,     -- The step number, when restoring from checkpoint.
       imodel: nn.Module,   -- A model object (created with mode "init").
       rngs: PRNGkeys)      -- Initial random numbers.
    """

    # Set up random number generators.
    # ---------------------------------
    logging.info("==== Training loop: initializing model ====")
    logging.info("Process %d of %d", jax.process_index(), jax.process_count())
    logging.info("Local device count = %d", jax.local_device_count())
    logging.info("Number of replicas = %d",
                 jax.process_count() * jax.local_device_count())
    logging.info("Using random number seed %d", self.random_seed)

    prng = jax.random.PRNGKey(self.random_seed)
    prng, init_rng = jax.random.split(prng)

    # Grab rngs, which provide different random numbers for each replica.
    if self.replicate_mode:
      prngs = jax.random.split(prng, jax.local_device_count())
    else:
      prngs = prng
    del prng

    # Create a dictionary of prng keys for initialization.
    rng_key_names_init = list(self.rng_key_names) + ["params"]
    init_rngs = jax.random.split(init_rng, len(rng_key_names_init))
    init_rngs = {key: init_rngs[i] for i, key in enumerate(rng_key_names_init)}
    del init_rng

    # Build Model
    # -------------------------------------------------------------------------
    logging.info("Initializing the model.")

    # Create a model, which will be used to initialize trainable parameters.
    imodel = self.model_definition(mode="init")

    # The init function will lazily initialize the model, given a fake input.
    # It returns initialized variables, without doing a fwd pass.
    model_init_fn = jax.jit(imodel.init)
    variables = model_init_fn(init_rngs, imodel.get_fake_input())

    # Split variables into trainable and non-trainable sets.
    mstate, params = variables.pop("params")
    del variables  # Delete to avoid wasting resources.

    # Create an optimizer for params.
    optimizer_def = self.optimizer_factory.create_optimizer_def()
    optimizer = optimizer_def.create(params)

    # tstate holds the full training state of the model.
    tstate = TrainState(optimizer, mstate)
    if self.print_variables:
      logging.info("params = %s", tstate.optimizer.target)
      logging.info("state = %s", tstate.state)

    # Load a pre-trained model or restore it from checkpoint.
    if self.workdir or self.load_dir:
      restore_checkpoints = self.restore_checkpoints
    else:
      restore_checkpoints = False

    start_step = 0
    if restore_checkpoints:
      tstate = self.restore_checkpoint(tstate)
      start_step = int(tstate.optimizer.state.step)

    # Log info on trainable parameters (before replicating them).
    self._write_parameter_info(tstate)
    # raise ValueError("That's all folks!")

    # Replicate the training state across local devices.
    if self.replicate_mode:
      tstate = jax_utils.replicate(tstate)

    return (tstate, start_step, imodel, prngs)

  def restore_checkpoint(self, train_state: TrainState) -> TrainState:
    """Load a pre-trained model or restore it from a checkpoint."""

    # Figure out if we have an existing checkpoint.
    if not self.workdir:
      logging.info("No working directory specified.")
      existing_checkpoint = False
    elif not gfile.exists(self.workdir):
      logging.info("No existing checkpoint directory %s", self.workdir)
      existing_checkpoint = False
    elif not gfile.isdir(self.workdir):
      raise ValueError(f"workdir {self.workdir} must be a directory.")
    else:
      ckpath = checkpoints.latest_checkpoint(self.workdir, "checkpoint_")
      if ckpath:
        logging.info("Found existing checkpoint in %s", self.workdir)
        existing_checkpoint = True
      else:
        logging.info("No existing checkpoint in %s", self.workdir)
        existing_checkpoint = False

    # If any checkpoints exist in workdir, then use those first.
    # This will ensure that the task will restore properly if it's preempted.
    if existing_checkpoint:
      logging.info("Restoring model from last checkpoint %s:", self.workdir)
      load_dir = self.workdir
    elif self.load_dir:
      logging.info("Loading pre-trained model from %s:", self.load_dir)
      load_dir = self.load_dir
    else:
      logging.warning("Unable to load model.")
      return train_state
    loaded_train_state = checkpoints.restore_checkpoint(load_dir, train_state)
    step = int(loaded_train_state.optimizer.state.step)
    self.post_load_checkpoint_fn(load_dir, step)

    if self.restore_state_variables:
      # Restore complete state.
      logging.info("Restoring all variables and state.")
      train_state = loaded_train_state
      del loaded_train_state
    else:
      # Restore trainable variables, but not other state.
      logging.info("Only restoring trainable parameters.")
      train_state = TrainState(loaded_train_state.optimizer, train_state.state)
      del loaded_train_state

    return train_state

  def save_checkpoint(self, tstate: TrainState, step: int,
                      param_summary: Optional[MetricsSummary]):
    """Save a checkpoint with the model state.

    Args:
      tstate: The training state.
      step: The current step number.
      param_summary: Optional metrics summary to write parameter statistics.
    """

    logging.info("Saving checkpoint in directory %s", self.workdir)
    if self.replicate_mode:
      save_state = jax_utils.unreplicate(tstate)
    else:
      save_state = tstate
    checkpoints.save_checkpoint(self.workdir, save_state, step)

    # While we're at it, record distributions of trainable parameters.
    if param_summary is not None:
      logging.info("Recording parameter distributions.")
      params_dict = jax.device_get(
          _flatten_dict_string_keys(save_state.optimizer.target))
      param_distribs = self._compute_parameter_distributions(params_dict)
      param_summary.add(param_distribs)

  def create_training_task(self, mode: str, imodel: nn.Module, prngs: PRNGKeys,
                           writers: Dict[str, MetricWriter]) -> TrainingTask:
    """Create a new TrainingTask for the given mode.

    Args:
      mode: The mode for the task, e.g. "train", "test", "generate".
      imodel: The model object from initialize_model.
      prngs: The PRNGKeys from initialize_model.
      writers: A dictionary of summary writers.

    Returns:
      A TrainingTask object.
    """

    logging.info("Training loop: creating task for mode %s", mode)
    if self.use_separate_metric_directories:
      prefix = ""
    else:
      prefix = mode

    if mode == "train":
      ds = self.get_training_dataset_iterator
    elif mode == "test":
      ds = self.get_test_dataset_iterator
    else:
      ds = None

    # We summarize metrics over multiple training steps.
    # These types control how the summary is computed.
    metric_summary_ops = {
        "step_time": "mean",
        "learning_rate": "last",
        **imodel.metrics_summary_operations(aggregate_over="steps")
    }
    summary = self.metrics_summary_factory(metric_summary_ops)
    extra_summary = self.metrics_summary_factory({})
    summary_writer = self._get_summary_writer(mode, writers)

    return TrainingTask(
        mode=mode,
        dataset=ds,
        step_function=self._compile_step_function(mode),
        prng_keys=prngs,
        summary=summary,
        extra_summary=extra_summary,
        summary_writer=summary_writer,
        summary_prefix=prefix,
        # --- options ---
        replicate_mode=self.replicate_mode,
        print_input_every_steps=self.print_input_every_steps,
        pretty_print_input_function=self.pretty_print_input_function,
        process_summaries_function=self.process_summaries_function,
        extra_summaries_function=self.extra_summaries_fn)

  def train(self):
    """Runs the training and evaluation loop."""

    # The master process saves checkpoints and summaries to disk.
    is_master_process = jax.process_index() == 0
    if self.workdir:
      save_checkpoints = self.save_checkpoints
    else:
      save_checkpoints = False

    # --- Create and initialize the model. ---
    (tstate, start_step, imodel, prngs) = self.initialize_model()

    # Log experiment hyper-parameters.
    writers = {}
    train_writer = self._get_summary_writer("train", writers)
    if start_step == 0:
      self._write_config(train_writer)

    # Additional summary objects.
    param_summary = self.metrics_summary_factory({})  # Parameter statistics.

    # --- Create task objects for test, train, and generate. ---
    tasks = {}
    train_task = self.create_training_task("train", imodel, prngs, writers)
    tasks["train"] = train_task

    if (self.get_test_dataset_iterator is not None and
        self.test_every_steps != 0):
      test_task = self.create_training_task("test", imodel, prngs, writers)
      tasks["test"] = test_task
      if self.generate_every_steps != 0:
        gen_task = self.create_training_task("generate", imodel, prngs,
                                             writers)
        tasks["generate"] = gen_task

    # Register any additional actions.
    register_interstep_callbacks()

    # Main Training Loop
    # --------------------------------------------------------------------------
    logging.info("==== Training loop: starting main loop ====")
    with metric_writers.ensure_flushes(*writers.values()):
      for step in range(start_step, self.num_steps):
        # Log status every so often to monitor progress.
        if should_run(step, self.status_every_steps):
          logging.info("Step: %d", step)

        # Train.
        train_x = train_task.get_next_input()
        (tstate, _) = train_task.run_step(tstate, train_x, step)
        run_interstep_callbacks("train", step)
        del train_x

        # Test.
        if should_run(step, self.test_every_steps):
          if self.num_test_steps > 1:
            logging.info("Test cycle: %d iterations.", self.num_test_steps)
          for sub_step in range(0, self.num_test_steps):
            test_x = test_task.get_next_input()

            # TODO(delesley): This is an ugly hack to run generate steps.
            # Run a generate step using test data.
            # Generate is run just *before* the last test iteration.
            if ((sub_step == self.num_test_steps - 1) and
                should_run(step, self.generate_every_steps)):
              logging.info("Generate cycle.")
              (tstate, _) = gen_task.run_step(tstate, test_x, step)
              run_interstep_callbacks("generate", step)

            (tstate, _) = test_task.run_step(tstate, test_x, step,
                                             sub_step=sub_step)
            run_interstep_callbacks("test", step, sub_step)
          del test_x

        # --- Save checkpoints on the master host. ---
        is_last_step = (step == self.num_steps - 1)
        checkpoint_current_step = (
            save_checkpoints and
            (should_run(step, self.checkpoint_every_steps) or is_last_step))
        if checkpoint_current_step:
          if is_master_process:
            self.save_checkpoint(tstate, step, param_summary)
          self.post_save_checkpoint_fn(self.workdir, step)

        # --- Flush summaries to disk. ---
        if should_run(step, self.log_every_steps):
          for tsk in tasks.values():
            tsk.flush(step)
          param_summary.write(train_writer, step, prefix="params")

    logging.info("Training Finished.")
    if self.replicate_mode:
      tstate = jax_utils.unreplicate(tstate)
    if self.print_variables:
      logging.info("params = %s", tstate.optimizer.target)
      logging.info("state = %s", tstate.state)

  def _compile_step_function(self, mode: str) -> StepFunction:
    """Compile a step function (training or test)."""

    # Create a model object, and a step function that is a closure over the
    # object.  Flax modules are supposed to be "stateless", in that all state
    # is contained the TrainState object that is passed as an input parameter.
    # However, creating the model object may involve allocating expensive
    # data structures, or launching processes, and should only be done once.
    model = self.model_definition(mode=mode)
    if mode == "train":
      step_fn = functools.partial(self.train_step, model)
    else:
      step_fn = functools.partial(self.other_step, model)

    if self.replicate_mode:
      assert not self.trace_debug_mode
      logging.info("Compiling mode %s with pmap.", mode)
      p_fn = jax.pmap(step_fn, donate_argnums=(0,), axis_name="batch")
    elif self.trace_debug_mode:
      logging.info("Compiling mode %s with trace_debug.", mode)
      p_fn = step_fn
    else:
      logging.info("Compiling mode %s with jit.", mode)
      p_fn = jax.jit(step_fn, donate_argnums=(0,))
    return p_fn

  def _get_summary_writer(self, mode: str,
                          writers: Dict[str, MetricWriter]) -> MetricWriter:
    """Create a summary writer for the given mode.

    Args:
      mode: the mode for the summaries, e.g. "test", "train"
      writers: a dictionary which caches previously-created writers.

    Returns:
      A writer for the given mode.
    """

    if self.use_separate_metric_directories:
      # Create a separate writer & directory for each mode.
      w_mode = mode
      summary_dir = os.path.join(self.workdir, mode)
    else:
      # Create a single default writer for all modes.
      w_mode = "train"
      summary_dir = self.workdir

    if w_mode in writers:
      # Return previously created and cached writer.
      logging.info("Returning cached summary writer (%s) for mode %s",
                   w_mode, mode)
      return writers[w_mode]

    if not self.workdir:
      # No working directory, so log only.
      logging.info("Creating logging writer (%s) for mode %s", w_mode, mode)
      writer = metric_writers.LoggingWriter()
    else:
      # Create a new writer for workdir.
      # Only the master will actually write summaries to workdir.
      logging.info("Creating summary writer (%s) for mode %s in directory %s",
                   w_mode, mode, summary_dir)
      is_master = jax.process_index() == 0
      gfile.makedirs(summary_dir)
      writer = metric_writers.create_default_writer(summary_dir,
                                                    just_logging=not is_master)
    writers[w_mode] = writer
    return writer

  def _write_config(self, writer):
    """Write the configuration file to the working directory."""

    is_master = jax.process_index() == 0
    config_str = gin.operative_config_str()
    logging.info("Gin config: \n%s", config_str)

    # Write configuration to workdir.
    if is_master and self.workdir:
      config_file_name = os.path.join(self.workdir, "config.gin")
      with gfile.GFile(config_file_name, "w") as f:
        f.write(config_str)

    # Write config string text to tensorboard.
    writer.write_texts(0, {"config": gin.markdown(config_str)})

  def _write_parameter_info(self, tstate: TrainState):
    """Write information on state and trainable parameters to the log."""

    # Write information on parameters to log file.
    params_dict = _flatten_dict_string_keys(tstate.optimizer.target)
    total_nparams = 0
    for (k, v) in params_dict.items():
      nparams = np.prod(v.shape)
      total_nparams += nparams
      logging.info("parameter: %s, shape %s, size %d", k, v.shape, nparams)
    logging.info("Total parameters: %d", total_nparams)

    # Write information on state variables to log file.
    state_dict = _flatten_dict_string_keys(tstate.state)
    state_size = 0
    total_state = 0
    for (k, v) in state_dict.items():
      if hasattr(v, "shape"):
        state_size = np.prod(v.shape)
        total_state += state_size
        logging.info("state: %s, shape %s, size %d", k, v.shape, state_size)
      else:
        # Some other stuff may be stored in the state.
        logging.info("state: %s [unknown]", k)
    logging.info("Total state size: %d", total_state)

  def _compute_parameter_distributions(self, params_dict):
    """Compute info on distributions of parameters."""

    scalar_params_dict = {}
    for (k, v) in params_dict.items():
      # Convert from bfloat16, which crashes when serializing a NaN.
      v = np.asarray(v, dtype=jnp.float32)
      scalar_params_dict[k + "_mean"] = np.mean(v)
      scalar_params_dict[k + "_stddev"] = np.std(v)
      # scalar_params_dict[k + "_min"] = np.min(v)
      # scalar_params_dict[k + "_max"] = np.max(v)
    return scalar_params_dict


def _flatten_dict_string_keys(params):
  """Flattens a nested dictionary to have string keys and '/' separators."""
  return {"/".join(k): v for k, v in flatten_dict(unfreeze(params)).items()}
