# Copyright 2025 Google.
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

import os
from typing import (Any, Callable, Dict, Optional, Sequence, Tuple)

from absl import logging
from clu import metric_writers
from flax import struct
import gin
import jax
import jax.numpy as jnp
import  metrics_summary
import  model_info
import  optimizer_config as opt_config
import  training_task
import tensorflow.compat.v2 as tf


ModelInfo = model_info.ModelInfo
ModelDefinition = model_info.ModelDefinition
DatasetIteratorFunction = model_info.DatasetIteratorFunction
TrainState = model_info.TrainState
LearningRateScheduleFn = opt_config.LearningRateScheduleFn

Optimizer = model_info.Optimizer
PRNGKey = model_info.PRNGKey
TrainingTask = training_task.TrainingTask
StepFunction = training_task.StepFunction
Metrics = training_task.Metrics
MetricWriter = metric_writers.MetricWriter
MetricsSummary = metrics_summary.MetricsSummary


gfile = tf.io.gfile
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
  # Takes a single argument 'mode', which can be "test", "train", or "generate".
  model_definition: ModelDefinition = gin.REQUIRED

  # Set one of either batch_size or batch_size_per_replica.
  batch_size: int = 0                  # Total batch size over all replicas.
  batch_size_per_replica: int = 0      # Batch size per replica.

  # Returns an iterator over the data set.
  get_dataset_iterator_function: DatasetIteratorFunction = gin.REQUIRED

  workdir: str = ""                    # Working directory for checkpoints.
  load_dir: str = ""                   # Optional directory to load model.
  num_steps: int = 100000              # Number of steps to train.
  status_every_steps: int = 10         # Log step number every N steps.
  log_every_steps: int = 100           # Log scalar data every N steps.
  test_every_steps: int = 10           # Test model every N steps.
  num_test_steps: int = 1              # Number of iterations to test.
  reset_test_task: bool = True         # Reset test task before each eval.
                                       # Reduces noise but can skew the test
                                       # distribution.
  generate_every_steps: int = 1000     # Generate examples every N steps.
  print_input_every_steps: int = 1000  # Print example data every N steps.
  parameter_metrics_every_steps: int = 2000  # Distributions of parameters.

  save_checkpoints: bool = True        # Save training checkpoints
  checkpoint_every_steps: int = 5000   # Save checkpoints every N steps.
  keep_saving_configs: bool = False    # Save gin config when saving chkpts.

  # Record metrics for "train", "test", etc. in separate directories.
  # Otherwise they will be saved with separate prefixes.
  use_separate_metric_directories: bool = True

  # Optimizer options.
  optimizer_factory: opt_config.OptimizerConfig = gin.REQUIRED
  learning_rate_schedule: LearningRateScheduleFn = opt_config.lr_cosine_decay

  # Maximum steps for the LR schedule.  Zero means use num_steps.
  max_scheduled_steps: int = 0
  warmup_steps: int = 1000               # Number of warmup steps.
  learning_rate_multiplier: float = 1.0  # Used to scale the learning rate.

  random_seed: int = 42                  # Initial random seed.
  # Names of random number generators used by the model.
  rng_key_names: Optional[Sequence[str]] = ("dropout",)

  # Function to compute additional summary information.
  # Takes a MetricsSummary object and a mode string (e.g. "test") as arguments,
  # returns a MetricsSummary object.
  process_summaries_function: Optional[Callable[[Any, str], Any]] = None

  # Function to pretty print the input for each training step.
  pretty_print_input_function: Optional[Callable[[Any], Any]] = None

  # Classes to use for summarizing metrics.
  metrics_summary_factory: Callable[[], MetricsSummary] = MetricsSummary
  extra_summaries_fn: training_task.ExtraSummariesFunction = None

  def learning_rate_schedule_fn(self, step: jnp.ndarray) -> jnp.ndarray:
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

    base_lrate = float(self.optimizer_factory.learning_rate())
    lr_multiplier = float(self.learning_rate_multiplier)

    # Linear increase in learning rate up to warmup_steps.
    if self.warmup_steps > 0:
      warmup_steps = float(self.warmup_steps)
      lr_warmup_ramp = jnp.minimum(step, warmup_steps) / warmup_steps

      # Hold step at a constant value during the warmup period.
      # Required for some schedules, like rsqrt_decay.
      step = jnp.maximum(step, warmup_steps)
      del warmup_steps
    else:
      lr_warmup_ramp = 1.0

    # Get the scheduled learning rate.
    lrate = self.learning_rate_schedule(step, max_steps)

    # Multiply lrate by the base, warmup and multiplier factors.
    lrate = lrate * base_lrate * lr_warmup_ramp * lr_multiplier
    return jnp.asarray(lrate, dtype=jnp.float32)

  def create_model_info(self) -> ModelInfo:
    """Initialize the model and/or load it from a checkpoint."""

    mdl_info = ModelInfo(
        model_definition=self.model_definition,
        optimizer_def=self.optimizer_factory.create_optimizer_def(),
        workdir=self.workdir,
        load_dir=self.load_dir,
        rng_key_names=self.rng_key_names,
        batch_size=self.batch_size,
        batch_size_per_replica=self.batch_size_per_replica,
        random_seed=self.random_seed)

    logging.info("Initialized model.")
    return mdl_info

  def save_checkpoint(self, mdl_info: ModelInfo,
                      tstate: TrainState, step: int):
    """Save checkpoint."""
    mdl_info.save_checkpoint(tstate, step)

  def create_training_task(self, mode: str, mdl_info: ModelInfo,
                           writers: Dict[str, MetricWriter]) -> TrainingTask:
    """Create a new TrainingTask for the given mode.

    Args:
      mode: The mode for the task, e.g. "train", "test", "generate".
      mdl_info: A ModelInfo object from initialize_model().
      writers: A dictionary of summary writers.

    Returns:
      A TrainingTask object.
    """

    logging.info("Training loop: creating task for mode %s", mode)
    if self.use_separate_metric_directories:
      prefix = ""
    else:
      prefix = mode

    # Get an iterator over the data set for the given mode
    if mode == "train" or mode == "test":
      ds = self.get_dataset_iterator_function(mode, mdl_info.num_shards,
                                              mdl_info.shard_id,
                                              mdl_info.batch_size_per_shard)
    else:
      ds = None

    # Get a step function for the given mode.
    if mode == "train":
      step_fn = mdl_info.train_step_fn(self.learning_rate_schedule_fn)
    else:
      step_fn = mdl_info.other_step_fn(mode)

    summary = self.metrics_summary_factory()
    extra_summary = self.metrics_summary_factory()
    summary_writer = self._get_summary_writer(mode, writers)

    return TrainingTask(
        mode=mode,
        dataset=ds,
        step_function=step_fn,
        mdl_info=mdl_info,
        summary=summary,
        extra_summary=extra_summary,
        summary_writer=summary_writer,
        summary_prefix=prefix,
        # --- options ---
        print_input_every_steps=self.print_input_every_steps,
        pretty_print_input_function=self.pretty_print_input_function,
        process_summaries_function=self.process_summaries_function,
        extra_summaries_function=self.extra_summaries_fn)

  def train(self):
    """Runs the training and evaluation loop."""

    # --- Create and initialize the model. ---
    mdl_info = self.create_model_info()
    tstate = mdl_info.initialize_model()

    # Create summary writer for train mode.
    writers = {}
    train_writer = self._get_summary_writer("train", writers)

    # Additional summary objects.
    param_summary = self.metrics_summary_factory()  # Parameter statistics.

    # --- Create task objects for test, train, and generate. ---
    tasks = {}
    train_task = self.create_training_task("train", mdl_info, writers)
    tasks["train"] = train_task

    test_task = None
    gen_task = None
    if self.test_every_steps != 0:
      test_task = self.create_training_task("test", mdl_info, writers)
      tasks["test"] = test_task
      if self.generate_every_steps != 0:
        gen_task = self.create_training_task("generate", mdl_info, writers)
        tasks["generate"] = gen_task

    # Register any additional actions.
    register_interstep_callbacks()

    # Log experiment hyper-parameters.
    self._write_config(train_writer, mdl_info.step)

    # Main Training Loop
    # -------------------------------------------------------------------------
    logging.info("==== Training loop: starting main loop ====")
    with metric_writers.ensure_flushes(*writers.values()):
      for step in range(mdl_info.step, self.num_steps):
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
          assert test_task is not None
          if self.num_test_steps > 1:
            logging.info("Test cycle: %d iterations.", self.num_test_steps)
          if self.reset_test_task:
            test_task.reset_dataset()
          test_x = None
          for sub_step in range(0, self.num_test_steps):
            test_x = test_task.get_next_input()

            # TODO(delesley): This is an ugly hack to run generate steps.
            # Run a generate step using test data.
            # Generate is run just *before* the last test iteration.
            if ((sub_step == self.num_test_steps - 1) and
                should_run(step, self.generate_every_steps)):
              assert gen_task is not None
              logging.info("Generate cycle.")
              (tstate, _) = gen_task.run_step(tstate, test_x, step)
              run_interstep_callbacks("generate", step)

            (tstate, _) = test_task.run_step(tstate, test_x, step,
                                             sub_step=sub_step)
            run_interstep_callbacks("test", step, sub_step)
          del test_x

        # --- Flush summaries to disk. ---
        # Only the master process will write to disk; see _get_summary_writer.
        if should_run(step, self.log_every_steps):
          for tsk in tasks.values():
            tsk.flush(step)

        # --- Log distributions of trainable parameters. ---
        if should_run(step, self.parameter_metrics_every_steps):
          param_metrics = mdl_info.compute_parameter_metrics(tstate)
          param_summary.merge(param_metrics)
          param_summary.write(train_writer, step, prefix="params")
          del param_metrics

        # --- Save checkpoints on the master host. ---
        is_last_step = (step == self.num_steps - 1)
        if should_run(step, self.checkpoint_every_steps) or is_last_step:
          self.save_checkpoint(mdl_info, tstate, step)
          if self.keep_saving_configs:
            self._write_config(train_writer, step=step)
      # end of with statement
    logging.info("Training Finished.")

    # Return current model state.
    # Can be used in colab to step the model forward from this point.
    return (mdl_info, tstate, tasks)

  def _get_summary_writer(self, mode: str,
                          writers: Dict[str, MetricWriter]) -> MetricWriter:
    """Create a summary writer for the given mode.

    Note that only the master process will write to disk.

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

  def _write_config(self, writer, step: int):
    """Write the configuration file to the working directory."""

    is_master = jax.process_index() == 0
    config_str = gin.operative_config_str()
    # logging.info("Gin config: \n%s", config_str)  # logged in launcher.py.

    # Write config string text to tensorboard.
    writer.write_texts(step, {"config": gin.markdown(config_str)})

    # Write configuration to workdir.
    if is_master and self.workdir:
      logging.info("Writing config.gin")
      config_file_name = os.path.join(self.workdir, f"config_{step}.gin")
      with gfile.GFile(config_file_name, "w") as f:
        f.write(config_str)

