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

"""ModelInfo handles initialization and checkpointing of models.
"""

import dataclasses
import enum
import functools
import math
import os
import re
import typing
from typing import Any, Callable, Collection, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

from absl import logging
import flax
from flax import jax_utils
from flax.core import scope as flax_scope
from flax.training import checkpoints as flax_checkpoints
import gin
import jax
from jax.experimental import checkify
from jax.experimental import multihost_utils
import jax.numpy as jnp
import  metrics_summary
import  model_base
import  optimizer_config as opt_config
import  train_state
import numpy as np
import orbax.checkpoint as orbax


import tensorflow.compat.v2 as tf


# A function which creates a model object, given a mode string.
# A function from (mode:str) -> model_base.ModelBase
ModelDefinition = Any


# Function which returns an iterator over the data set.
# The dataset will be sharded over multiple parallel processes, and batched
# over multiple replicas (local devices) within each process.
# See ModelInfo, below.
#
# It takes four arguments:
#   mode: str                  -- e.g. "test", or "train"
#   num_shards: int            -- Total number of shards in the data set.
#   shard_id: int              -- The ID of the shard for this process.
#   batch_size_per_shard: int  -- The batch size for each shard.
Dataset = Callable[[], Iterator[Any]]
DatasetIteratorFunction = Callable[[str, int, int, int], Dataset]


Array = model_base.Array
Metrics = model_base.Metrics
ModelBase = model_base.ModelBase
PRNGKey = model_base.PRNGKey
FrozenVariableDict = model_base.FrozenVariableDict

Optimizer = opt_config.Optimizer
OptimizerDef = opt_config.OptimizerDef
TrainState = train_state.TrainState
MetricsSummary = metrics_summary.MetricsSummary

HardwareMesh = Any
LogicalAxisRules = Any
PjitPartitioner = Any


CheckpointFn = Callable[[str, int], None]
StepFuncPrng = Callable[[TrainState, Any, PRNGKey], Tuple[TrainState, Metrics]]
StepFunction = Callable[[TrainState, Any], Tuple[TrainState, Metrics]]
LrScheduleFn = Callable[[Array], Array]  # Learning rate schedule fn


gfile = tf.io.gfile
unfreeze = flax.core.unfreeze
flatten_dict = flax.traverse_util.flatten_dict


@gin.constants_from_enum
class CheckifyMode(frozenset[checkify.ErrorCategory], enum.Enum):
  NONE = frozenset()
  USER_CHECKS = checkify.user_checks
  NAN_CHECKS = checkify.nan_checks
  INDEX_CHECKS = checkify.index_checks
  DIV_CHECKS = checkify.div_checks
  FLOAT_CHECKS = checkify.float_checks
  AUTOMATIC_CHECKS = checkify.automatic_checks
  ALL_CHECKS = checkify.all_checks


@gin.configurable
@dataclasses.dataclass(slots=True)
class LoadPretrainedConfig:
  r"""Configuration data for loading and transforming pretrained models.

  Attributes:
    populate_key_from_value: Map "names of variables I want to populate" to
      "where do I populate them from". Example:
          {r"(.*)/submodule_name/(.*)": r"\1/\2"}
      will load the pretrained module into a submodule named `submodule_name`.
    load_same_named_variables: If True, variables not mentioned as keys in
      populate_key_from_value will be loaded from checkpointed variables of the
      same name (when the latter exist). Thus, a True value functions like
          {r"(.*)": r"\1"}
      if we could guarantee this would be applied only after all the other
      regexes have failed to match.
      If False, variables not mentioned as keys will not be loaded but left in
      the untrained initial state.
    reset_optimizer: If True, the step, momentum, etc are reset upon loading a
      pretrained model.
    load_dir: Path to the checkpoint to load.
    next_step: Step to start with if this is the last LoadPretrainedConfig in
      the sequence.
  """
  populate_key_from_value: Dict[str, str] = dataclasses.field(
      default_factory=dict
  )
  load_same_named_variables: bool = True
  reset_optimizer: bool = False
  load_dir: str = ""
  # TODO(cstaats): Fix loading step from checkpoint directly.
  # (The following field is a quick workaround to get the step set correctly.)
  next_step: Optional[int] = None


def _extract_globals(tstate: TrainState) -> Dict[str, Any]:
  return {"step": tstate.step}


# Default CheckpointFn
# These are additional functions for managing checkpoints.
def _do_nothing(load_dir: str, step: int) -> None:
  del load_dir
  del step
  return None


@gin.configurable
class ModelInfo:
  """Handles model initialization, replication, loading and saving checkpoints.

  The ModelInfo object initializes a model, and holds various information
  about it that is used for replication, training, and inference.

  For distributed training, there are multiple processes, and each process has
  multiple devices. Both devices and processes are split over two logical axes:
  batch, and model.  (Each process is a tile of devices in the device grid.)
  Model parameters are sharded/replicated across devices, while the data
  pipeline is sharded/replicated across processes.

  Each process pulls from the data pipeline, and sends the data to its local
  devices.  The number of shards in the data set is the number of processes
  (i.e. machines) along the batch axis.  E.g. assume 8 processes are arranged
  in a 4x2 grid, so batch x model = 4x2 with respect to the processes.  Then
  the dataset will be broken into 4 shards (batch axis), and each shard will
  be replicated across two processes (model axis).

  Attributes:
    step:                        Step of last saved or loaded checkpoint.
    workdir:                     Directory for checkpoints and metrics.
    rng_key_names:               Names of flax random number generators.
    replicated_prng:             Replicated PRNGKey for calling a step_fn.
    metrics_summary_operations:  To create a MetricsSummary object.

    batch_size:                  Total batch size over all replicas.
    batch_size_per_shard:        batch_size // num_shards
    batch_size_per_replica:      bach_size // num_batch_devices
    num_batch_devices:           Number of devices along the batch dimension.
    num_shards:                  Number of processes along the batch dimension.
    shard_id:                    The ID of the dataset shard for this process.

  Important methods:
    initialize_model: Initialize or load parameters to get a valid TrainState.
    save_checkpoint:          Save TrainState to a checkpoint.
    copy_input_to_devices:    Prepare local data before calling a step function.
    read_metrics_from_device: Read back computed metrics as a local numpy array.
    train_step_fn:            Get a step function for training.
    other_step_fn:            Get a step function for testing or inference.
  """

  def __init__(
      self,
      model_definition: ModelDefinition,
      optimizer_def: OptimizerDef,
      workdir: str = "",
      load_dir: str = "",
      rng_key_names: Optional[Sequence[str]] = ("dropout",),
      batch_size: int = 0,
      batch_size_per_replica: int = 0,
      random_seed: int = 42,
      # --- gin configurable parameters ---
      replicate_mode: str = "pmap",
      pjit_partitioner_factory: Any = gin.REQUIRED,
      model_parallel_submesh: Optional[HardwareMesh] = None,
      trace_debug_mode: bool = False,
      restore_state_variables: bool = True,
      post_load_checkpoint_fn: CheckpointFn = _do_nothing,
      post_save_checkpoint_fn: CheckpointFn = _do_nothing,
      verbose_output: bool = False,
      checkify_mode: CheckifyMode = CheckifyMode.NONE,
      load_pretrained_config: Sequence[LoadPretrainedConfig] = gin.REQUIRED,
  ):
    """Initialize the model and/or load it from a checkpoint.

    Args:
      model_definition: A function that creates a model in a given mode,
          where mode is, e.g. "train", "test", or "generate".
      optimizer_def: An optimizer definition.
      workdir: The working directory for checkpoints.
      load_dir: An (optional) directory to load a pre-trained model.
      rng_key_names: Names of flax random number generators.
      batch_size: The total batch size, over all replicas.
      batch_size_per_replica: The batch size per replica, which can be
                              specified as an alternative to batch_size.
      random_seed: The seed used for initialization, and subsequent prngs.

      replicate_mode: one of {"pmap", "pjit", "none"}.
      pjit_partitioner_factory:  Should be t5x.partitioning.PjitPartitioner
          when using pjit.  However, this must be set by gin, not passed as a
          normal argument, because PjitPartitioner is configured using gin
          dynamic_registration, and cannot be called directly.
      model_parallel_submesh: is a 4-tuple that specifies the `(x, y, z, c)`
          submesh model-parallel device tile when using pjit.
          See t5x PjitPartitioner for details.

      trace_debug_mode: If True, run model in interpreted mode.
      restore_state_variables: Restore non-trainable cache from checkpoint.
      post_load_checkpoint_fn: Function to call after loading checkpoint.
      post_save_checkpoint_fn: Function to call after saving checkpoint.
      verbose_output: Log additional information for debugging.
      checkify_mode: Use jax.experimental.checkify to allow dynamic assertions
          and/or (very expensively) isolating floating-point errors.
      load_pretrained_config: Configuration for loading and transforming a
          pretrained checkpoint. Settings do not apply when restoring an
          earlier checkpoint of the current training run.
    """

    if replicate_mode not in {"pmap", "pjit", "none"}:
      raise ValueError(f"replicate_mode {replicate_mode} must be one of" +
                       "{'pmap', 'pjit', 'none'}.")

    self.step = 0
    self.workdir = workdir
    self.rng_key_names = rng_key_names
    self.metrics_summary_operations = None  # Set in initialize_model.

    self._model_definition = model_definition
    self._optimizer_def = optimizer_def
    self._load_dir = load_dir
    self._replicate_mode = replicate_mode
    self._pjit_partitioner_factory = pjit_partitioner_factory
    self._trace_debug_mode = trace_debug_mode
    if trace_debug_mode:
      assert self._replicate_mode == "none"
    self._checkify_mode = checkify_mode
    self._load_pretrained_config = load_pretrained_config
    if self._load_dir:
      if len(self._load_pretrained_config) != 1:
        raise ValueError(
            "Bad configuration: the load_dir commandline flag requires exactly"
            " one LoadPretrainedConfig.")
      self._load_pretrained_config[0].load_dir = self._load_dir
    else:
      # If we kept a load_pretrained_config for a load_dir, but there's no
      # load_dir, delete the load_pretrained_config.
      if (len(self._load_pretrained_config) == 1
          and not self._load_pretrained_config[0].load_dir):
        self._load_pretrained_config = []

    self._restore_state_variables = restore_state_variables
    self._post_load_checkpoint_fn = post_load_checkpoint_fn
    self._post_save_checkpoint_fn = post_save_checkpoint_fn
    self._verbose_output = verbose_output

    # Set in precompile; declared here for readability.
    self._tstate_partition_spec = None
    self._input_partition_spec = None

    # Set in _init_checkpoint_manager; declared here for readability.
    self._orbax_checkpoint_manager = None
    self._orbax_save_args = None
    self._orbax_restore_args = None

    # Compiled on demand.
    self._compute_parameter_distributions_fn = None

    # Set up random number generators.
    # Each process will initialize the model using the same random seed.
    # Step functions also use deterministic random seeds.
    prng = jax.random.PRNGKey(random_seed)
    (prng, init_prng) = jax.random.split(prng)
    self._init_prng = init_prng   # Used in _init_from_scratch
    self.replicated_prng = self._replicate_prngs(prng)   # Handed to step_fn

    # Get a T5X partitioner to help with partitioning.
    if self._replicate_mode == "pjit":
      if model_parallel_submesh is None:
        raise ValueError("Must specify model_parallel_submesh when using pjit.")
      # logical_axis_rules must be gin-configured for PjitPartitioner.
      self._pjit_partitioner = self._pjit_partitioner_factory(
          model_parallel_submesh=model_parallel_submesh)

    # Compute sharding and batch size information.
    self._compute_dataset_sharding()  # Get self.num_batch_devices
    if batch_size_per_replica != 0:
      if batch_size != 0:
        assert batch_size == (self.num_batch_devices * batch_size_per_replica)
      else:
        batch_size = self.num_batch_devices * batch_size_per_replica

    if batch_size % self.num_batch_devices != 0:
      raise ValueError("Batch size %d must be divisible by mesh batch dim %d" %
                       (batch_size, self.num_batch_devices))
    self.batch_size = batch_size
    self.batch_size_per_replica = batch_size // self.num_batch_devices
    self.batch_size_per_shard = batch_size // self.num_shards

    logging.info("==== Creating ModelInfo ====")
    logging.info("Process %d of %d", jax.process_index(), jax.process_count())
    logging.info("Local device count = %d", jax.local_device_count())
    logging.info("Replicate mode = %s", replicate_mode)
    if self._replicate_mode == "pjit":
      logging.info("Mesh = %s", self._pjit_partitioner.mesh)
    if model_parallel_submesh is not None:
      logging.info("Model parallel submesh = %s", model_parallel_submesh)
    logging.info("Number of replicas = %d", self.num_batch_devices)
    logging.info("Number of shards = %d", self.num_shards)
    logging.info("Using random number seed %d for model", random_seed)
    logging.info("Total batch_size = %d", self.batch_size)
    logging.info("Batch size per replica = %d", self.batch_size_per_replica)
    logging.info("Batch size per shard = %d", self.batch_size_per_shard)

  def initialize_model(self) -> TrainState:
    """Initialize the model, either from scratch or loaded from a checkpoint.

    Returns:
      An initialized TrainState object.
    """
    logging.info("==== Initializing the model. ====")

    # Create a model object for initialization, and get model variables.
    imodel = self._model_definition(mode="init")
    dummy_tstate = self.precompile(imodel)

    # Log info on trainable parameters (before replicating them).
    write_parameter_info(dummy_tstate, self._tstate_partition_spec)  # pytype: disable=wrong-arg-types  # jax-api-types

    # In general, restore_checkpoint needs a fully partitioned tstate,
    # so dummy_tstate is not sufficient.
    # TODO(delesley): Support checkpoint loading from dummy_tstate.
    tstate = self.init_model_from_scratch(imodel)
    self._init_checkpoint_manager(tstate)     # Requires a real tstate.
    r_tstate = self.restore_checkpoint(tstate)  # Requires a real tstate.
    if r_tstate is not None:
      logging.info("initialize_model: Loaded TrainState.")
      tstate = r_tstate

    # TODO(delesley): Add support for a global "step" flax variable.

    # Replicate the training state across local devices, when using pmap.
    if self._replicate_mode == "pmap":
      tstate = _global_or_local_to_pmap(tstate)

    return tstate

  def precompile(self, imodel: ModelBase) -> TrainState:
    """Initialize the model, but return a dummy TrainState with only shape info.

    Args:
      imodel: A model object.

    Returns:
      A TrainState that contains model parameters with shape information, which
      can be used to load checkpoints.
    """

    logging.info("==== Precompiling model. ====")
    if self._replicate_mode in ("pmap", "none"):
      dummy_input = imodel.get_fake_input(self.batch_size_per_replica)
    elif self._replicate_mode == "pjit":
      dummy_input = imodel.get_fake_input(self.batch_size)
    else:
      raise ValueError("Unknown replicate mode: %s" % self._replicate_mode)

    init_fn = self.model_init_fn(imodel)
    dummy_tstate = jax.eval_shape(init_fn, dummy_input)

    if self._replicate_mode == "pjit":
      # Flax will sow parameter partitioning info into dummy_tstate.
      # The pjit_partitioner will map this info onto the configured mesh, to
      # produce partition specs that can be used with pjit.
      self._tstate_partition_spec = (
          self._pjit_partitioner.get_mesh_axes(dummy_tstate))
      self._input_partition_spec = self._pjit_partitioner.data_partition_spec
    else:
      # The orbax checkpointer will still want this information.
      self._tstate_partition_spec = jax.tree.map(lambda x: None, dummy_tstate)
      self._input_partition_spec = None
    return dummy_tstate

  def _call_and_handle_errors(
      self, maybe_checkified: Callable[..., Any], *args, **kwargs
  ) -> Any:
    if self._checkify_mode:
      err, retv = maybe_checkified(*args, **kwargs)
      err.throw()
      return retv
    return maybe_checkified(*args, **kwargs)

  def init_model_from_scratch(self, imodel: ModelBase) -> TrainState:
    """Create and initialize model parameters from scratch.

    Args:
      imodel: A model object.

    Returns:
      An initialized TrainState object.
    """

    logging.info("==== Initializing model parameters from scratch. ====")
    if self._tstate_partition_spec is None:
      raise ValueError("Must call precompile before init_model_from_scratch.")

    init_fn = self.model_init_fn(imodel)
    if self._checkify_mode:
      init_fn = checkify.checkify(init_fn, errors=self._checkify_mode)

    if self._replicate_mode == "pmap" or self._replicate_mode == "none":
      # JIT-compile for single device.
      # Parameters will be replicated later with _replicate_tstate.
      dummy_input = imodel.get_fake_input(self.batch_size_per_replica)
      compiled_init_fn = init_fn if self._trace_debug_mode else jax.jit(init_fn)
      return self._call_and_handle_errors(compiled_init_fn, dummy_input)
    else:
      assert self._replicate_mode == "pjit"
      assert not self._checkify_mode, (
          "Checkify mode with pjit is not currently supported in meliad. If you"
          " want to implement it, see"
          " https://jax.readthedocs.io/en/latest/debugging/checkify_guide.html#pjit"
      )
      # Use pjit to immediately shard parameters over multiple devices.
      dummy_input = imodel.get_fake_input(self.batch_size)
      compiled_init_fn = self._pjit_partitioner.partition(
          init_fn,
          in_axis_resources=None,
          out_axis_resources=self._tstate_partition_spec)
      return compiled_init_fn(dummy_input)

  def model_init_fn(self, imodel: ModelBase) -> Callable[[Any], TrainState]:
    """Returns a function that initializes the model given a dummy input."""
    init_rng_dict = self._get_init_prng_dict(self._init_prng)

    def init_fn(dummy_input: Any):
      model_vars = imodel.model_init(init_rng_dict, dummy_input)
      if self._verbose_output:
        # Log model variables as returned from flax init.
        logging.info("=== model_vars ===")
        for (mk, mv) in flatten_dict_string_keys(model_vars).items():
          logging.info("model_var: %s = %s", mk, mv)
      return self._create_tstate(model_vars)

    return init_fn

  def _create_tstate(self, model_variables: FrozenVariableDict) -> TrainState:
    """Create a new TrainState object."""

    logging.info("Creating TrainState object.")
    # logging.info("model_variables = %r", model_variables)

    if self._replicate_mode == "pjit":
      logging.info("Creating TrainState with pjit partitioner.")
      use_axes = True
    else:
      logging.info("Creating TrainState without partitioning.")
      use_axes = False

    return train_state.TrainState.create(
        self._optimizer_def, model_variables, use_axes=use_axes
    )

  def _reinit_optimizer(self, tstate: TrainState) -> TrainState:
    """Reinitialize optimizer state (e.g. momentum)."""

    logging.info("Reinitializing optimizer.")

    if self._replicate_mode == "pjit":
      raise NotImplementedError("_reinit_optimizer is not implemented for pjit")
    params = tstate.params
    optimizer = self._optimizer_def.create(params)
    return train_state.TrainState(
        optimizer,
        params_axes=None,
        flax_mutables=tstate.flax_mutables,
        flax_mutables_axes=None,
    )

  def restore_checkpoint(self, tstate: TrainState) -> Optional[TrainState]:
    """Load a pre-trained model or restore it from a checkpoint.

    If loading fails, restore_checkpoint will return None.

    If loading succeeds, this will return the successfully loaded TrainState.
    However, depending on configuration, only part of the TrainState may be
    loaded -- e.g. when loading a pre-trained model that only specifies some
    of the parameters.  Any non-loaded parameters will be carried over from
    tstate, so tstate must be a properly initialized TrainState object.

    Args:
      tstate: A TrainState object, which defines the structure to load.

    Returns:
      Either a newly loaded TrainState object, or None.
    """
    logging.info("==== Trying to restore model from checkpoint. ====")

    # If any checkpoints exist in workdir, then use those first.
    # This will ensure that the task will restore properly if it's preempted.
    loaded_dir = ""
    loaded_tstate = None
    if self._checkpoint_exists(self.workdir, "working directory"):
      logging.info("Restoring model from last checkpoint in %s...",
                   self.workdir)
      loaded_dir = self.workdir
      loaded_tstate = self._restore_orbax_checkpoint(loaded_dir, tstate)
      if loaded_tstate is not None:
        logging.info("Successfully restored model from checkpoint %s",
                     self.workdir)

    # Otherwise attempt to load a pre-trained model, if one has been specified.
    # It is possible to load several pre-trained models, and combine them into
    # a larger composite model.
    if loaded_tstate is None:
      logging.info("No existing checkpoint.")
      if self._load_pretrained_config:
        logging.info("Attempting to load pre-trained model(s).")

      for config in self._load_pretrained_config:
        if not config.load_dir:
          raise ValueError(f"load_dir not specified in config {repr(config)}")
        if not self._checkpoint_exists(
            config.load_dir, "loading pretrained model"):
          raise ValueError(f"checkpoint {config.load_dir} does not exist.")
        logging.info("Loading pre-trained model from %s:", config.load_dir)
        loaded_dir = config.load_dir
        logging.info("Loading pre-trained model from %s...", loaded_dir)
        loaded_tstate = self._restore_orbax_checkpoint(
            loaded_dir,
            tstate,
            pretraining_config=config,
        )
        if loaded_tstate is None:
          raise ValueError(
              f"Unable to restore model from checkpoint {config.load_dir}.")
        else:
          logging.info("Successfully loaded pre-trained model.")
        if config.reset_optimizer:
          loaded_tstate = self._reinit_optimizer(loaded_tstate)
        if config.next_step is not None:
          loaded_tstate = loaded_tstate.replace_step(config.next_step)

    # Return immediately on failure.
    if loaded_tstate is None:
      logging.info("Unable to load model.")
      return None

    self.step = int(loaded_tstate.step)
    logging.info("Successfully loaded model.  Next step = %d.", self.step)
    self._post_load_checkpoint_fn(loaded_dir, self.step)

    if self._restore_state_variables:
      # Restore complete state.
      logging.info("Restoring all variables and state.")
      return loaded_tstate
    else:
      # Otherwise restore trainable variables, but not other state.
      logging.info("Only restoring trainable parameters.")
      tstate = tstate.replace_params(loaded_tstate.params)
      return tstate

  def save_checkpoint(self, tstate: TrainState, step: int):
    """Save a checkpoint with the model state.

    Args:
      tstate: The training state.
      step: The current step number.
    """

    if not self.workdir:
      logging.info("No working directory.  Skipping save_checkpoint.")
      return
    if self._replicate_mode == "pmap":
      # Orbax will choke on a pmap-replicated TrainState.
      save_tstate = _pmap_to_global(tstate)
    else:
      save_tstate = tstate

    logging.info("Saving checkpoint for step %d in directory %s",
                 step, self.workdir)
    self.step = step
    self._save_orbax_checkpoint(save_tstate, step)
    # Run any final post-checkpoint tasks.
    self._post_save_checkpoint_fn(self.workdir, step)

  def _init_checkpoint_manager(self, tstate: TrainState):
    """Create a checkpoint manager that can save and restore tstate."""
    if self._tstate_partition_spec is None:
      raise ValueError("Must call precompile before _init_checkpoint_manager.")

    # Follow the t5x convention of saving and restoring the state_dict.
    tstate_dict = tstate.state_dict()
    tstate_mesh_axes_dict = self._tstate_partition_spec.state_dict()
    if self._replicate_mode == "pjit":
      mesh = self._pjit_partitioner.mesh
    else:
      mesh = None

    # Non-partitioned parameters are aggregated into a single save file.
    # Partitioned parameters are stored in separate directories.
    def _save_args(value, mesh_axes):
      del mesh_axes
      dtype = None
      if hasattr(value, "dtype"):
        dtype = value.dtype
      return orbax.type_handlers.SaveArgs(dtype=dtype)

    # Construct partitioning information for all parameters.
    # Partitioned parameters cannot be loaded from the aggregate file.
    def _restore_args(value, mesh_axes):
      nonlocal mesh
      restore_type = type(value)
      if (mesh is not None and isinstance(value, jax.Array)):
        return orbax.type_handlers.ArrayRestoreArgs(
            restore_type=restore_type,
            mesh=mesh,
            mesh_axes=mesh_axes,
            global_shape=value.shape,
            dtype=value.dtype)
      else:
        dtype = None
        if hasattr(value, "dtype"):
          dtype = value.dtype
        return orbax.type_handlers.RestoreArgs(
            restore_type=restore_type,
            dtype=dtype)

    # Per-parameter mesh info is used by orbax to restore pjit-ed params.
    self._orbax_save_args = jax.tree_util.tree_map(
        _save_args, tstate_dict, tstate_mesh_axes_dict)
    self._orbax_restore_args = jax.tree_util.tree_map(
        _restore_args, tstate_dict, tstate_mesh_axes_dict)

    # Create a Checkpoint Manager only if we have a working directory.
    if not self.workdir:
      self._orbax_checkpoint_manager = None
      return

    # The checkpoint will be saved in a subdir named "checkpoint_N/tstate".
    self._orbax_checkpoint_manager = orbax.CheckpointManager(
        self.workdir,
        {"tstate": orbax.Checkpointer(orbax.PyTreeCheckpointHandler())},
        options=orbax.CheckpointManagerOptions(
            max_to_keep=1,  # TODO(delesley): make this configurable.
            # Use same prefix with flax_checkpoints.latest_checkpoint.
            step_prefix="checkpoint",
            create=True))

  def _save_orbax_checkpoint(self, tstate: TrainState, step: int):
    """Save an orbax checkpoint for tstate to workdir."""

    if not self._orbax_checkpoint_manager:
      logging.warning("Can't save checkpoint for step %d,"
                      "no CheckpointManager was created.", step)
      return
    # The checkpoint will be saved in a subdir named "checkpoint_N/tstate".
    logging.info("save_orbax_checkpoint: wait_until_finished().")
    self._orbax_checkpoint_manager.wait_until_finished()
    logging.info("save_orbax_checkpoint: save().")

    saved_tstate_dict = tstate.state_dict()
    logging.info("tstate.step = %s", tstate.step)
    print_pytree(saved_tstate_dict, "", "tstate_dict = ")

    self._orbax_checkpoint_manager.save(
        step=step,
        args=orbax.args.Composite(
            tstate=orbax.args.PyTreeSave(
                saved_tstate_dict, save_args=self._orbax_save_args
            )
        ),
        force=True,
    )
    logging.info("save_orbax_checkpoint: Done saving.")

  def _restore_orbax_checkpoint(
      self,
      checkpoint_dir: str,
      tstate: TrainState,
      pretraining_config: Optional[LoadPretrainedConfig] = None
  ) -> Optional[TrainState]:
    """Restore the latest checkpoint from checkpoint_dir, or None."""
    if self._orbax_restore_args is None:
      raise ValueError("Must call _init_checkpoint_manager before calling "
                       "_restore_orbax_checkpoint.")

    # Sanity check for test code that saves and immediately loads a checkpoint.
    if self._orbax_checkpoint_manager:
      self._orbax_checkpoint_manager.wait_until_finished()

    # Use a Checkpointer instead of CheckpointManager so we can specify the
    # step number as whichever one is the latest one.  Also, we may not
    # have a manager if there's no working directory.
    ckpath = flax_checkpoints.latest_checkpoint(checkpoint_dir, "checkpoint_")
    if not ckpath:
      raise ValueError(f"No checkpoint found in directory {checkpoint_dir}.")

    # The meliad CheckpointManager saves in a subdir named "tstate".
    # T5X checkpoints do not follow this convention.
    ckpath_tstate = os.path.join(ckpath, "tstate")
    if gfile.exists(ckpath_tstate):
      logging.info("Found meliad checkpoint %s.", ckpath_tstate)
      ckpath = ckpath_tstate
    else:
      return self._restore_t5x_checkpoint(checkpoint_dir, ckpath, tstate)

    # Follow the t5x convention of saving and restoring the state_dict.
    logging.info("restore_orbax_checkpoint: Restoring.")
    checkpointer = orbax.Checkpointer(orbax.PyTreeCheckpointHandler())

    # Frozen variables have no optimizer state. This is represented in the
    # state_dict and _orbax_restore_args by having pytree leaves with value None
    # under param_states. We use this fact to identify these paths. This may be
    # an implementation detail, but since t5x is mostly static at this point,
    # it's unlikely to change.
    item_state_dict = tstate.state_dict()
    restore_args = self._orbax_restore_args
    transforms = {}
    for path in _none_paths(item_state_dict, None):
      logging.info("nonepath detected: %r", path)
      # Pytree leaves whose full paths start with the path above will have this
      # transform applied, which accepts a pytree (along with two other
      # arguments) and returns None. This seems to allow loading checkpoints
      # that include variables that are not present in memory, which otherwise
      # cause an exception.
      transforms[f"^{re.escape(path)}.*"] = orbax.RestoreTransform(
          multi_value_fn=lambda _a, _b, _c: None,
          # I *think* this indicates that multi_value_fn does not actually
          # depend on any of the variables present in the pytree that is one
          # of its arguments. Regardless, it seems to work.
          multi_value_fn_input_args={},
      )
    transforms_default_to_original = True

    # Possibly remap params from pretrained model.
    if pretraining_config is not None:
      logging.info("restore_orbax_checkpoint: use pretraining config.")
      for (k, v) in pretraining_config.populate_key_from_value.items():
        logging.info("restore_orbax_checkpoint: %s = %s", k, v)
        transforms[k] = orbax.RestoreTransform(original_key=v)
      transforms_default_to_original = (
          pretraining_config.load_same_named_variables
      )

    loaded_tstate_dict = checkpointer.restore(
        ckpath,
        item=item_state_dict,
        restore_args=restore_args,
        transforms=transforms,
        transforms_default_to_original=transforms_default_to_original,
    )
    if not loaded_tstate_dict:
      logging.info("restore_orbax_checkpoint: unable to restore.")
      return None
    for (k, _) in loaded_tstate_dict.items():
      logging.info("restore_orbax_checkpoint: found %s", k)
    print_pytree(loaded_tstate_dict, "", "tstate_dict = ")

    loaded_tstate = tstate.restore_state(loaded_tstate_dict)
    logging.info("Restored tstate: ")
    logging.info("tstate.step = %s", loaded_tstate.step)
    print_pytree(loaded_tstate.params, "", "tstate.params = ")
    print_pytree(loaded_tstate.flax_mutables, "", "tstate.flax_mutables = ")

    return loaded_tstate

  def _restore_t5x_checkpoint(self, checkpoint_dir: str, ckpath: str,
                              tstate: TrainState) -> Optional[TrainState]:
    """Load checkpoint using the T5X checkpoint library."""
    if self._replicate_mode != "pjit":
      logging.warning("Loading a T5X checkpoint requires pjit.")
      return None
    del checkpoint_dir
    del ckpath
    del tstate

    # Currently disabled, since TrainState is not fully compatible with T5X.
    # T5X TrainState depends on some internal details of T5X optimizers.
    #
    # logging.info("Trying to restore T5X checkpoint from %s.", ckpath)
    # t5x_tstate =
    # t5x_checkpointer = t5x_checkpoints.Checkpointer(
    #     t5x_tstate, self._pjit_partitioner, checkpoint_dir)
    # t5x_tstate = t5x_checkpointer.restore(path=ckpath)
    # return train_state.from_t5x_train_state(t5x_tstate)

    logging.warning("Loading of T5X checkpoints is currently disabled.")
    return None

  def _checkpoint_exists(self, checkpoint_dir: str, msg: str) -> bool:
    """Return True if checkpoint_dir contains an existing checkpoint."""
    if not checkpoint_dir:
      logging.info("No checkpoint directory specified for %s.", msg)
      return False
    elif not gfile.exists(checkpoint_dir):
      logging.warning("Checkpoint directory %s for %s does not exist.",
                      checkpoint_dir, msg)
      return False
    elif not gfile.isdir(checkpoint_dir):
      logging.warning("Checkpoint directory name %s for %s is not a directory.",
                      checkpoint_dir, msg)
      return False
    else:
      ckpath = flax_checkpoints.latest_checkpoint(checkpoint_dir, "checkpoint_")
      if ckpath:
        logging.info("Found existing checkpoint in directory %s for %s",
                     checkpoint_dir, msg)
        return True
      else:
        logging.info("No existing checkpoint in %s for %s",
                     checkpoint_dir, msg)
        return False

  def copy_input_to_devices(self, x: Any) -> Any:
    """Copy a batch of inputs from the data pipeline to the device(s).

    Args:
      x: A batch of local (i.e. sharded) inputs from the input pipeline.

    Returns:
      (Possibly global) arrays of inputs suitable for passing to the compiled
      step function.
    """
    if self._replicate_mode == "pmap":
      # Split a batch of inputs among local devices.
      return _split_batch_dimension(x, jax.local_device_count())
    elif self._replicate_mode == "pjit":
      # Convert from a local per-process shard to a global JAX array.
      return _map_local_to_host_inputs(x, self._pjit_partitioner)
    else:
      assert self._replicate_mode == "none"
      return x

  def read_metrics_from_device(self, device_metrics: Metrics) -> Metrics:
    """Read a dictionary of metrics from back from the device.

    Args:
      device_metrics: Metrics on the device, as computed by the step function.

    Returns:
      Metrics in the form of numpy arrays that can be inspected.
    """
    if self._replicate_mode == "pmap":
      # x[0] gets the metric from device 0 -- the first local replica.
      # We assume that MetricsSummary.merge_replicated_metrics (which is
      # called in the step function) has already combined the metrics from
      # multiple devices.
      device_metrics = jax.tree.map(lambda x: x[0], device_metrics)
    metrics_np = jax.device_get(device_metrics)  # Get numpy arrays.
    return metrics_np

  def compute_parameter_metrics(self, tstate: TrainState) -> Metrics:
    """Compute parameter distributions and read metrics from device."""
    if self._compute_parameter_distributions_fn is None:
      if self._replicate_mode == "pjit":
        # Compute mean and stddev across devices.
        cpfn = self._pjit_partitioner.partition(
            compute_parameter_distributions,
            in_axis_resources=(self._tstate_partition_spec,),
            out_axis_resources=None)
      else:
        # Params are replicated, so compute mean and stddev on a single device.
        cpfn = jax.jit(compute_parameter_distributions)
      self._compute_parameter_distributions_fn = cpfn

    # When using pmap, unreplicate the tstate first...
    if self._replicate_mode == "pmap":
      param_tstate = jax_utils.unreplicate(tstate)
    else:
      param_tstate = tstate
    param_metrics = self._compute_parameter_distributions_fn(param_tstate)
    return jax.device_get(param_metrics)

  def train_step_fn(self, lr_schedule: LrScheduleFn) -> StepFunction:
    """Compiles and returns a function to take a training step."""
    # Note that this creates a single model object.
    model = self._model_definition(mode="train")
    step_fn_prng = functools.partial(self._train_step, model, lr_schedule)
    compiled_step_fn_prng = self.compile_step_function(step_fn_prng)

    # Partially apply the PRNGKeys.
    def step_fn(tstate: TrainState, x: Any) -> Tuple[TrainState, Metrics]:
      return compiled_step_fn_prng(tstate, x, self.replicated_prng)
    return step_fn

  def other_step_fn(self, mode: str) -> StepFunction:
    """Compiles and returns a function for 'test' or 'generate' modes."""
    # Note that this creates a single model object.
    model = self._model_definition(mode=mode)
    step_fn_prng = functools.partial(self._other_step, model)
    compiled_step_fn_prng = self.compile_step_function(step_fn_prng)

    # Partially apply the PRNGKeys.
    def step_fn(tstate: TrainState, x: Any) -> Tuple[TrainState, Metrics]:
      return compiled_step_fn_prng(tstate, x, self.replicated_prng)
    return step_fn

  def compile_step_function(self, step_fn: StepFuncPrng) -> StepFuncPrng:
    """Compile a step function (training or test)."""
    if self._tstate_partition_spec is None:
      raise ValueError("Must call precompile before compile_step_function.")

    if self._checkify_mode:
      step_fn = checkify.checkify(step_fn, errors=self._checkify_mode)

    if self._replicate_mode == "pmap":
      assert not self._trace_debug_mode
      logging.info("Compiling with pmap.")
      compiled_step_fn = jax.pmap(
          step_fn, donate_argnums=(0,), axis_name="batch"
      )
    elif self._replicate_mode == "pjit":
      assert not self._trace_debug_mode
      assert not self._checkify_mode, (
          "Checkify mode with pjit is not currently supported in meliad. If you"
          " want to implement it, see"
          " https://jax.readthedocs.io/en/latest/debugging/checkify_guide.html#pjit"
      )
      logging.info("Compiling with pjit.")
      return self._pjit_partitioner.partition(
          step_fn,
          in_axis_resources=(
              self._tstate_partition_spec,   # tstate
              self._input_partition_spec,    # input values
              None                           # prng
          ),
          out_axis_resources=(
              self._tstate_partition_spec,   # tstate
              None                           # replicated metrics
          ),
          donate_argnums=(0,))
    else:
      assert self._replicate_mode == "none"
      if self._trace_debug_mode:
        logging.info("Running in interpreted mode with trace_debug.")
        compiled_step_fn = step_fn
      else:
        logging.info("Compiling with jit.")
        compiled_step_fn = jax.jit(step_fn, donate_argnums=(0,))

    if not self._checkify_mode:
      return typing.cast(StepFuncPrng, compiled_step_fn)

    compiled_step_fn = typing.cast(
        Callable[..., tuple[checkify.Error, tuple[TrainState, Metrics]]],
        compiled_step_fn,
    )
    # Wrap step_fn with a handler for errors from checkify.
    def step_fn_prng(tstate: TrainState, x: Any, prng: PRNGKey):
      return self._call_and_handle_errors(
          compiled_step_fn, tstate, x, prng
      )
    return step_fn_prng

  def _train_step(self, model: ModelBase, lr_schedule: LrScheduleFn,
                  tstate: TrainState, x: Any, prng: PRNGKey) -> (
                      Tuple[TrainState, Metrics]):
    """Perform a training step, pmapped over multiple devices.

    Args:
      model:       The model to use for the step function.
      lr_schedule: Function from integer step to learning rate.
      tstate:      Values for state variables, and the optimizer.
      x:           A batch of inputs to train on.
      prng:       A jax prng (random number generator) key.

    Returns:
      Tuple of (new_tstate, metrics: dictionary of scalar values)
    """
    (m_params, m_state, mutable_keys, step) = self._unpack_tstate(tstate)
    prng_dict = self._get_prng_dict(prng, step)

    # Refactor the model as a loss function from trainable params to loss, so
    # that we can differentiate with jax and get {d}loss/{d}params.
    # Inputs and non-trainable params are bound within the closure.
    # model:: x, { state_params } -> (loss, metrics), { new_state_params }
    # loss_fn:: params -> (loss, (metrics, new_state))
    def loss_fn(params):
      """Loss function."""
      ((loss, mets), nstate) = model.model_apply(
          {"params": params, **m_state},
          x,
          prng_dict=prng_dict,
          mutable_keys=mutable_keys,
          global_info=_extract_globals(tstate))
      return loss, (mets, nstate)

    # grad_fn:: params -> ((loss, (aux, nstate)), param_gradients)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    # Run forward and backward pass.
    (loss, (metrics, new_state)), param_grads = grad_fn(m_params)
    del loss  # loss is only recorded if it is part of the metrics

    # Apply gradients.
    if self._replicate_mode == "pmap":
      # Combine gradients from all replicas.
      param_grads = jax.lax.pmean(param_grads, axis_name="batch")
    lrate = lr_schedule(step)
    tstate = tstate.apply_gradient(param_grads, learning_rate=lrate,
                                   flax_mutables=new_state)

    # Metrics are summary values (usually scalars) that are logged over time.
    # Before logging, metrics are reduced across the batch dimension.
    logging.info("Unreduced metrics returned from model:")
    metrics["learning_rate"] = metrics_summary.average_metric(lrate)
    metrics_summary.log_metric_info(metrics)
    metrics = metrics_summary.reduce_metrics(metrics, self._replicate_mode)
    return (tstate, metrics)

  def _other_step(self, model: ModelBase,
                  tstate: TrainState, x: Any, prng: PRNGKey) -> (
                      Tuple[TrainState, Metrics]):
    """Perform a test or generate step, pmapped over multiple devices.

    Args:
      model:      The model to use for the step function.
      tstate:     Values for state variables, and the optimizer.
      x:          A batch of inputs to train on.
      prng:       A jax prng (random number generator) key.

    Returns:
      Tuple of (new_tstate, metrics: dictionary of scalar values)
    """

    (m_params, m_state, mutable_keys, step) = self._unpack_tstate(tstate)
    prng_dict = self._get_prng_dict(prng, step)

    ((loss, metrics), new_state) = model.model_apply(
        {"params": m_params, **m_state},
        x,
        prng_dict=prng_dict,
        mutable_keys=mutable_keys,
        global_info=_extract_globals(tstate))
    del loss  # loss is only recorded if it is part of the metrics

    # Metrics are summary values (usually scalars) that are logged over time.
    # Before logging, metrics are reduced across the batch dimension.
    logging.info("Unreduced metrics returned from model:")
    metrics_summary.log_metric_info(metrics)
    metrics = metrics_summary.reduce_metrics(metrics, self._replicate_mode)
    tstate = tstate.replace_flax_mutables(new_state)
    return (tstate, metrics)

  def _compute_dataset_sharding(self):
    """Figure out how to shard the data set."""
    if self._replicate_mode == "pmap":
      self.num_batch_devices = jax.local_device_count() * jax.process_count()
      self.num_shards = jax.process_count()
      self.shard_id = jax.process_index()
    elif self._replicate_mode == "pjit":
      data_axis = self._pjit_partitioner._data_axis  # pylint: disable=protected-access
      data_layout = self._pjit_partitioner.get_data_layout()
      self.num_batch_devices = self._pjit_partitioner.mesh.shape[data_axis]
      self.num_shards = data_layout.num_shards
      self.shard_id = data_layout.shard_id
    else:
      assert self._replicate_mode == "none"
      self.num_batch_devices = 1
      self.num_shards = 1
      self.shard_id = 0
    assert (self.num_batch_devices % self.num_shards) == 0

  def _replicate_prngs(self, prng: PRNGKey) -> PRNGKey:
    """Split a single prng key across local devices."""
    if self._replicate_mode == "pmap":
      # Give each local replica on each process a different prng key.
      prng = jax.random.fold_in(prng, jax.process_index())
      return jax.random.split(prng, jax.local_device_count())
    elif self._replicate_mode == "pjit":
      # We use a single prng; pjit handles the multiple device logic.
      return prng
    else:
      assert self._replicate_mode == "none"
      return prng

  def _get_init_prng_dict(self, prng: PRNGKey) -> Dict[str, PRNGKey]:
    """Create a dictionary of rng keys for initialization."""
    rng_key_names_init = list(self.rng_key_names) + ["params"]
    rngs = jax.random.split(prng, len(rng_key_names_init))
    return {key: rngs[i] for i, key in enumerate(rng_key_names_init)}

  def _get_prng_dict(self, prng: PRNGKey, step: int) -> Dict[str, PRNGKey]:
    """Create a dictionary of unique rng keys for each step."""
    prng = jax.random.fold_in(prng, step)
    rngs = jax.random.split(prng, len(self.rng_key_names))
    return {key: rngs[i] for i, key in enumerate(self.rng_key_names)}

  def _unpack_tstate(self, tstate: TrainState) -> (
      Tuple[FrozenVariableDict, FrozenVariableDict, Collection[str], Any]):
    """Unpacks tstate and populates the 'globals' variable collection."""
    m_params = tstate.params
    m_state = tstate.flax_mutables
    mutable_keys = [k for (k, _) in m_state.items()]
    step = tstate.step
    return (m_params, m_state, mutable_keys, step)


@gin.configurable
def compute_parameter_distributions(tstate: TrainState,
                                    include_minmax: bool = False) -> Metrics:
  """Calculate distributions of parameters."""
  params_dict = flatten_dict_string_keys(tstate.params)
  scalar_params_dict = {}
  for (k, v) in params_dict.items():
    # Convert from bfloat16, which crashes when serializing a NaN.
    v = jnp.asarray(v, dtype=jnp.float32)

    # Skip large arrays; pjit will crash if number of elements is not an int32.
    # Filed bug #15068.  https://github.com/google/jax/issues/15068
    nelems = math.prod(v.shape)
    if nelems > ((1 << 31) - 1):
      logging.info("Skipping large parameter %s of shape %s", k, v.shape)
      continue

    scalar = metrics_summary.scalar_metric

    scalar_params_dict[k + "_mean"] = scalar(jnp.mean(v))
    scalar_params_dict[k + "_stddev"] = scalar(jnp.std(v))
    if include_minmax:
      scalar_params_dict[k + "_min"] = scalar(jnp.min(v))
      scalar_params_dict[k + "_max"] = scalar(jnp.max(v))
  return scalar_params_dict


def write_parameter_info(tstate: TrainState,
                         tstate_partition_spec: TrainState):
  """Write information on state and trainable parameters to the log."""

  # Write information on parameters to log file.
  logging.info("==== Model parameters ====")
  params_dict = flatten_dict_string_keys(tstate.params)
  params_partition_dict = flatten_dict_string_keys(
      tstate_partition_spec.params)

  total_nparams = 0
  for (k, v) in params_dict.items():
    nparams = np.prod(v.shape)
    total_nparams += nparams
    if k in params_partition_dict:
      pspec = params_partition_dict[k]
    else:
      pspec = "<missing>"
    logging.info("parameter: %s, shape: %s, dtype: %s, size: %d, mesh_axes: %s",
                 k, v.shape, v.dtype, nparams, pspec)
  logging.info("Total parameters: %d", total_nparams)

  # Write information on state variables to log file.
  logging.info("==== Other model state variables ====")
  flax_mutables_dict = flatten_dict_string_keys(tstate.flax_mutables)
  flax_mutables_partition_dict = flatten_dict_string_keys(
      tstate_partition_spec.flax_mutables)

  total_state = 0
  for (k, v) in flax_mutables_dict.items():
    if k in flax_mutables_partition_dict:
      pspec = flax_mutables_partition_dict[k]
    else:
      pspec = "<missing>"
    if hasattr(v, "shape"):
      state_size = np.prod(v.shape)
      total_state += state_size
      logging.info("name: %s, shape: %s, dtype: %s, size: %d, mesh_axes: %s",
                   k, v.shape, v.dtype, state_size, pspec)
    else:
      # Some other stuff may be stored in the state.
      logging.info("name: %s [unknown]", k)
  logging.info("Total size of other state: %d", total_state)


def flatten_dict_string_keys(params):
  """Flattens a nested dictionary to have string keys and '/' separators."""
  return {"/".join(k): v for k, v in flatten_dict(unfreeze(params)).items()}


def _map_local_to_host_inputs(inputs: Any,
                              pjit_partitioner: PjitPartitioner) -> Any:
  """Convert local inputs on each host to a global jax array.

  Args:
    inputs: pytree of local inputs.
    pjit_partitioner: the partitioner holds info about the model/data mesh.

  Returns:
    A pytree of inputs as jax global arrays.
  """
  def local_to_host_fn(x: jax.Array) -> jax.Array:
    return multihost_utils.host_local_array_to_global_array(
        x,
        pjit_partitioner.mesh,
        pjit_partitioner.data_partition_spec)
  return jax.tree.map(local_to_host_fn, inputs)


def _split_batch_dimension(inputs: Any, num_replicas: int) -> Any:
  """Splits the leading batch dimension.

  Given inputs of shape [num_replicas * batch_size, ...], it will reshape
  them to [num_replicas, batch_size, ...].  This operation is intended to be
  used right before calling pmap, which will eliminate the num_replicas
  dimension.

  Args:
    inputs: pytree of inputs to split.
    num_replicas: Number of replicas.

  Returns:
    A pytree of inputs with an extra batch dimension.
  """

  def split_batch_dim(x):
    assert x.ndim > 0
    if (x.shape[0] % num_replicas) != 0:
      raise ValueError(f"Can't split {x.shape} into {num_replicas} replicas.")
    batch_size = x.shape[0] // num_replicas
    split_shape = [num_replicas, batch_size] + list(x.shape[1:])
    return np.reshape(x, split_shape)

  return jax.tree.map(split_batch_dim, inputs)


def _none_paths(
    tree: Mapping[str, Any] | None, prefix: Optional[str]
) -> Iterable[str]:
  """Recursively extract the full paths of any None-valued leaves.

  Args:
    tree: A pytree, specifically a nested dictionary in which all keys at all
      levels are strings.
    prefix: The path so far, which will be prepended to the additional keys. For
      a top-level call, the prefix should be either "" (to get a path starting
      with '/') or None.

  Yields:
    The full path (consisting of keys in nested dicts, joined with '/') to each
    leaf with value None.
  """
  # Cannot use tree_util because its functions ignore leaves with value `None`.
  if tree is None:
    yield prefix or ""
  elif isinstance(tree, Mapping):
    for k, v in tree.items():
      subprefix = (
          k if prefix is None else f"{prefix}/{k}"
      )  # Don't start with '/' if prefix is None
      for path in _none_paths(v, subprefix):
        yield path


# Set to true for verbose logging of to_pmap and from_pmap functions.
_VERBOSE_PMAP = False


def _pmap_to_global(tstate: TrainState) -> TrainState:
  """Convert all the Arrays in the state to have global sharding.

  This function must be called before saving checkpoints with orbax, because
  orbax hates pmap.  See also orbax.checkpoint.utils.
  fully_replicated_host_local_array_to_global_array()

  This is the inverse of _to_pmap(), thus all of the Arrays in the input
  state should have pmap-compatible sharding.  No data is copied.

  Args:
    tstate: A TrainState to checkpoint.

  Returns:
    A TrainState with the same structure as the input, except all the Arrays
    will have a global shape that is replicated across /all/ devices.  The
    leading dimension of size jax.local_device_count() is eliminated.
  """

  logging.info("from_pmap: Converting TrainState to global array for "
               "checkpoint.")
  mesh = jax.sharding.Mesh(jax.devices(), "x")

  def from_pmap_fn(path, a: Array) -> Array:
    nonlocal mesh
    assert isinstance(a, jax.Array)

    if jax.config.jax_pmap_no_rank_reduction:
      # Shard shape has the same rank, but a leading dimension of 1.
      shard_shape = a.addressable_data(0).shape
      local_shape = shard_shape[1:]
      valid_local_shape = (shard_shape[0] == 1)
    else:
      # Shard shape removes the leading dimension.
      local_shape = a.addressable_data(0).shape
      valid_local_shape = True

    if (a.shape == (jax.local_device_count(), *local_shape)
        and valid_local_shape):
      global_shape = local_shape   # Save with unreplicated shape.
    else:
      raise ValueError(f"Array at {path} of shape {a.shape} and "
                       f"local_shape {local_shape} with fully-replicated="
                       f"{a.is_fully_replicated} is not pmap-ed.")

    path_str = ""
    if _VERBOSE_PMAP:
      path_str = "/".join([_get_path_name(p) for p in path])
      logging.info("from_pmap: reshaping %s of shape %s, fully_replicated=%s"
                   " --> shape %s",
                   path_str, a.shape, a.is_fully_replicated, global_shape)

    # TODO(delesley): flax_mutables (e.g. kv-cache) are usually not replicated,
    # And thus should not be saved with this technique.

    # Array must be fully replicated across devices for this to work.
    if global_shape:
      pspec = jax.sharding.PartitionSpec(None)
    else:
      pspec = jax.sharding.PartitionSpec()

    # pmap-produced Array has a "scrambled" device order.  See:
    # orbax.checkpoint.utils.fully_replicated_host_local_array_to_global_array()
    dbs = sorted([shard.data for shard in a.addressable_shards],
                 key=lambda x: list(x.devices())[0].id,)
    if jax.config.jax_pmap_no_rank_reduction:
      # Remove leading dimension from shard.
      dbs = [s[0] for s in dbs]

    global_arr = jax.make_array_from_single_device_arrays(
        global_shape,
        jax.sharding.NamedSharding(mesh, pspec),
        dbs
    )
    if _VERBOSE_PMAP:
      logging.info("from_pmap: created %s of shape %s fully_replicated=%s",
                   path_str, global_arr.shape, global_arr.is_fully_replicated)
    return global_arr

  return jax.tree_util.tree_map_with_path(from_pmap_fn, tstate)


def _global_or_local_to_pmap(tstate: TrainState) -> TrainState:
  """Convert all the Arrays in the state to have pmap-compatible sharding.

  This function acts as the inverse of _pmap_to_global, and can be used to
  convert global arrays back to a pmap-friendly form after restoring from a
  checkpoint.  It will also call jax_utils.replicate for local arrays.

  Args:
    tstate: A TrainState object to replicate or reshape.

  Returns:
    A TrainState with the same structure as the input, except all the Arrays
    will be replicated across all /local/ devices, and have a leading dimension
    of size jax.local_device_count().
  """

  logging.info("to_pmap: Converting TrainState to pmap-friendly form.")

  def to_pmap_fn(path, a: Array) -> Array:
    assert isinstance(a, jax.Array)
    if a.is_fully_addressable:              # only exists on one host.
      if len(a.sharding.device_set) == 1:   # only exists on one device.
        if _VERBOSE_PMAP:
          path_str = "/".join([_get_path_name(p) for p in path])
          logging.info("to_pmap: replicating array %s of shape %s",
                       path_str, a.shape)
        return jax_utils.replicate(a)

    # TODO(delesley): flax_mutables (e.g. kv-cache) are usually not replicated,
    # And thus should not be saved with this technique.

    global_shape = a.shape  # Unreplicated shape without pmap leading dimension.
    dbs = [shard.data for shard in a.addressable_shards]

    # Add a leading dimension for local devices, as expected by pmap.
    assert global_shape == a.addressable_data(0).shape
    shape = (jax.local_device_count(), *global_shape)

    if jax.config.jax_pmap_no_rank_reduction:
      # Add a leading dimension for each shard
      dbs = [s[jnp.newaxis] for s in dbs]

    if _VERBOSE_PMAP:
      path_str = "/".join([_get_path_name(p) for p in path])
      logging.info("to_pmap: reshaping %s of shape %s, fully_replicated=%s"
                   " --> shape %s",
                   path_str, global_shape, a.is_fully_replicated, shape)

    pmap_arr = jax.make_array_from_single_device_arrays(
        shape,
        jax.sharding.PmapSharding.default(shape, 0, jax.local_devices()),
        dbs,
    )
    return pmap_arr

  return jax.tree_util.tree_map_with_path(to_pmap_fn, tstate)


def _get_path_name(p: Any) -> str:
  if hasattr(p, "name"):
    return p.name
  elif hasattr(p, "key"):
    return p.key
  else:
    return str(p)


def print_pytree_with_indices(pytree_value, pytree_name: str = "pytree"):
  # Prints a pytree with integer indices, for debugging purposes.
  (xs, pytdef) = jax.tree_util.tree_flatten(pytree_value)
  pytindices = jax.tree_util.tree_unflatten(pytdef, range(len(xs)))
  logging.info("%s=%s", pytree_name, pytindices)


def print_pytree(pytree_value, indent: str = "", prefix=""):
  """Pretty print a pytree for debugging purposes."""

  if isinstance(pytree_value, dict):
    logging.info("%s%s{\n", indent, prefix)
    for (k, v) in pytree_value.items():
      print_pytree(v, indent + "  ", f"{k}: ")
    logging.info("%s},\n", indent)
  elif  isinstance(pytree_value, flax_scope.FrozenDict):
    logging.info("%s%s flax.FrozenDict {\n", indent, prefix)
    for (k, v) in pytree_value.items():
      print_pytree(v, indent + "  ", f"{k}: ")
    logging.info("%s},\n", indent)
  elif isinstance(pytree_value, tuple):
    logging.info("%s%s(\n", indent, prefix)
    for v in pytree_value:
      print_pytree(v, indent + "  ", "")
    logging.info("%s),\n", indent)
  elif isinstance(pytree_value, list):
    logging.info("%s%s[\n", indent, prefix)
    for v in pytree_value:
      print_pytree(v, indent + "  ", "")
    logging.info("%s],\n", indent)
  elif isinstance(pytree_value, jax.Array):
    if not pytree_value.shape:  # shape == ()
      logging.info("%s%s%s", indent, prefix, pytree_value)
    else:
      logging.info("%s%sjax.Array(%s, %s)",
                   indent, prefix, pytree_value.shape, pytree_value.dtype)
  else:
    # Leaf node.
    logging.info(
        "%s%s%s,", indent, prefix, pytree_value.__class__
    )

