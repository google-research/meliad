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

"""Defines a class to hold the current state during training."""

from typing import Any, Optional, Tuple

from absl import logging
import flax
from flax import traverse_util
from flax.core import scope as flax_scope
from flax.linen import partitioning as flax_partitioning

import jax
import jax.numpy as jnp
import  optimizer_config as opt_config


EMPTY_DICT = flax.core.freeze({})
FrozenDict = flax_scope.FrozenDict
FrozenVariableDict = flax_scope.FrozenVariableDict
MutableVariableDict = flax_scope.MutableVariableDict
VariableDict = flax_scope.VariableDict


type Optimizer = opt_config.Optimizer
type OptimizerDef = opt_config.OptimizerDef


# Copied from t5x.train_state.FlaxOptimTrainState.
# Added additional "use_axes" argument to TrainState.create().
class TrainState(flax.struct.PyTreeNode):
  """Simple train state for holding parameters, step, optimizer state."""

  _optimizer: Optimizer
  # Contains axis metadata (e.g., names) matching parameter tree.
  params_axes: Optional[FrozenVariableDict] = None
  # Flax mutable fields.
  flax_mutables: FrozenDict = EMPTY_DICT
  # Contains axis metadata (e.g., names) matching flax_mutables tree.
  flax_mutables_axes: Optional[FrozenVariableDict] = None

  @classmethod
  def create(
      cls,
      optimizer_def: OptimizerDef,
      model_variables: FrozenVariableDict,
      use_axes: bool = True,
  ) -> "TrainState":
    other_variables, params = flax.core.frozen_dict.pop(
        model_variables, "params"
    )

    # Separate out the parameter axes information.
    if "params_axes" in other_variables:
      other_variables, params_axes = flax.core.frozen_dict.pop(
          other_variables, "params_axes"
      )
      if use_axes:
        _validate_params_axes(params_axes, params)
      else:
        params_axes = None
    else:
      params_axes = None

    # Split other_variables into mutables and their corresponding axes.
    (flax_mutables, flax_mutables_axes) = _split_variables_and_axes(
        other_variables, validate_axes=use_axes
    )
    if not use_axes:
      flax_mutables_axes = None

    # If the optimizer supports `set_param_axes`, then assume that the model
    # code is emitting these axes and use it.
    if use_axes and hasattr(optimizer_def, "set_param_axes"):
      if params_axes is None:
        raise ValueError(
            "The optimizer supports params_axes for model-based "
            "partitioning, but the model is not emitting them."
        )
      # `get_axis_names` removes "_axes" suffix in the leaf name and replaces
      # `AxisMetadata` with `PartitionSpec`.
      axis_names = flax_partitioning.get_axis_names(params_axes)
      optimizer_def.set_param_axes(axis_names)

    optimizer = optimizer_def.create(params)
    flax_mutables_axes = flax_mutables_axes or None
    logging.info("TrainState: params = %s", params)
    logging.info("TrainState: flax_mutables = %s", flax_mutables)

    return TrainState(
        optimizer,
        params_axes=params_axes,
        flax_mutables=flax_mutables,
        flax_mutables_axes=flax_mutables_axes,
    )

  @property
  def params(self) -> FrozenVariableDict:
    return self._optimizer.target

  @property
  def step(self) -> jnp.ndarray:
    return self._optimizer.state.step

  def state_dict(self) -> MutableVariableDict:
    state_dict = self._optimizer.state_dict()
    if self.flax_mutables:
      state_dict["flax_mutables"] = flax.core.unfreeze(self.flax_mutables)
    return state_dict

  def apply_gradient(
      self, grads, learning_rate, flax_mutables=EMPTY_DICT
  ) -> "TrainState":
    new_optimizer = self._optimizer.apply_gradient(
        grads, learning_rate=learning_rate
    )
    return self.replace(_optimizer=new_optimizer, flax_mutables=flax_mutables)

  def replace_params(self, params: VariableDict) -> "TrainState":
    return self.replace(_optimizer=self._optimizer.replace(target=params))

  def replace_flax_mutables(self, flax_mutables: FrozenDict) -> "TrainState":
    return self.replace(flax_mutables=flax_mutables)

  def replace_step(self, step: jax.Array) -> "TrainState":
    state_dict = self.state_dict()
    state_dict["state"]["step"] = step
    return self.restore_state(state_dict)

  def restore_state(self, state_dict: VariableDict) -> "TrainState":
    new_optimizer = self._optimizer.restore_state(state_dict)
    new_flax_mutables = (
        flax.core.freeze(state_dict["flax_mutables"])
        if "flax_mutables" in state_dict
        else EMPTY_DICT
    )
    return self.replace(
        _optimizer=new_optimizer,
        flax_mutables=new_flax_mutables,
    )


# Copied from t5x.train_state._validate_params_axes.
def _validate_params_axes(params_axes: Any, params: Any) -> None:
  """Check that all parameters have named axes."""
  axis_names = flax_partitioning.get_axis_names(params_axes)
  missing_params_axes = (
      set(traverse_util.flatten_dict(params, sep="/")) -
      set(traverse_util.flatten_dict(axis_names, sep="/"))
  )
  if missing_params_axes:
    raise ValueError(
        f"Missing axis names for parameters: {missing_params_axes}"
    )


# Copied from t5x.train_state, with addition of validate_axes flag.
def _split_variables_and_axes(
    variables_and_axes: FrozenVariableDict,
    validate_axes: bool = True,
) -> Tuple[FrozenVariableDict, FrozenVariableDict]:
  """Splits `variables_and_axes` into two separate dicts with the same keys."""
  # For each `key`, `key_axes` (if any) are its axes in `variables_and_axes`.
  variables = {}
  axes = {}
  for k, v in variables_and_axes.items():
    if k.endswith("_axes"):
      k_base = k[:-5]   # k without "_axes".
      axes[k_base] = v
      if validate_axes:
        _validate_params_axes(v, variables_and_axes[k_base])
    else:
      variables[k] = v
  return flax.core.freeze(variables), flax.core.freeze(axes)




