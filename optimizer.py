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

"""Replacement for the now-deprecated flax.optim library.

Implements a simplified version of the flax.optim interface using optax.
Meliad can use either optax optimizers with this wrapper, or t5x optimizers.
"""

from typing import Any

from flax import serialization
from flax import struct
import jax
import jax.numpy as jnp
import optax


# Function f(learning_rate: optax.Schedule) -> optax.GradientTransformation
type OptaxOptFactory = Any


class OptimizerDef:
  """Base class for an optimizer definition."""

  def __init__(self, optax_optimizer_factory: OptaxOptFactory):
    """Create an optimizer definition.

    Args:
      optax_optimizer_factory:  A function that takes a single argument
          learning_rate_fn of type optax.Schedule, and returns an optax
          GradientTransformation.
    """
    self.optax_optimizer_factory = optax_optimizer_factory

  def create(self, target):
    """Creates a new optimizer for the given target.

    Args:
      target: the object to be optimized. This is typically a variable dict
        of trainable parameters returned by `flax.linen.Module.init()`.

    Returns:
      An instance of `Optimizer`.
    """

    # We have to create the LearningRateScheduleState first, so that we can
    # get the learning_rate_fn from it to create the optimizer.
    lr_schedule_state = LearningRateScheduleState()
    optax_optimizer = self.optax_optimizer_factory(
        learning_rate_fn=lr_schedule_state.learning_rate_fn
    )
    opt_target = target
    opt_state = OptimizerState(
        optax_state=optax_optimizer.init(target),
        step=jnp.array(0, dtype=jnp.int32),
    )
    return Optimizer(
        lr_schedule_state=lr_schedule_state,
        optax_optimizer=optax_optimizer,
        target=opt_target,
        state=opt_state
    )


class LearningRateScheduleState:
  """Holds the current learning rate as part of the optimizer state."""

  def __init__(self):
    self.learning_rate = jnp.array(0.0, dtype=jnp.float32)

  def learning_rate_fn(self, count: jax.Array) -> jax.Array:
    """A learning rate schedule for optax."""
    # We ignore the 'count' parameter, which comes from some opaque location
    # within the optax optimizer.  Instead, we track the current step manually,
    # compute the current learning rate manually, and return it here.
    del count
    return self.learning_rate


class OptimizerState(struct.PyTreeNode):
  """Holds the optimizer state along with the current step."""

  # TrainState expects the "state" member to have a "step" field.
  # ModelInfo will extract the "step" field, calculate the learning rate,
  # and pass it to apply_gradients below.
  optax_state: Any = struct.field(pytree_node=True)
  step: jax.Array = struct.field(pytree_node=True)


class Optimizer(struct.PyTreeNode):
  """A wrapper around an optax optimizer that mimics flax.optim.Optimizer."""

  lr_schedule_state: Any = struct.field(pytree_node=False)
  optax_optimizer: Any = struct.field(pytree_node=False)
  target: Any = struct.field(pytree_node=True)
  state: OptimizerState = struct.field(pytree_node=True)

  def apply_gradient(self, grads, learning_rate: jax.Array) -> "Optimizer":
    """Applies a pytree of gradients to the target.

    Args:
      grads: A pytree of gradients.
      learning_rate: The current learning rate.

    Returns:
      A new optimizer with the updated target and state.
    """

    # Update the current learning rate.  The optax optimizer has been created
    # with a "schedule" that uses this learning rate.  The training loop
    # implements its own schedule from self.state.step, rather than relying
    # on optax.
    self.lr_schedule_state.learning_rate = learning_rate

    params = self.target
    optax_state = self.state.optax_state
    (updates, new_optax_state) = self.optax_optimizer.update(
        grads, optax_state, params=params
    )
    new_params = optax.apply_updates(params, updates)
    new_step = self.state.step + 1
    return self.replace(
        target=new_params,
        state=self.state.replace(
            optax_state=new_optax_state,
            step=new_step,
        )
    )

  def state_dict(self):
    """Returns a flax state dictionary with the optimizer state."""
    return serialization.to_state_dict({
        "target": serialization.to_state_dict(self.target),
        "state": serialization.to_state_dict(self.state)
    })

  def restore_state(self, state_dict):
    """Sets the state of the optimizer, given a flax state dictionary."""
    new_target = serialization.from_state_dict(
        self.target, state_dict["target"]
    )
    new_state = serialization.from_state_dict(
        self.state, state_dict["state"]
    )
    return self.replace(target=new_target, state=new_state)
