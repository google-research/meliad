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

"""Gin configurable optimizer definitions.
"""

from typing import Any, Optional

from absl import logging
from flax import optim
from flax import struct
import gin
import jax.numpy as jnp
import numpy as np


OptimizerDef = Any


@struct.dataclass
class OptimizerConfig:
  """Base class for optimizer configurations."""

  learning_rate: float = 0.01    # All optimizers have a learning rate.

  def create_optimizer_def(self) -> OptimizerDef:
    raise ValueError("Not implemented.")


@gin.configurable
@struct.dataclass
class AdamConfig(OptimizerConfig):
  """Creates and configures the Adam optimizer."""

  # Adam does not use parameter scale, and thus requires a smaller lrate.
  # This will be multiplied by the learning rate schedule.
  learning_rate: float = 0.05

  beta1: float = 0.9               # For moving average of gradient.
  beta2: float = 0.98              # For moving average of gradient magnitude.
  weight_decay_rate: float = 0.0   # Relative to learning rate.

  def create_optimizer_def(self) -> optim.OptimizerDef:
    logging.info("Using Adam Optimizer. lr=%f, b1=%f, b2=%f",
                 self.learning_rate, self.beta1, self.beta2)
    return optim.Adam(beta1=self.beta1,
                      beta2=self.beta2,
                      weight_decay=self.weight_decay_rate)


@gin.configurable
@struct.dataclass
class FlaxAdafactorConfig(OptimizerConfig):
  """Creates and configures the Adafactor optimizer."""

  # Adafactor scales gradients according to parameter scale.
  # This will be multiplied by the learning rate schedule.
  learning_rate: float = 1.0
  beta1: Optional[float] = 0.9      # Enables momentum with extra memory cost.

  def create_optimizer_def(self) -> optim.OptimizerDef:
    # Use wd_lr_exponent to get weight_decay relative to learning rate.
    logging.info("Using Flax Adafactor Optimizer. lr=%f, b1=%f",
                 self.learning_rate, self.beta1)
    return optim.Adafactor(beta1=self.beta1)




# ----------------------------------------------------------------------------
# Learning rate schedules for use with any optimizer.
#
# In keeping with the Chinchilla model: https://arxiv.org/abs/2203.15556.
# A learning rate schedule is a function that decays the learning rate from
# step zero to max_steps.  The desired maximum number of steps must be set at
# the start of training.
# ----------------------------------------------------------------------------


@gin.configurable
def lr_constant(step: jnp.ndarray, max_steps: int,
                learning_rate: float = 0.01) -> jnp.ndarray:
  """Returns constant_lr on each step.

  Args:
    step: The current training step (unused).
    max_steps: Unused.
    learning_rate: The constant learning rate to use.

  Returns:
    The learning rate for the current step.
  """
  del step
  del max_steps
  return jnp.asarray(learning_rate, dtype=jnp.float32)


@gin.configurable
def lr_rsqrt_decay_std(step: jnp.ndarray, max_steps: int,
                       max_lr: Optional[float] = None) -> jnp.ndarray:
  """Inverse square root decay function: LR = 1/sqrt(step).

  Provided for compatibility.  No min_lr, and it ignores max_steps.
  Should be used with warmup: pass step = max(step, warmup_steps).
  Maximum learning rate is 1/sqrt(warmup_steps) ~= 0.03 for 1000 warmup steps.

  Args:
    step: The current training step.
    max_steps: Unused.
    max_lr: If specified, learning rate will be clipped to the maximum value.

  Returns:
    The learning rate for the current step.
  """
  # This function implements standard rsqrt decay as used in the memorizing
  # and block-recurrent transformer papers, (https://arxiv.org/abs/2203.08913,
  # https://arxiv.org/abs/2203.07852) which does not decay to a specified
  # minimum learning rate over max_steps.
  del max_steps

  # Avoid divide by zero; force at least 100 warmup steps and a max LR of 0.1.
  step = jnp.maximum(step, 100.0)
  lrate = 1.0 / jnp.sqrt(step)
  if max_lr is not None:
    lrate = jnp.minimum(lrate, max_lr)  # Clip to max_lr
  return lrate


@gin.configurable
def lr_rsqrt_decay(step: jnp.ndarray, max_steps: int,
                   max_lr: float = 0.05,
                   min_lr: float = 0.001) -> jnp.ndarray:
  """Inverse sqrt decay from max_lr to min_lr over max_steps.

  This function implements rsqrt decay, but adjusts the decay rate so that
  min_lr is reached at max_steps.

  Note: with a warmup period, the maximum LR produced by the schedule is:
  min_lr / sqrt(warmup_steps / max_steps), which may be less than max_lr.
  e.g. if min_lr is 0.001, then the maximum LR will be 0.01 for
  warmup_steps=1000 and max_steps=100_000.

  Args:
    step: The current training step.
    max_steps: The step value at the end of training.
    max_lr: LR will be clipped to max at the start of training.
    min_lr: LR to output at max_steps.

  Returns:
    The learning rate for the current step.
  """
  assert max_lr > min_lr

  # Avoid divide by zero; force at least 100 warmup steps and a max LR of 0.1.
  step = jnp.maximum(step, 100.0)
  lrate = min_lr / jnp.sqrt(step / float(max_steps))
  lrate = jnp.minimum(lrate, max_lr)  # Clip to max_lr
  return lrate


@gin.configurable
def lr_exponential_decay(step: jnp.ndarray, max_steps: int,
                         max_lr: float = 0.01,
                         min_lr: float = 0.001) -> jnp.ndarray:
  """Exponential decay from max_lr to min_lr over max_steps.

  Continues to decay at the same rate after max_steps.

  Args:
    step: The current training step.
    max_steps: The step value at the end of training.
    max_lr: LR to output at step 0.
    min_lr: LR to output at max_steps.

  Returns:
    The learning rate for the current step.
  """
  assert max_lr > min_lr

  lrate = max_lr * jnp.power(min_lr / max_lr, step / float(max_steps))
  return lrate


@gin.configurable
def lr_linear_decay(step: jnp.ndarray, max_steps: int,
                    max_lr: float = 0.01,
                    min_lr: float = 0.001,
                    decay_after: bool = True) -> jnp.ndarray:
  """Linear decay from max_lr to min_lr over max_steps.

  If decay_after, then LR will continue to decay exponentially by a factor
  of 2 every max_steps after the linear decay.

  Args:
    step: The current training step.
    max_steps: The step value at the end of training.
    max_lr: LR to output at step 0.
    min_lr: LR to output at max_steps.
    decay_after: If true, do exponential decay after the linear decay,
        by a factor of 2 every max_steps.

  Returns:
    The learning rate for the current step.
  """
  assert max_lr > min_lr

  lrate = min_lr + (max_lr - min_lr) * ((max_steps - step) / max_steps)
  lrate = jnp.maximum(lrate, min_lr)

  if decay_after:
    exp_lrate = lr_exponential_decay(step, max_steps,
                                     max_lr=2*min_lr, min_lr=min_lr)
    lrate = jnp.where(step < max_steps, lrate, exp_lrate)

  return lrate


@gin.configurable
def lr_cosine_decay(step: jnp.ndarray, max_steps: int,
                    max_lr: float = 0.01,
                    min_lr: float = 0.001,
                    decay_after: bool = True,
                    spike_steps: int = 0,
                    spike_lr: float = 0.0) -> jnp.ndarray:
  """Cosine decay function from max_lr to min_lr over max_steps.

  Used in the Chinchilla model: https://arxiv.org/abs/2203.15556.

  If decay_after, then LR will continue to decay exponentially by a factor
  of 2 every max_steps after the original ramp.

  If spike_steps > 0, there will be an initial linear decay from spike_lr
  down to max_lr over the first spike_steps steps.  This implements a brief
  period of higher LR early in training, similar to the curve for rsqrt_decay.
  The model can generally tolerate a high LR early in training, and make a
  lot of progress very quickly.  Try spike_steps=10_000, spike_lr = 0.04.

  Args:
    step: The current training step.
    max_steps: The number of training steps to decay over.
    max_lr: The maximum learning rate at the start of training.
    min_lr: The minimum learning rate at the end of training.
    decay_after: If true, do exponential decay after the cosine day,
        by a factor of 2 every max_steps.
    spike_steps: The number of steps for the initial spike.
    spike_lr: The maximum LR during the initial spike.

  Returns:
    The learning rate for the current step.
  """
  assert max_lr > min_lr

  pi = float(np.pi)
  step_ramp = jnp.minimum(step, max_steps) / max_steps  # ramp: 0 to 1.0.

  lrate = (1 + jnp.cos(pi * step_ramp)) * 0.5   # ranges from 1 to 0.
  lrate = min_lr + lrate * (max_lr - min_lr)

  if spike_steps > 0 and spike_lr > 0.0:
    assert spike_lr > max_lr
    spike_lrate = spike_lr * ((spike_steps - step) / spike_steps)
    lrate = jnp.maximum(lrate, spike_lrate)

  if decay_after:
    exp_lrate = lr_exponential_decay(step, max_steps,
                                     max_lr=2*min_lr, min_lr=min_lr)
    lrate = jnp.where(step < max_steps, lrate, exp_lrate)

  return lrate

