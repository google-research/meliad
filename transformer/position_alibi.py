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

"""Class for relative position biases computed via a neural net."""

from typing import Optional

import einops
from flax import linen
import gin
import jax
import jax.numpy as jnp
from transformer import position


Array = jax.Array


gin.external_configurable(jax.nn.initializers.constant)


@gin.configurable
class BoundedALiBi(linen.Module):
  r"""Computes a modified version of the ALiBi position encoding.

  In the ALiBi paper
  @inproceedings{press2021train,
    title={Train Short, Test Long: Attention with Linear Biases Enables Input
      Length Extrapolation},
    author={Press, Ofir and Smith, Noah and Lewis, Mike},
    booktitle={International Conference on Learning Representations},
    year={2021},
    note={\url{https://openreview.net/pdf?id=R8sQPpGCv0}}
  }
  the authors propose that attention biases should have the form
  $m \cdot -\textrm{distance}$,
  where `distance` is the distance between the attender and attendee token
  (typically an integer) and $m$ is a head-specific number. They further
  propose that the slopes $m$ should be given by a geometric sequence
  $2^{-8i/n}$ where $n$ is the number of heads and $i \in \{1..n\}$.

  This implementation incorporates two modifications:

  * Either the slopes themselves are learnable (in which case the ALiBi
    sequence is used to initialize them), or there is a single scalar, shared
    across all the heads in the layer, by which all the slopes are multiplied.
  * There is a learned lower bound on the attention bias, which may be either
    head-specific or shared by the entire layer.

  In the default case, the formula for the attention bias becomes
  $\operatorname{max} \{s \cdot m \cdot \textrm{distance}, -\labs b\rabs\}$
  where $s$ is the learned scalar, $b$ is the learned bound, and $m$ is the
  slope.

  Attributes:
    num_heads:
    trainable_slopes:
    per_head_bound:
    bound_initializer: How to initialize the lower bound on the position bias.
      By default, initialized to one (which in practice means the bias is
      clamped to [-1, 1]).
  """
  num_heads: int
  trainable_slopes: bool = False
  per_head_bound: bool = False
  bound_initializer: jax.nn.initializers.Initializer = jax.nn.initializers.ones
  fixed_bound: Optional[float] = None

  def setup(self):
    alibi_slopes = jnp.asarray(
        [
            -(2 ** (-8 * i / self.num_heads))
            for i in range(1, self.num_heads + 1)
        ],
        dtype=jnp.float32,
    )
    if self.trainable_slopes:
      self._slopes = -jnp.abs(  # ensures slopes never positive
          self.param(
              "slopes",
              jax.nn.initializers.constant(alibi_slopes),
              self._slopes.shape,
              jnp.float32,
          )
      )
      # no need for trainable scale when the individual slopes are trainable
      self._scale = 1.0
    else:
      self._slopes = alibi_slopes
      self._scale = self.param(
          "scale",
          linen.initializers.ones,
          (),
          jnp.float32,
      )

    if self.fixed_bound is not None:
      bound = jnp.asarray([self.fixed_bound], dtype=jnp.float32)
    else:
      bound = self.param(
          "bound",
          self.bound_initializer,
          (self.num_heads if self.per_head_bound else 1,),
          jnp.float32,
      )
    # Note: If you initialize a 4-dimensional array, AdaFactor needs extra info
    # about how to factor it even if all but one of the dimensions are 1. So
    # instead, we create a parameter with rank one and then rearrange it to add
    # extra dimensions.
    self._bound = einops.rearrange(bound, "h -> 1 h 1 1")

  def __call__(
      self,
      relative_positions: Array,
      bidirectional: bool = True,
  ) -> Array:
    """Compute relative position biases.

    Args:
      relative_positions: Array of shape (batch_size, num_queries, num_keys).
      bidirectional: Dummy parameter included for interface compatibility.
        (BoundedALiBi is always symmetric.)

    Returns:
      Biases of shape (batch_size, num_heads, num_queries, num_keys).
    """
    del bidirectional  # Unused.
    relative_positions = jnp.abs(
        relative_positions
    )  # ensures positions always positive
    biases = jnp.einsum("bqk,h->bhqk", relative_positions, self._slopes)
    biases *= jnp.abs(self._scale)
    biases = jnp.maximum(biases, -jnp.abs(self._bound))
    return biases


@gin.configurable
class BoundedALiBiIntegerPositions(linen.Module):
  """A wrapper to easily use BoundedALiBi with index-based positions."""

  num_heads: int
  trainable_slopes: Optional[bool] = False
  per_head_bound: bool = False

  def setup(self):
    kwargs = {}
    if self.trainable_slopes is not None:
      kwargs["trainable_slopes"] = self.trainable_slopes
    if self.per_head_bound is not None:
      kwargs["per_head_bound"] = self.per_head_bound
    self._wrapped_module = BoundedALiBi(num_heads=self.num_heads, **kwargs)

  def __call__(
      self,
      num_queries: int,
      num_keys: int,
      offset: Optional[int] = None,
      bidirectional: bool = True,
  ) -> Array:
    del bidirectional  # Unused.
    # Find the distance between each query and each key.
    # The last N queries are lined up with the last N keys
    # (which is appropriate for XL/sliding window).
    relative_position = position.relative_positions_np(
        num_queries=num_queries, num_keys=num_keys, offset=offset
    )
    relative_position = jnp.asarray(relative_position, dtype=jnp.float32)
    relative_position = einops.rearrange(relative_position, "q k -> 1 q k")
    return self._wrapped_module(relative_position)
