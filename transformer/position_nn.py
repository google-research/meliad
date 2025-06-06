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

import types
from typing import Any, Callable, Optional, Tuple

import einops
from flax import linen
import gin
import jax.numpy as jnp
from transformer import position
import numpy as np
import scipy


Array = Any


@gin.configurable
class NNRelativePositionBiases(linen.Module):
  """Computes learnable relative position buckets.

  Attributes:
    num_heads: Number of heads in the attention layer. Each head will get a
      different relative position weighting.
    dtype: Type of arrays through this module.
    num_hidden: How many hidden units to learn for each layer.
    max_coeff_out: The maximum allowed magnitude of the entries of the hidden ->
      output matrix. Enforcing a maximum magnitude seems to help with learning
      stability.
    translation_stddev: The standard deviation of the normal distribution used
      to compute the bias term in the input -> hidden computation.
  """

  num_heads: int
  dtype: Any
  param_dtype: Any = jnp.float32
  num_hidden: int = 128
  max_coeff_out: float = 3.0
  translation_stddev: float = 10.0

  @linen.compact
  def __call__(
      self,
      num_queries: int,
      num_keys: int,
      offset: Optional[int] = None,
      bidirectional: bool = True,
  ) -> Array:
    # Find the distance between each query and each key.
    # The last N queries are lined up with the last N keys
    # (which is appropriate for XL/sliding window).
    relative_position = position.relative_positions_np(
        num_queries=num_queries, num_keys=num_keys, offset=offset
    )
    relative_position = jnp.asarray(relative_position, dtype=self.dtype)
    # The hidden layer is shared among the various heads to prevent it from
    # becoming monstrously large and slow.
    dense_1_kernel = self.param(
        "dense_1_kernel",
        # TODO(cstaats): Explore using a unit normal gaussian.
        linen.initializers.ones_init(),
        (self.num_hidden,),
        self.param_dtype,
    )
    # We rely on the randomly initialized horizontal translation to break the
    # symmetry among the hidden units initially.
    bias = self.param(
        "bias",
        linen.initializers.normal(stddev=self.translation_stddev),
        (self.num_hidden,),
        self.param_dtype,
    )
    bias = einops.rearrange(bias, "i -> i 1 1")
    # TODO(cstaats): Should the bias term be added *before* the einsum so that
    # it better represents horizontal translation of the hidden unit as a
    # function of position?
    hidden_preactivations = (
        jnp.einsum("i,qk->iqk", dense_1_kernel, relative_position) + bias
    )
    # TODO(cstaats): Make the activation function configurable.
    # TODO(cstaats): Find a default setup with a more conventional activation.
    hidden_activations = jnp.arcsinh(hidden_preactivations)
    # For the hidden -> output matrix, each head has its own row of learnable
    # weights, so the different heads can learn different position functions
    # even though they share a hidden layer.
    dense_2_kernel = self.param(
        "dense_2_kernel",
        linen.linear.default_kernel_init,
        (self.num_heads, self.num_hidden),
        self.param_dtype,
    )
    # The following line of code confines the coefficients to the range
    # [-self.max_coeff_out, self.max_coeff_out].
    # TODO(cstaats): Try using tanh rather than sin for clamping.
    dense_2_kernel = self.max_coeff_out * jnp.sin(
        dense_2_kernel / self.max_coeff_out
    )
    # TODO(cstaats): Try clamping the output rather than the coefficients.
    values = jnp.einsum("hi,iqk->hqk", dense_2_kernel, hidden_activations)
    out = einops.rearrange(values, "h q k -> 1 h q k")
    return out


@gin.configurable
def exponent_for_sequence(sequence_length: float, final_result: float) -> float:
  """Computes an exponent base for generating a desired sequence.

  Computes a float b such that the sequence
  [b**0 - 1, b**1 - 1, ..., b**sequence_length - 1]
  will start at 0 and end at final_result.

  Args:
    sequence_length: length of the desired sequence.
    final_result: desired value of b**sequence_length - 1.

  Returns:
    A float b with the property described above (up to floating-point
    computation error).
  """
  return (final_result + 1) ** (1 / (sequence_length - 1))


@gin.configurable
def asinh_exponential_translations(
    np: types.ModuleType,  # pylint: disable=redefined-outer-name
    pos: Array,
    basis_size: int = gin.REQUIRED,
    exponent_base: float = gin.REQUIRED,
) -> Array:
  """Computes asinh translated right by exponential sequence."""
  pos = einops.rearrange(pos, "t -> t 1")
  column_indices = np.asarray(range(basis_size), dtype=pos.dtype)
  column_indices = einops.rearrange(column_indices, "i -> 1 i")
  return np.arcsinh(pos - (exponent_base**column_indices - 1))


@gin.configurable
def relu_exponential_translations(
    np: types.ModuleType,  # pylint: disable=redefined-outer-name
    pos: Array,
    basis_size: int = gin.REQUIRED,
    exponent_base: float = gin.REQUIRED,
) -> Array:
  """Computes relus translated right by exponential sequence."""
  pos = einops.rearrange(pos, "t -> t 1")
  column_indices = np.asarray(range(basis_size), dtype=pos.dtype)
  column_indices = einops.rearrange(column_indices, "i -> 1 i")
  return np.maximum(0.0, pos - (exponent_base**column_indices - 1))


@gin.configurable
def powers(
    np: types.ModuleType,  # pylint: disable=redefined-outer-name
    pos: Array,
    basis_size: int = gin.REQUIRED,
    max_power: float = 2.0,
) -> Array:
  """Computes pos**p where p ranges linearly over (0, max_power]."""
  pos = einops.rearrange(pos, "t -> t 1")
  exponents = np.linspace(
      max_power, 0, basis_size, endpoint=False, dtype=pos.dtype
  )
  exponents = einops.rearrange(exponents, "i -> 1 i")
  return np.power(pos, exponents)


@gin.configurable
class OrthogonalBasisPositionBias(linen.Module):
  """Computes a learnable position function bias for a specified function space.

  Given an arbitrary vector of (linearly independent) position functions,
  computes a position function (for each head) that is a learnable linear
  combination of these position functions. During initialization, the specified
  position functions are (essentially) replaced by an approximately orthonormal
  basis for the same function space in order to produce better gradients for the
  coefficients.

  Attributes:

  mode: Used to distinguish whether training-only transformations (e.g.,
    dropout) should be applied.
  num_heads: Number of heads in the attention layer. Each head will learn a
    different set of coefficients, hence a different relative position function.
  dtype: Type of arrays through this module.
  basis_size: The number of basis functions. Equivalently, the dimension of the
    vector space of expressible relative position bias functions. So far,
    the best results are obtained when this is no bigger than 7 or 8.
  initial_basis_functions: Conceptually, the vector of functions whose linear
    combinations form the expressible functions of position.

    Should be implemented as a function to convert a length-n vector of
    positions to an (n, basis_size) matrix expressing the values of the
    (non-orthogonal) basis functions at those positions. The first argument
    should be either `np` or `jnp` so that computations can be done with or
    without jax.
  orthogonality_range: The (start, stop, num) parameters for the numpy.linspace
    on which orthonormality is computed.
  scramble_orthogonal_basis: If true, a random orthogonal transformation (NOT
    permutation matrix, despite the name) is applied within the function space
    to make the different coefficients behave more uniformly. Without this,
    later elements of the orthogonal basis will tend to have more oscillations
    and steep slopes than earlier basis functions.
  coeff_initializer: Initializer for the orthogonal_basis_coeffs param. Defaults
    to `linen.initializers.zeros_init()`, which tends to produce stable training
    (avoids blowups) but may be suboptimal.
  """

  mode: str
  num_heads: int
  dtype: Any
  basis_size: int = gin.REQUIRED
  initial_basis_functions: Callable[[types.ModuleType, Array], Array] = (
      asinh_exponential_translations
  )
  orthogonality_range: Tuple[float, float, int] = (0, 1024, 4096)
  scramble_orthogonal_basis: bool = False
  coeff_initializer: Callable[..., Any] = linen.initializers.zeros_init()

  def setup(self):
    t = np.linspace(*self.orthogonality_range, endpoint=False, dtype=np.float64)
    cols = self.initial_basis_functions(np, t)
    q, r = np.linalg.qr(cols)
    # Scale so that r has unit eigenvalues (which should, theoretically, make
    # it better conditioned for inverting).
    eigenvalues = np.diag(r)
    q *= eigenvalues[np.newaxis, :]
    r /= eigenvalues[:, np.newaxis]
    self._r_inv = jnp.asarray(
        scipy.linalg.solve_triangular(r, np.eye(self.basis_size)),
        dtype=jnp.float32,
    )
    if self.scramble_orthogonal_basis:
      # Multiply by a random orthogonal matrix to make the order of the original
      # basis functions less significant.
      self._r_inv = (
          self._r_inv
          @ scipy.stats.ortho_group(dim=self.basis_size, seed=42).rvs()
      )

    self._orthogonal_basis_coeffs = self.param(
        "orthogonal_basis_coeffs",
        self.coeff_initializer,
        (self.basis_size, self.num_heads),
        jnp.float32,
    )

  def __call__(
      self,
      num_queries: int,
      num_keys: int,
      offset: Optional[int] = None,
      bidirectional: bool = True,
  ) -> Array:
    # Find the distance between each query and each key.
    # The last N queries are lined up with the last N keys
    # (which is appropriate for XL/sliding window).
    relative_position = position.relative_positions_np(
        num_queries=num_queries, num_keys=num_keys, offset=offset
    )
    relative_position = jnp.asarray(relative_position, dtype=self.dtype)
    # TODO(cstaats): Enable bidirectional functions of position.
    relative_position = jnp.abs(relative_position)
    relative_position = einops.rearrange(relative_position, "q k -> (q k)")
    hidden_activations = self.initial_basis_functions(jnp, relative_position)
    values = hidden_activations @ (self._r_inv @ self._orthogonal_basis_coeffs)
    out = einops.rearrange(values, "(q k) h -> 1 h q k", q=num_queries)
    return out
