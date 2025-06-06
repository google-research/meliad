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

"""Core NN components used in models.
"""

import math
from typing import Any, Callable, Literal, Optional, Sequence, Tuple, Union

from absl import logging
import einops
from flax import linen as nn
import gin
import jax
from jax import lax
from jax.nn import initializers
import jax.numpy as jnp

import  metrics_summary


PRNGKey = Any
Array = jnp.ndarray
Shape = Tuple[int, ...]
Dtype = Union[jnp.dtype, str]
Initializer = initializers.Initializer


vshape = metrics_summary.vshape


def scalar_initializer(x):
  """Like linen.zeros, but initializes a parameter to a scalar value."""
  def init_fun(key, shape, dtype):
    del key
    return jnp.broadcast_to(jnp.array(x, dtype=dtype), shape)
  return init_fun


def swish(x: Array) -> Array:
  """Swish function, which is very similar to gelu."""
  return x * jax.nn.sigmoid(x)


def soft_abs(x: Array) -> Array:
  """Soft version of absolute value, that is smoothly differentiable."""
  return jnp.sqrt(jnp.square(x) + 1) - 1


def get_activation_function(fname: Optional[str]) -> Callable[[Array], Array]:
  """Get activation function from the specified string."""
  if fname is None:
    return lambda x: x
  elif fname == "relu":
    return jax.nn.relu
  elif fname == "swish":
    return swish
  elif fname == "sigmoid":
    return jax.nn.sigmoid
  elif fname == "tanh":
    return jax.nn.tanh
  elif fname == "softmax":
    return lambda x: jax.nn.softmax(x, axis=-1)
  else:
    raise ValueError("Unknown activation function %s" % fname)


# Adapted from flax.linen.softmax.
def safe_softmax(x: Array,
                 axis: Optional[Union[int, Tuple[int, ...]]] = -1,
                 min_x: Optional[Array] = None) -> Array:
  r"""Softmax function.

  Computes the function which rescales elements to the range :math:`[0, 1]`
  such that the elements along :code:`axis` sum to :math:`1`.

  This version of softmax is intended for use with causal attention masks, and
  safely covers the situation where all elements are masked out.  If min_x is
  not None, then probabability will be distributed between the values in x, and
  min_x.  If x >> min_x, then the probability allocated to min_x will be zero,
  and this function will be the same as the usual softmax.  However, if
  x << min_x, (because all the values in x are masked out) then probability
  will be allocated to min_x instead, and the probability allocated to x will
  be 0.  I.e., attention will attend to nothing if everything is masked out.

  .. math ::
    \mathrm{softmax}(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

  Args:
    x: input array
    axis: the axis or axes along which the softmax should be computed. The
      softmax output summed across these dimensions should sum to :math:`1`.
      Either an integer or a tuple of integers.
    min_x: the value of a minimum element which will be included in the
      softmax sum.  The value of min_x should be small when compared to the
      expected values in x.  If all of the values in x are smaller than
      min_x, then probability will be allocated to the minimum element
      instead, and the result of softmax will sum to less than 1.

  Returns:
    An array of the same shape as x.
  """
  # Subtract maximum value in x for numerical stability, so that the exponent
  # never exceeds numerical precision.
  x_max = lax.stop_gradient(jnp.max(x, axis, initial=min_x, keepdims=True))
  if min_x is not None:
    min_x = jnp.asarray(min_x, dtype=x.dtype)
    x_max = jnp.maximum(x_max, min_x)
  unnormalized = jnp.exp(x - x_max)
  x_sum = jnp.sum(unnormalized, axis=axis, keepdims=True)
  if min_x is not None:
    x_sum = x_sum + jnp.exp(min_x - x_max)
  return unnormalized / x_sum


def joint_softmax(xs: Sequence[Array],
                  axis: Optional[Union[int, Tuple[int, ...]]] = -1,
                  min_x: Optional[Array] = None) -> Sequence[Array]:
  r"""Joint softmax function over a list of arrays.

  This function is the same as safe_softmax(), except that it performs softmax
  over a list of arrays, not just a single array.  The result is the same as
  performing softmax if the arrays in xs were concatenated together, but it
  avoids the memory overhead of doing a concatenate operation.  In other words,
  the denominator of softmax is summed over all arrays in xs.

  Args:
    xs: list of input arrays.
    axis: the axis or axes along which the softmax should be computed. The
      softmax output summed across these dimensions should sum to :math:`1`.
      Either an integer or a tuple of integers.
    min_x: the value of a minimum element which will be included in the
      softmax sum.  The value of min_x should be small when compared to the
      expected values in x.  If all of the values in x are smaller than
      min_x, then probability will be allocated to the minimum element
      instead, and the result of softmax will sum to less than 1.

  Returns:
    An sequence of arrays of the same shape as xs.
  """

  if min_x is not None:
    min_x = jnp.asarray(min_x, dtype=xs[0].dtype)

  # Subtract maximum value in x for numerical stability, so that the exponent
  # never exceeds numerical precision.
  all_x_max = None
  for x in xs:
    ximax = lax.stop_gradient(
        jnp.max(x, axis=axis, initial=min_x, keepdims=True)
    )
    if all_x_max is None:
      all_x_max = ximax
    else:
      # All arrays in xs must have the same shape, except the axis dimension.
      assert all_x_max.shape == ximax.shape
      all_x_max = jnp.maximum(all_x_max, ximax)

  # Compute a total sum over all arrays.
  unnormalized_xs = []
  total_ux_sum = None
  for x in xs:
    ux = jnp.exp(x - all_x_max)
    unnormalized_xs.append(ux)
    ux_sum = jnp.sum(ux, axis=axis, keepdims=True)
    if total_ux_sum is None:
      total_ux_sum = ux_sum
    else:
      total_ux_sum += ux_sum

  # Adjust sum for presence of min_x.  See safe_softmax for details.
  if min_x is not None:
    total_ux_sum += jnp.exp(min_x - all_x_max)

  # Divide by sum and return.
  return [(ux / total_ux_sum) for ux in unnormalized_xs]


def dropout_multiplier_mask(rng, dropout_rate: float, shape: Shape,
                            dtype: Dtype):
  """Returns an array which can be multiplied by an input to perform dropout.

  Args:
    rng: A random number generator.
    dropout_rate: The rate at which to drop.
    shape: The shape of the output array.
    dtype: The type of the output array.

  Returns:
    An array of given shape, where values are { 0.0, 1.0/keep_probibility. }.
  """
  if dropout_rate <= 0.0:
    return jnp.ones(shape, dtype=dtype)

  logging.info("dropout mask: %s", shape)
  keep_prob = 1.0 - dropout_rate
  keep = jax.random.bernoulli(rng, keep_prob, shape)
  dropout_multiplier = (keep.astype(dtype) / jnp.asarray(keep_prob, dtype))
  return dropout_multiplier


def tiled_dropout(x: Array, shape: Shape, dropout_rate: float,
                  rng_function: Callable[[], jax.Array],
                  deterministic: bool) -> Array:
  """Tiles a dropout mask over a larger array.

  This will generate a smaller dropout mask of the given shape, and tile it
  over a larger array, which reduces the computational cost and memory
  associated with generating a large dropout mask.

  Args:
    x: The input array.
    shape: The shape of the dropout mask to tile.
    dropout_rate: The rate at which to drop.
    rng_function: A function which returns a random number generator, e.g.
                  lambda. self.make_rng("dropout").  The function will not
                  be called if dropout is not enabled.
    deterministic: If True, don't do dropout.

  Returns:
    An array of the same shape as x, with some values dropped out.
  """
  if deterministic or dropout_rate <= 0.0:
    return x

  if x.ndim != len(shape):
    raise ValueError("Shapes must have same number of dimensions %r, %r." %
                     (x.shape, shape))

  # Normally the tile shape is smaller than x.shape; that's the whole point.
  # If the shape tile is larger than x.shape, then trim the larger dimensions
  # instead of throwing an error.
  min_shape = tuple([
      min(xsh, ssh) for (xsh, ssh) in zip(x.shape, shape)
  ])
  if min_shape != shape:
    logging.info("tiled dropout: trimming tile from shape %s to %s",
                 shape, min_shape)
    shape = min_shape

  # if x.shape is larger than the tile shape, but not evenly divisible by it
  # the tile shape reduce by half until it is evenly divisible.
  # (minimum tile size is 8).
  shape_list = list(shape)
  for index, (xd, sd) in enumerate(zip(x.shape, shape)):
    if xd % sd != 0:
      logging.info("tiled dropout before: x.shape[%d] = %d, shape[%d] = %d",
                   index, xd, index, sd)
      while sd > 8 and xd % sd != 0:
        sd = sd // 2
      if (xd % sd) != 0:
        raise ValueError("Incompatible shapes %r, %r" % (xd, sd))

      shape_list[index] = sd
      logging.info("tiled dropout after: x.shape[%d] = %d, shape[%d] = %d",
                   index, xd, index, sd)
  shape = tuple(shape_list)

  # Check that all dimensions are evenly divisible by the tile shape.
  for (xd, sd) in zip(x.shape, shape):
    if (xd % sd) != 0:
      raise ValueError("Incompatible shapes %r, %r" % (x.shape, shape))

  # Get random number generator for dropout.
  rng = rng_function()

  repeats = [(1 if sd == 1 else xd // sd) for (xd, sd) in zip(x.shape, shape)]
  logging.info("tiled dropout %r, tile: %r", x.shape, shape)

  dtype = x.dtype
  keep_prob = 1.0 - dropout_rate
  keep = jax.random.bernoulli(rng, keep_prob, shape)
  keep = jnp.tile(keep, repeats)
  keep = jnp.broadcast_to(keep, x.shape)
  x_scaled = x / jnp.asarray(keep_prob, dtype=dtype)
  return lax.select(keep, x_scaled, jnp.zeros_like(x, dtype=dtype))


def scale_initializer(scale: float, initializer: Initializer) -> Initializer:
  def scaled_initializer(*args, **kwargs):
    return scale * initializer(*args, **kwargs)
  return scaled_initializer


@gin.configurable
class AltConv(nn.Module):
  """Computes a one-dimensional, stride 1 conv with CAUSAL padding and bias.

  The convolution is along axis -2 of the input, which is assumed to be the
  sequence dimension.

  This seems to be a bit faster than linen.Conv for our specific use case
  (long sequence, kernel size 4, etc).

  Attributes:
    features: number of convolution filters.
    receptive_field: shape (i.e., length) of the (necessarily one-dimensional)
      convolution kernel.
    kernel_init: initializer for the convolution kernel.
    bias_init: initializer for the bias.
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
  """

  features: int
  receptive_field: int
  kernel_init: Initializer = initializers.lecun_normal()
  bias_init: Initializer = initializers.zeros
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32

  def setup(self):
    dense_modules = []
    for i in range(self.receptive_field):
      dense_modules.append(
          nn.Dense(
              self.features,
              use_bias=True,
              dtype=self.dtype,
              param_dtype=self.param_dtype,
              kernel_init=scale_initializer(
                  1 / math.sqrt(self.receptive_field), self.kernel_init
              ),
              bias_init=self.bias_init,
              name=f"dense_{i}",
          )
      )
    self._dense_modules = dense_modules

  def __call__(self, x: Array) -> Array:
    if self.receptive_field == 0:
      return x
    # TODO(cstaats): Explore concatenating the rolled arrays along the embedding
    # axis and using a single dense matmul. (The concatenation approach should
    # be preferred if it is not slower.)
    retv = self._dense_modules[0](x)
    for i in range(1, self.receptive_field):
      retv += self._dense_modules[i](
          # TODO(cstaats): Explore alternative optimizations. In particular,
          # there's no reason to roll the final elements back to the beginning
          # unless the roll operation has specific optimizations that would not
          # otherwise apply.
          jnp.roll(x, i, axis=-2)
          .at[..., :i, :]
          .set(0)
      )
    return retv


@gin.configurable
class NeuralEmbeddingTable(nn.Module):
  """Implements a two-layer MLP with an optional skip connection."""

  input_embedding_size: int
  output_embedding_size: int
  virtual_vocab_size: int = gin.REQUIRED
  skip_type: Literal["concat", "add", "none"] = "none"
  use_final_layernorm: bool = False
  hidden_activation: Literal["relu", "softmax"] = "relu"
  initializer_scale: float = 1.0
  dtype: Any = jnp.float32   # The dtype used for computations.

  final_activation: Literal[None, "relu"] = None
  conv_length: int = 1

  def setup(self):
    kernel_init = jax.nn.initializers.variance_scaling(
        scale=self.initializer_scale,
        mode="fan_in",
        distribution="truncated_normal",
    )
    self.initial_layer_norm = LayerNorm(dtype=self.dtype)
    if self.conv_length == 1:
      self.query_to_key = nn.Dense(
          self.virtual_vocab_size,
          use_bias=True,
          kernel_init=kernel_init,
          dtype=self.dtype,
      )
    else:
      self.query_to_key = AltConv(
          features=self.virtual_vocab_size,
          receptive_field=self.conv_length,
          kernel_init=kernel_init,
          dtype=self.dtype,
          name="conv",
      )
    num_output_features = self.output_embedding_size
    if self.skip_type == "concat":
      # Output "meaning vec" will be concatenated with input "spelling vec".
      num_output_features -= self.input_embedding_size
      assert num_output_features > 0
    else:
      assert self.skip_type in ("add", "none")
    self.key_to_value = nn.Dense(
        num_output_features,
        use_bias=True,
        kernel_init=kernel_init,
        dtype=self.dtype,
    )
    if self.use_final_layernorm:
      self.final_layer_norm = LayerNorm(dtype=self.dtype)

  def __call__(
      self,
      x: Array,
      apply_dropout: bool = False,
      dropout_rate: float = 0.0,
      rng_function: Optional[Callable[[], Any]] = None,
  ) -> Array:
    x = jnp.asarray(x, dtype=self.dtype)
    assert jnp.shape(x)[-1] == self.input_embedding_size
    if self.skip_type != "none":
      orig_embedding = x
    else:
      orig_embedding = None  # Get rid of spurious lint warnings.

    x = self.initial_layer_norm(x)
    h = self.query_to_key(x)
    logging.info("NeuralEmbeddingTable: activation = %s",
                 self.hidden_activation)
    if self.hidden_activation == "relu":
      h = nn.relu(h)
    elif self.hidden_activation == "softmax":
      # Ordinarily, h must be scaled by 1/sqrt(x) for softmax, but this should
      # be done already by query_to_key, given the fan_in initialization.
      h = nn.softmax(h, axis=-1)
    else:
      raise ValueError(f"Invalid activation type {self.hidden_activation}.")
    y_out = self.key_to_value(h)

    if apply_dropout:
      logging.info("neural_embedding_table: dropout rate = %s", dropout_rate)
      if rng_function is None:
        raise ValueError("rng_function must be specified for dropout.")
      drop_tile_shape = (1, 128, self.output_embedding_size)
      y_out = tiled_dropout(
          y_out,
          shape=drop_tile_shape,
          dropout_rate=dropout_rate,
          rng_function=rng_function,
          deterministic=False,
      )

    if self.use_final_layernorm:
      y_out = self.final_layer_norm(y_out)

    if self.final_activation is not None:
      if self.final_activation == "relu":
        y_out = nn.relu(y_out)
      else:
        raise ValueError(f"Unrecognized activation: {self.final_activation=}")

    if self.skip_type == "add":
      if self.output_embedding_size < orig_embedding.shape[-1]:
        # Add a slice of orig_embedding to the (smaller) output.
        y_out += orig_embedding[..., : self.output_embedding_size]
      elif self.output_embedding_size > orig_embedding.shape[-1]:
        # Add all of orig_embedding to a slice of the (larger) output.
        y_out = y_out.at[..., : orig_embedding.shape[-1]].add(orig_embedding)
      else:
        # Standard residual embedding.
        y_out += orig_embedding
    elif self.skip_type == "concat":
      # Concatenate "meaning" and "spelling" vectors together.
      # Can be combined with "add" on the expansion side of a funnel.
      y_out = jnp.concatenate([orig_embedding, y_out], axis=-1)
    else:
      # No residual connection of any kind.
      assert self.skip_type == "none"
    return y_out


@gin.configurable
class MLP(nn.Module):
  """Implements a multi-layer perceptron, with optional resnet or gate."""

  # Arguments to module.
  num_output_features: int                # Length of output vectors.

  # Gin configurable parameters.
  num_layers: int = gin.REQUIRED          # Number of layers in the MLP.
  num_hidden_units: int = gin.REQUIRED    # Length of hidden unit vectors.
  hidden_activation: Optional[str] = "relu"  # Hidden layer activation fn.
  final_activation: Optional[str] = None     # Final layer activation fn.
  use_bias: bool = True                   # Use a bias in each dense layer.
  gate_type: Optional[str] = None         # { "residual", "bias", "full" }
  initializer_scale: float = 1.0          # Scale of initial values.
  dtype: Any = jnp.float32

  def setup(self):
    kernel_init = jax.nn.initializers.variance_scaling(
        scale=self.initializer_scale, mode="fan_in",
        distribution="truncated_normal")

    assert self.num_layers > 0
    hlayers = []
    for i in range(0, self.num_layers - 1):
      assert self.num_hidden_units > 0
      hlayer = nn.Dense(self.num_hidden_units,
                        use_bias=self.use_bias,
                        kernel_init=kernel_init,
                        dtype=self.dtype,
                        name=f"hidden{i}")
      hlayers.append(hlayer)
    self.hidden_layers = hlayers
    self.output_layer = nn.Dense(self.num_output_features,
                                 use_bias=self.use_bias,
                                 kernel_init=kernel_init,
                                 dtype=self.dtype)

    if self.gate_type is None or self.gate_type == "residual":
      return

    # We use a low but non-zero bias so that adafactor knows how to scale it.
    gate_bias_init = jax.nn.initializers.normal(stddev=0.1)
    # Also use a lower than normal kernel.
    gate_kernel_init = jax.nn.initializers.variance_scaling(
        scale=0.1, mode="fan_in", distribution="truncated_normal")

    if self.gate_type == "bias":
      self.gate_bias = self.param("gate_bias", gate_bias_init,
                                  (self.num_output_features,), jnp.float32)
    elif self.gate_type == "full":
      self.gate_layer = nn.Dense(self.num_output_features,
                                 use_bias=True,
                                 bias_init=gate_bias_init,
                                 kernel_init=gate_kernel_init,
                                 dtype=self.dtype)
    elif self.gate_type == "lstm":
      self.input_gate = nn.Dense(self.num_output_features,
                                 use_bias=True,
                                 bias_init=gate_bias_init,
                                 kernel_init=gate_kernel_init,
                                 dtype=self.dtype)
      self.forget_gate = nn.Dense(self.num_output_features,
                                  use_bias=True,
                                  bias_init=gate_bias_init,
                                  kernel_init=gate_kernel_init,
                                  dtype=self.dtype)
    else:
      raise ValueError("Unsupported gate_type: %s" % self.gate_type)

  def _gate(self, y_hidden: Array, state: Array, y_out: Array) -> Array:
    """Compute the value to use for the gate."""

    if self.gate_type == "residual":
      # Residual connection: just add y_out to the state.
      logging.info("mlp: residual")
      return state + y_out

    elif self.gate_type == "bias":
      # Simple gate: use a gru_style gate with a learned bias (no kernel).
      bias = jnp.asarray(self.gate_bias, dtype=self.dtype)
      bias = jnp.reshape(bias, (1,) * (y_out.ndim - 1) + (-1,))  # batch dims.
      g = jax.nn.sigmoid(bias)
      logging.info("mlp: gate bias = %s", vshape(g))
      return (state * g) + (y_out * (1 - g))

    elif self.gate_type == "full":
      # Normal GRU style gate -- compute g using both a kernel and bias.
      g = jax.nn.sigmoid(self.gate_layer(y_hidden) + 1)  # biased to remember
      logging.info("mlp: gate full = %s", vshape(g))
      return (state * g) + (y_out * (1 - g))

    elif self.gate_type == "lstm":
      # LSTM style gate with input and forget gates.
      fg = jax.nn.sigmoid(self.forget_gate(y_hidden) + 1)  # biased to remember
      ig = jax.nn.sigmoid(self.input_gate(y_hidden) - 1)
      logging.info("mlp: gate lstm = %s, %s", vshape(ig), vshape(fg))
      return (state * fg) + (y_out * ig)

    else:
      raise ValueError("Unsupported gate type %s" % self.gate_type)

  def __call__(self, x: Array, state: Optional[Array],
               apply_dropout: bool = False,
               dropout_rate: float = 0.0,
               drop_tile_shape: Optional[Shape] = None,
               rng_function: Optional[Callable[[], Any]] = None) -> Array:
    """Apply the multi-layer perceptron to the input x.

    For simple MLPs, returns f(x), where f is the MLP function.
    For resnets and gated architectures, it returns
      state + f(x)            -- for resnet.
      g*state + (1-g)*f(x)    -- for gated architecture, where g is the gate.

    Args:
      x: The input to the MLP.
      state: The prior value, if this MLP is used as part of a resnet or gated
             architecture.
      apply_dropout: If true, applies dropout to the result.
      dropout_rate: The dropout rate to use.
      drop_tile_shape: The dropout tile shape.
      rng_function: Gets a random number seed for dropout.

    Returns:
      The combination of f(x) and the (optional) prior state.
    """

    x = jnp.asarray(x, self.dtype)
    hidden_act_fun = get_activation_function(self.hidden_activation)
    final_act_fun = get_activation_function(self.final_activation)
    if self.hidden_layers:
      # Apply some number of hidden layers.
      y = x
      for layer in self.hidden_layers:
        logging.info("mlp: hidden %d, %s", self.num_hidden_units,
                     self.hidden_activation)
        y = hidden_act_fun(layer(y))
    else:
      # Apply the hidden activation function to the input.
      logging.info("mlp: activation = %s", self.hidden_activation)
      y = hidden_act_fun(x)

    y_hidden = y  # The hidden layer right before the output.
    logging.info("mlp: final activation = %s", self.final_activation)
    y_out = self.output_layer(y_hidden)  # The MLP final output.
    y_out = final_act_fun(y_out)         # Apply final activation function.
    logging.info("mlp: final = %s", vshape(y_out))

    # Optionally apply dropout to the output.
    if apply_dropout:
      if drop_tile_shape is None:
        raise ValueError("drop_tile_shape must be specified for dropout.")
      if rng_function is None:
        raise ValueError("rng_function must be specified for dropout.")
      logging.info("mlp: dropout rate = %s", dropout_rate)
      y_out = tiled_dropout(
          y_out, shape=drop_tile_shape, dropout_rate=dropout_rate,
          rng_function=rng_function, deterministic=False)

    if state is None:
      # Simple MLP.  No gate to combine y_out with the state.
      assert self.gate_type is None
      logging.info("mlp: gate type = None.")
      return y_out

    # When using state, gate_type must be specified.
    assert self.gate_type is not None
    return self._gate(y_hidden, state, y_out)


# TODO(delesley) This functionality should be moved out of this class, and
# out of the MLP class above, and moved into transformer_base.  The interface
# to MLP is unwieldy due to technical debt from past experiments.
@gin.configurable
class MLPWrapper(nn.Module):
  """A wrapper which does MLP + residual gate + dropout."""

  # Arguments to module.
  num_output_features: int             # Length of output vector.

  # Gin configurable parameters.
  mlp_factory: Any = gin.REQUIRED

  # TODO(delesley): Usually passed by caller -- should be gin-configured.
  num_hidden_units: int = gin.REQUIRED

  # TODO(delesley): Unused -- for compatibility with TransformerBase.
  gate_type: Any = None
  final_activation: Any = None

  # Usually passed by caller.
  dtype: Any = jnp.float32

  def setup(self):
    self.mlp_layer = self.mlp_factory(self.num_output_features,
                                      num_hidden_units=self.num_hidden_units,
                                      dtype=self.dtype)

  def __call__(self, x: Array, state: Optional[Array],
               apply_dropout: bool = False,
               dropout_rate: float = 0.0,
               drop_tile_shape: Optional[Shape] = None,
               rng_function: Optional[Callable[[], Any]] = None) -> Array:
    """Apply an MLP to the input x.

    Implements the MLP interface expected by TransformerBase.
    For simple MLPs, returns f(x), where f is the MLP function.
    If state is provided, returns f(x) + state, where f is the MLP function.

    Args:
      x: The input to the MLP.
      state: The prior value, if this MLP is used as part of a resnet or gated
             architecture.
      apply_dropout: If true, applies dropout to the result.
      dropout_rate: The dropout rate to use.
      drop_tile_shape: The dropout tile shape.
      rng_function: Gets a random number seed for dropout.

    Returns:
      The combination of f(x) and the (optional) prior state.
    """

    # Apply MLP.
    y_out = self.mlp_layer(x)

    # Optionally apply dropout to the output.
    if apply_dropout:
      if drop_tile_shape is None:
        raise ValueError("drop_tile_shape must be specified for dropout.")
      if rng_function is None:
        raise ValueError("rng_function must be specified for dropout.")
      logging.info("mlp: dropout rate = %s", dropout_rate)
      y_out = tiled_dropout(
          y_out, shape=drop_tile_shape, dropout_rate=dropout_rate,
          rng_function=rng_function, deterministic=False)

    # Residual connection from the output back onto state.
    if state is not None:
      y_out += state

    return y_out


@gin.configurable
class SimpleMLP(nn.Module):
  """A Simple MLP with nonlinearity."""

  # Arguments to module.
  num_output_features: int             # Length of output vector.

  # Usually passed by caller -- see MLPWrapper.
  num_hidden_units: int = 4096

  # Gin configurable parameters.
  hidden_activation: Optional[str] = "relu"

  # Usually passed by caller.
  dtype: Any = jnp.float32

  def setup(self):
    kernel_init = jax.nn.initializers.variance_scaling(
        scale=1.0, mode="fan_in", distribution="truncated_normal")
    if self.hidden_activation == "softmax":
      # The "fan_in" is inappropriate for softmax, since it assumes that the
      # input values have a normal distribution.
      logging.info("Using appropriate 2nd layer kernel for softmax.")
      out_kernel_init = jax.nn.initializers.truncated_normal(stddev=1.0)
    else:
      out_kernel_init = kernel_init

    self.input_layer = nn.Dense(self.num_hidden_units,
                                use_bias=True,
                                kernel_init=kernel_init,
                                dtype=self.dtype)
    self.output_layer = nn.Dense(self.num_output_features,
                                 use_bias=True,
                                 kernel_init=out_kernel_init,
                                 dtype=self.dtype)

  def __call__(self, x: Array) -> Array:
    logging.info("SimpleMLP: x = %s", vshape(x))
    h = self.input_layer(x)   # (..., num_hidden)
    logging.info("SimpleMLP: h = %s", vshape(h))
    actfun = get_activation_function(self.hidden_activation)
    h = actfun(h)             # apply relu, etc.
    logging.info("SimpleMLP: activation_fun = %s", self.hidden_activation)
    y = self.output_layer(h)  # (..., num_output_features)
    logging.info("SimpleMLP: y = %s", vshape(y))
    return y


@gin.configurable()
class MultiHeadMLP(nn.Module):
  """A variation on the MLP module which does multi-head attention instead."""

  # Arguments to module.
  num_output_features: int             # Length of output vectors.

  # Usually passed by caller -- see MLPWrapper.
  num_hidden_units: int = gin.REQUIRED

  # Gin configurable parameters.
  num_heads: int = 8
  head_dim: int = 128
  normalize_keys_queries: bool = True  # Do key/query normalization.
  param_dtype: Any = jnp.float32       # dtype at which parameters are stored
  hidden_activation: Optional[str] = "softmax"

  # Usually passed by caller.
  dtype: Any = jnp.float32             # dtype used for computations

  def setup(self):
    kernel_init = jax.nn.initializers.variance_scaling(
        scale=1.0, mode="fan_in", distribution="truncated_normal")

    num_queries = self.num_heads * self.head_dim
    self.query_projection_layer = nn.Dense(num_queries,
                                           use_bias=False,
                                           kernel_init=kernel_init,
                                           dtype=self.dtype)
    self.output_projection_layer = nn.Dense(self.num_output_features,
                                            use_bias=False,
                                            kernel_init=kernel_init,
                                            dtype=self.dtype)

    k_init = jax.nn.initializers.truncated_normal(stddev=1.0)
    if self.hidden_activation == "softmax":
      # No fan-in for softmax.
      v_init = k_init
    else:
      # Same as variance scaling with fan-in, but the value for fan-in
      # in this case should self.num_hidden_units, not num_kv_hidden.
      v_stddev = float(1.0 / math.sqrt(self.num_hidden_units))
      v_init = jax.nn.initializers.truncated_normal(stddev=v_stddev)

    # We use 2D parameters to keep Adafactor happy, and reshape them later.
    num_kv_hidden = self.num_hidden_units * self.num_heads
    self.trainable_keys = self.param("keys",
                                     k_init,
                                     (num_kv_hidden, self.head_dim),
                                     self.param_dtype)
    self.trainable_values = self.param("values",
                                       v_init,
                                       (num_kv_hidden, self.head_dim),
                                       self.param_dtype)
    # Used for normalized keys/queries.
    self.attention_scale = self.param("attention_scale",
                                      jax.nn.initializers.ones,
                                      (self.num_heads,),
                                      self.param_dtype)

  def _normalize_kq(self, kq: Array) -> Array:
    """Normalize function for keys and queries."""
    epsilon = jnp.array(1.0e-6, dtype=self.dtype)
    kq_sum_sqr = jnp.sum(jnp.square(kq), axis=-1, keepdims=True)
    norm_kq = kq * jax.lax.rsqrt(kq_sum_sqr + epsilon)
    return jnp.asarray(norm_kq, dtype=self.dtype)

  def __call__(self, x: Array) -> Array:
    x = jnp.asarray(x, dtype=self.dtype)
    logging.info("mh_mlp: x = %s", vshape(x))

    keys = jnp.asarray(self.trainable_keys, dtype=self.dtype)
    values = jnp.asarray(self.trainable_values, dtype=self.dtype)
    attn_scale = jnp.asarray(self.attention_scale, dtype=self.dtype)

    # Note -- we store keys and values as 2D arrays for simplicity;
    # optimizers like Adafactor cannot handle 3D arrays without annotations.
    # There is no cost to these reshapes on TPU.
    num_hidden = self.num_hidden_units
    keys = keys.reshape(num_hidden, self.num_heads, self.head_dim)
    values = values.reshape(num_hidden, self.num_heads, self.head_dim)
    logging.info("mh_mlp: keys = %s", vshape(keys))
    logging.info("mh_mlp: values = %s", vshape(values))

    # Project x to a set of queries.
    queries = self.query_projection_layer(x)  # (..., nh * hd)
    queries = einops.rearrange(queries,
                               "... (h d) -> ... h d",
                               h=self.num_heads,
                               d=self.head_dim)
    logging.info("mh_mlp: queries = %s", vshape(queries))

    if self.normalize_keys_queries:
      logging.info("mh_mlp: normalize keys, queries.")
      # Keys and queries are first normalized to unit vectors, so the
      # dot-product is the cosine of the angle between them.  We then multiply
      # by a learned attention_scale, which controls the "temperature" of
      # attention.  (https://arxiv.org/abs/2010.04245)
      keys = self._normalize_kq(keys)
      queries = self._normalize_kq(queries)
      keys = keys * attn_scale.reshape(1, self.num_heads, 1)
    else:
      # Scale queries by 1 / sqrt(d) when using unnormalized keys & queries.
      d_scale = jax.lax.rsqrt(float(self.head_dim)).astype(self.dtype)
      queries = queries * d_scale

    # Compute the attention matrix
    attn = jnp.einsum("...hd,khd->...hk", queries, keys)
    logging.info("mh_mlp: attn = %s", vshape(attn))

    # Apply activation function, which is usually softmax.
    logging.info("mh_mlp: activation_function = %s", self.hidden_activation)
    activation_fun = get_activation_function(self.hidden_activation)
    attn = activation_fun(attn)

    # Compute weighted sum of values.
    yvals = jnp.einsum("...hk,khd->...hd", attn, values)
    logging.info("mh_mlp: yvals = %s", vshape(yvals))
    yvals = einops.rearrange(yvals,
                             "... h d -> ... (h d)",
                             h=self.num_heads,
                             d=self.head_dim)
    logging.info("mh_mlp: yvals_reshaped = %s", vshape(yvals))

    # Project back to input dimension.
    y = self.output_projection_layer(yvals)
    logging.info("mh_mlp: y = %s", vshape(y))
    return y


@gin.configurable
def multihead_parameter_equivalent_num_hidden(mlp_dim: int,
                                              num_heads: int,
                                              head_dim: int) -> int:
  """Configures num_hidden_units for MultiHeadMLP."""
  # Returns the number of hidden units for a simple MLP, such that it is
  # FLOPs and parameter equivalent to a multi-head MLP.
  return mlp_dim + (num_heads * head_dim)


@gin.configurable
class DenseEmbedding(nn.Module):
  """DenseEmbedding does bidirectional lookups into an embedding table.

  The embedding_to_logits method will take an embedding vector, match it
  against the entries in the table, and return logits for the best matching
  entry, which can be sampled from with softmax.

  The onehot_to_embedding uses the same table for the reverse lookup: given
  a one-hot vocabulary entry (or the output of softmax), it will return the
  embedding vector.
  """

  embedding_size: int
  vocab_size: int
  embedding_stddev: float = 1.0
  dtype: Any = jnp.float32
  param_dtype: Any = jnp.float32   # Precision to store trainable params.

  def setup(self):
    embed_init = nn.initializers.normal(stddev=self.embedding_stddev,
                                        dtype=self.param_dtype)
    # Create a single table that maps embedding_vector <--> vocab_entry
    self.vocab_kernel = self.param(
        "vocab_kernel",
        embed_init,
        (self.embedding_size, self.vocab_size))

  def embedding_to_logits(self, x):
    """Given an input vector x, lookup the vector in a table, and return logits.

    The logits can be converted to a (soft) one-hot via softmax.

    Args:
      x: Input vectors of shape (..., embedding_size)

    Returns:
      Logits of shape (..., vocab_size)
    """
    x = jnp.asarray(x, self.dtype)
    logging.info("dense_embed_to_logits: x = %r", vshape(x))
    vkernel = jnp.asarray(self.vocab_kernel, self.dtype)
    logits = einops.einsum(x, vkernel, "... e, e v -> ... v")
    # Rescale logits to a reasonable range for softmax.
    logits = logits / jnp.sqrt(x.shape[-1]).astype(self.dtype)
    return logits

  def onehot_to_embedding(self, x):
    """Given a (soft) one-hot, return the embedding vector.

    This operation inverts embeddings_to_logits.  Given an one-hot vector,
    this will return a weighted sum of embedding vectors.

    one_hot_to_embeddings(softmax(embeddings_to_logits(x))) ~= x

    Args:
      x: Input vectors of shape (..., vocab_size)

    Returns:
      Vectors of shape (..., embedding_size)
    """

    x = jnp.asarray(x, self.dtype)
    logging.info("dense_onehot_to_embed: x = %r", vshape(x))
    vkernel = jnp.asarray(self.vocab_kernel, self.dtype)
    y = einops.einsum(x, vkernel, "... v, e v -> ... e")
    return y


# Modified slightly from the flax implementation.
@gin.configurable
class LayerNorm(nn.Module):
  """Layer normalization (https://arxiv.org/abs/1607.06450).

  Operates on the last axis of the input data.

  It normalizes the activations of the layer for each given example in a
  batch independently, rather than across a batch like Batch Normalization.
  i.e. applies a transformation that maintains the mean activation within
  each example close to 0 and the activation standard deviation close to 1.

  Attributes:
    epsilon: A small float added to variance to avoid dividing by zero.
    dtype: the dtype of the computation (default: float32).
    use_bias:  If True, bias (beta) is added.
    use_scale: If True, multiply by scale (gamma).
    use_mean: If True, compute and adjust for the mean.
      Note that that T5X layernorm does not use the mean.
      Empirically, ignoring the mean can stabilize learning in transformers.
    use_scalar_scale_bias: If True, using a single scalar for scale & bias.
    enable_layernorm: If False, does not perform layernorm.
    bias_init: Initializer for bias, by default, zero.
    scale_init: Initializer for scale, by default, one.
  """
  epsilon: float = 1e-6
  dtype: Any = jnp.float32
  use_scale: bool = True               # Apply a learned scale.
  use_bias: bool = False               # Apply a learned bias.
  use_mean: bool = False               # Calculate and adjust for the mean.
  use_scalar_scale_bias: bool = False  # Learn a single scalar scale & bias.
  enable_layernorm: bool = True        # Turn off layernorm if false.
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
  scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones

  @nn.compact
  def __call__(self, x):
    """Applies layer normalization on the input.

    Args:
      x: the inputs

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    if not self.enable_layernorm:
      return x
    x = jnp.asarray(x)

    # Calculate mean and variance at higher precision.
    xf = jnp.asarray(x, jnp.float32)
    if self.use_mean:
      mean = jnp.mean(xf, axis=-1, keepdims=True)
      xf = xf - mean
    else:
      mean = None  # kill lint warning.
    var = jnp.mean(lax.square(xf), axis=-1, keepdims=True)
    mul = lax.rsqrt(var + self.epsilon)

    # Rescale x
    # if not use_mean, then rescale around zero instead. (A simplification.)
    if self.use_mean:
      y = (x - mean) * mul
    else:
      y = x * mul

    if self.use_scalar_scale_bias:
      # Learn a single scalar value for bias and scale.
      # (Which mirrors the single value for mean and stddev above.)
      num_scale_bias_features = 1
    else:
      # Learn a different value per neuron/feature for bias and scale.
      num_scale_bias_features = x.shape[-1]

    # Apply learned scale and bias.
    if self.use_scale:
      y = y * jnp.asarray(
          self.param("scale", self.scale_init, (num_scale_bias_features,)),
          dtype=self.dtype)
    if self.use_bias:
      y = y + jnp.asarray(
          self.param("bias", self.bias_init, (num_scale_bias_features,)),
          dtype=self.dtype)
    return y.astype(self.dtype)


def neighbor_cosine_similarity(
    x: Array, alignment: None | Literal["left", "right"] = None, **pad_kwargs
) -> Array:
  """Computes the cosine similarity between pairs of neighboring vectors.

  Args:
    x: Array of shape [batch_size, sequence_length, num_hidden]
    alignment: If "left", padding is added on the right. If "right", padding is
      added on the left. If None, no padding is added and the returned array has
      sequence length one less than the input.
    **pad_kwargs: Arguments to be passed to jnp.pad. Should include mode if not
      'constant'. Should *not* include pad_width. Ignored if alignment is None.

  Returns:
    Array of shape [batch_size, sequence_length-1] whose (b,i)th entry is the
    cosine similarity between x[b,i] and x[b,i+1]. If alignment is "left", the
    shape is instead [batch_size, sequence_length]. If alignment is "right", the
    shape is [batch_size, sequence_length] and the (b,i)th entry is the
    similarity between x[b,i-1] and x[b,i].
  """
  # Normalize along the num_hidden dimension:
  x = x / jnp.maximum(1e-6, jnp.linalg.norm(x, axis=2, keepdims=True))
  first_neighbor = x[:, :-1, :]
  second_neighbor = x[:, 1:, :]
  cosine_similarity = jnp.einsum("bsh,bsh->bs", first_neighbor, second_neighbor)
  if alignment is not None:
    pad_width = (
        (0, 0),  # Don't pad the batch dimension!
        (1, 0) if alignment == "right" else (0, 1),  # sequence length padding
    )
    cosine_similarity = jnp.pad(cosine_similarity, pad_width, **pad_kwargs)
  return cosine_similarity
