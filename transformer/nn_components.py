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

"""Core NN components used in models.
"""

from typing import Any, Callable, Optional, Tuple, Union

from absl import logging
from flax import linen as nn
import gin
import jax
from jax import lax
from jax.nn import initializers
import jax.numpy as jnp


PRNGKey = Any
Array = jnp.ndarray
Shape = Tuple[int]
Dtype = Union[jnp.dtype, str]


def scalar_initializer(x):
  """Like linen.zeros, but initializes a parameter to a scalar value."""
  def init_fun(key, shape, dtype):
    del key
    return jnp.broadcast_to(jnp.array(x, dtype=dtype), shape)
  return init_fun


def swish(x: Array) -> Array:
  """Swish function, which is very similar to gelu."""
  return x * nn.sigmoid(x)


def soft_abs(x: Array) -> Array:
  """Soft version of absolute value, that is smoothly differentiable."""
  return jnp.sqrt(jnp.square(x) + 1) - 1


def get_activation_function(fname: Optional[str]) -> Callable[[Array], Array]:
  """Get activation function from the specified string."""
  if fname is None:
    return lambda x: x
  elif fname == "relu":
    return nn.relu
  elif fname == "swish":
    return swish
  elif fname == "sigmoid":
    return nn.sigmoid
  elif fname == "tanh":
    return nn.tanh
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
                  rng_function: Callable[[], jax.random.KeyArray],
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
      logging.info("mlp: gate bias = %r", g)
      return (state * g) + (y_out * (1 - g))

    elif self.gate_type == "full":
      # Normal GRU style gate -- compute g using both a kernel and bias.
      g = jax.nn.sigmoid(self.gate_layer(y_hidden) + 1)  # biased to remember
      logging.info("mlp: gate full = %r", g)
      return (state * g) + (y_out * (1 - g))

    elif self.gate_type == "lstm":
      # LSTM style gate with input and forget gates.
      fg = jax.nn.sigmoid(self.forget_gate(y_hidden) + 1)  # biased to remember
      ig = jax.nn.sigmoid(self.input_gate(y_hidden) - 1)
      logging.info("mlp: gate lstm = %r, %r", ig, fg)
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
    logging.info("mlp: final = %r", y_out)

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
