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

"""Base class for transformer layers."""

from typing import Any, Dict, Optional, Tuple

from absl import logging

from flax import linen as nn
import gin
import jax
import jax.numpy as jnp

from transformer import nn_components


Array = Any
OptArray = Optional[Array]

# Tuple of keys,values,queries
KVQTuple = Tuple[OptArray, OptArray, OptArray]


vshape = nn_components.vshape


@gin.configurable
class KVQLayer(nn.Module):
  """Generate keys, values, and queries for attention.

  Self-attention and cross-attention each have their own KVQLayer;
  See TransformerBase, below.
  """

  # Options set by parent (TransformerBase)
  embedding_size: int
  num_heads: int
  head_size: int

  # Compute queries.  Usually True.
  has_queries: bool = True

  # Compute keys,values.
  # May be set to False for the cross-attention KVQLayer, if keys/values are
  # shared between self and cross-attention. (E.g. block-recurrent transformer.)
  has_kv: bool = True

  # Normalize keys and queries.
  normalize_keys: bool = True

  # Usually set by parent.
  dtype: Any = gin.REQUIRED

  def setup(self):
    kernel_init = nn.initializers.variance_scaling(
        scale=1.0, mode="fan_in", distribution="truncated_normal")

    # Compute keys, values, queries.
    # Disable bias.  This prevents a failure mode whereby the attention matrix
    # can become filled with very large uniform values, due to high bias.
    if self.has_kv:
      self.keys_layer = nn.Dense(
          features=self.num_heads * self.head_size,
          use_bias=False,   # No bias for keys.
          kernel_init=kernel_init,
          dtype=self.dtype)
      self.values_layer = nn.Dense(
          features=self.num_heads * self.head_size,
          use_bias=False,   # No bias for values.
          kernel_init=kernel_init,
          dtype=self.dtype)
    if self.has_queries:
      self.queries_layer = nn.Dense(
          features=self.num_heads * self.head_size,
          use_bias=False,   # No bias for queries.
          kernel_init=kernel_init,
          dtype=self.dtype)

    # When normalizing keys and queries, attention must be scaled with
    # learned parameters.
    if self.normalize_keys:
      self.attention_scale = self.param("attention_scale",
                                        jax.nn.initializers.ones,
                                        (self.num_heads,), jnp.float32)

  def attention_scale_factor(self) -> Optional[Array]:
    """Returns the attention scale, when keys and queries are normalized."""
    if self.normalize_keys:
      # Take absolute value to ensure if the attention_scale becomes negative,
      # the temperature will still be positive and learnable.
      return jnp.abs(jnp.asarray(self.attention_scale, dtype=self.dtype))
    else:
      return None

  def __call__(self,
               xs_q: Array,
               xs_kv: Optional[Array]) -> KVQTuple:
    """Takes a embeddings as input, and returns keys, values, queries.

    For self-attention, xs_kv should be None, in which it will default to the
    same array as xs_q.  For cross-attention, they should be different arrays.

    Args:
      xs_q: input sequence for computing queries of shape:
          (batch_size, query_sequence_length, embedding_size)
      xs_kv: input sequence for computing keys,values to attend over:
          (batch_size, kv_sequence_length, embedding_size)
          If None, defaults to xs_q for self-attention.

    Returns:
      (keys, values, queries) of shape
          (batch_size, sequence_length, num_heads, head_size)
    """

    # Check input shapes.
    (batch_size, num_queries, _) = xs_q.shape
    if xs_kv is not None:
      (batch_size_kv, num_keys, _) = xs_kv.shape
      assert batch_size_kv == batch_size
    else:
      xs_kv = xs_q
      num_keys = num_queries

    # Compute queries.
    if self.has_queries:
      queries = self.queries_layer(xs_q)  # (b, num_q, num_heads * head_size)
      logging.info("kvq: queries = %s", vshape(queries))
      q_shape = (batch_size, num_queries, self.num_heads, self.head_size)
      queries = jnp.reshape(queries, q_shape)
    else:
      queries = None

    # Compute keys and values.
    if self.has_kv:
      keys = self.keys_layer(xs_kv)  # (b, num_k, num_heads * head_size)
      values = self.values_layer(xs_kv)
      logging.info("kvq: keys = %s, values = %s", vshape(keys), vshape(values))
      kv_shape = (batch_size, num_keys, self.num_heads, self.head_size)
      keys = jnp.reshape(keys, kv_shape)
      values = jnp.reshape(values, kv_shape)
    else:
      keys = None
      values = None

    if self.normalize_keys:
      # Normalize both keys and queries.
      # The learned attention_scale_factors() will return non-None.
      logging.info("kvq: normalize keys, queries.")
      if keys is not None:
        keys = normalize_kq(keys, self.dtype)
      if queries is not None:
        queries = normalize_kq(queries, self.dtype)
    else:
      # Scale queries by 1 / sqrt(d) when using unnormalized keys, queries.
      d_scale = jax.lax.rsqrt(float(self.head_size)).astype(self.dtype)
      logging.info("kvq: scale queries by 1/sqrt(d).")
      if queries is not None:
        queries = queries * d_scale

    # Return keys, values, and queries.
    return (keys, values, queries)


@gin.configurable
class TransformerBase(nn.Module):
  """TransformerBase implements everything except attention.

  It handles:
    - Projection to (keys, values, queries) before attention.
    - Projection MLP back to embedding_size after attention.
    - Final FFN layer.
    - layernorm, dropout, and normalization of keys and queries.

  This functionality is ecapsulated here so that it can be reused with more
  complicated attention mechanisms.
  """

  # Options set by parent module (TransformerLayer).
  mode: str
  embedding_size: int
  num_heads: int
  head_size: int
  mlp_dim: int

  use_self_attention: bool = True     # Compute kqv for self-attention.
  use_cross_attention: bool = False   # Compute kqv for cross-attention.
  num_position_embeddings: int = 0    # Learned position embeddings.

  # Compute separate keys,values for use with cross-attention.
  cross_attention_dedicated_kv: bool = True

  # Configurable hyperparameters.
  attn_mlp_factory: Any = gin.REQUIRED
  ffn_factory: Any = gin.REQUIRED

  normalize_keys: bool = True
  dropout_rate: float = 0.0
  pre_attn_dropout: bool = True
  post_attn_dropout: bool = False
  pre_ffn_dropout: bool = False
  post_ffn_dropout: bool = True

  # Usually set by parent.
  dtype: Any = gin.REQUIRED

  def is_training(self) -> bool:
    return self.mode == "train"

  def _get_dropout_rng(self):
    return self.make_rng("dropout")

  def setup(self):
    # Keys, values, queries for self-attention.
    if self.use_self_attention:
      self._kvq = KVQLayer(
          self.embedding_size, self.num_heads, self.head_size,
          has_queries=True,
          has_kv=True,
          dtype=self.dtype)

    # Keys, values, queries for cross-attention.
    if self.use_cross_attention:
      # If cross_attention_dedicated_kv, then compute a separate set of
      # keys, values, for use with cross-attention. (Otherwise keys,values
      # are shared with e.g. self-attention.)
      self._cross_kvq = KVQLayer(
          self.embedding_size, self.num_heads, self.head_size,
          has_queries=True,
          has_kv=self.cross_attention_dedicated_kv,
          dtype=self.dtype)

    # Post-attention projection.
    self.post_attn_mlp = self.attn_mlp_factory(
        self.embedding_size,
        num_hidden_units=0,
        gate_type="residual",
        final_activation=None,
        dtype=self.dtype)  # pytype: disable=wrong-keyword-args  # trace-all-classes

    # Final FNN.
    self.ffn = self.ffn_factory(
        self.embedding_size,
        num_hidden_units=self.mlp_dim,
        gate_type="residual",
        final_activation=None,
        dtype=self.dtype)  # pytype: disable=wrong-keyword-args  # trace-all-classes

    # Learned position embeddings (if any) for absolute positions.
    if self.num_position_embeddings > 0:
      self.position_embeddings = self.param(
          "position_embeddings",
          jax.nn.initializers.normal(stddev=1.0),
          (self.num_position_embeddings, self.embedding_size),
          jnp.float32)

    # Layernorm.
    self.pre_attn_layernorm = nn_components.LayerNorm()
    if self.use_cross_attention:
      self.pre_cross_attn_layernorm = nn_components.LayerNorm()
    self.pre_ffn_layernorm = nn_components.LayerNorm()

  def force_init(self, xs: Array, cross_xs: OptArray):
    """Force flax initialization of self, prior to use with lax.scan.

    Args:
      xs: The input sequence that the module will be called with.
      cross_xs: An (optional) sequence to cross-attend to.
    """
    logging.info("tbase: Begin forced initialization.")
    xs = self.pre_attention(xs)
    batch_size = xs.shape[0]
    seq_len = xs.shape[1]

    # Init kvq modules.
    if self.use_self_attention:
      _ = self.kvq(xs)
    if self.use_cross_attention:
      _ = self.cross_kvq(xs, cross_xs)

    # Create a dummy attention result.
    attn_ys_shape = (batch_size, seq_len, self.num_heads, self.head_size)
    dummy_attn_ys = None
    dummy_cross_attn_ys = None
    if self.use_self_attention:
      dummy_attn_ys = jnp.zeros(attn_ys_shape, dtype=self.dtype)
    if self.use_cross_attention:
      dummy_cross_attn_ys = jnp.zeros(attn_ys_shape, dtype=self.dtype)

    # Init the MLP layer.
    _ = self.post_attn_ffn(xs, dummy_attn_ys, dummy_cross_attn_ys)
    logging.info("tbase: End forced initialization.")

  def attention_scale_factors(self) -> Dict[str, OptArray]:
    """Returns the attention scales, when keys and queries are normalized.

    Returns: {"self": self_attn_scale, "cross": cross_attn_scale}
    """
    return {
        "self": self.self_attention_scale_factor(),
        "cross": self.cross_attention_scale_factor()
    }

  def self_attention_scale_factor(self) -> OptArray:
    """Return the self-attention scales, when keys/queries are normalized."""
    if self.use_self_attention:
      return self._kvq.attention_scale_factor()
    else:
      return None

  def cross_attention_scale_factor(self) -> OptArray:
    """Return the cross-attention scales, when keys/queries are normalized."""
    if self.use_cross_attention:
      return self._cross_kvq.attention_scale_factor()
    else:
      return None

  def pre_attention(self, xs: Array) -> Array:
    """Preprocess embeddings before before attention."""

    # Pre-attention layernorm.
    logging.info("tbase: pre_attn xs = %s", vshape(xs))
    xs = jnp.asarray(xs, dtype=self.dtype)
    xs = self.pre_attn_layernorm(xs)

    # Pre-attention dropout (optional).
    if self.pre_attn_dropout and self.is_training():
      drop_tile_shape = (1, 128, self.embedding_size)
      logging.info("tbase: pre_attn dropout; rate = %s.", self.dropout_rate)
      xs = nn_components.tiled_dropout(xs,
                                       drop_tile_shape,
                                       self.dropout_rate,
                                       rng_function=self._get_dropout_rng,
                                       deterministic=False)

    # Add (optional) learned position embeddings.
    # Note that these are /not/ added to the residual path.
    if self.num_position_embeddings > 0:
      (_, num_kq, _) = xs.shape
      if num_kq != self.num_position_embeddings:
        raise ValueError(f"Number of keys/queries {num_kq} does not match "
                         f"number of learned position embeddings "
                         f"{self.num_position_embeddings}.")
      logging.info("tbase: Using learned position embeddings.")
      xs_pos = jnp.asarray(self.position_embeddings, dtype=self.dtype)
      xs_pos = jnp.expand_dims(xs_pos, 0)  # Add batch dimension.
      xs = xs + xs_pos

    return xs

  def pre_cross_attention(self, cross_xs: Array) -> Array:
    """Preprocess embeddings before before cross-attention."""

    # Pre-attention layernorm.
    logging.info("tbase: pre_attn cross_xs = %s", vshape(cross_xs))
    assert self.use_cross_attention
    cross_xs = jnp.asarray(cross_xs, dtype=self.dtype)
    cross_xs = self.pre_cross_attn_layernorm(cross_xs)

    # Pre-attention dropout (optional).
    if self.pre_attn_dropout and self.is_training():
      drop_tile_shape = (1, 128, self.embedding_size)
      logging.info("tbase: pre_cross_attn dropout; rate = %s.",
                   self.dropout_rate)
      cross_xs = nn_components.tiled_dropout(cross_xs,
                                             drop_tile_shape,
                                             self.dropout_rate,
                                             rng_function=self._get_dropout_rng,
                                             deterministic=False)
    return cross_xs

  def kvq(self, xs: Array) -> KVQTuple:
    """Compute keys, values, queries for self-attention."""
    assert self.use_self_attention
    return self._kvq(xs, None)

  def cross_kvq(self, xs_q: Array, xs_kv: Array) -> KVQTuple:
    """Compute keys, values, queries for cross-attention."""
    assert self.use_cross_attention
    return self._cross_kvq(xs_q, xs_kv)

  def post_attn_ffn(self,
                    xs: Array,
                    attn_ys: Optional[Array],
                    cross_attn_ys: Optional[Array]) -> Array:
    """Combines the output of attention with the original input sequence.

    Post-attn MLP on attn_ys, followed by resnet/gate.
    Note that either attn_ys or cross_attn_ys (or both) must be non-None.
    Pre-FFN layernorm and dropout, then the FFN layer, followed by resnet/gate.

    Args:
      xs: Original input sequence (residual path) of shape
          (batch_size, sequence_length, embedding_size)
      attn_ys: Output of the self-attention module, of shape
          (batch_size, sequence_length, num_heads, head_size)
      cross_attn_ys: Output of the cross-attention module, of shape
          (batch_size, sequence_length, num_heads, head_size)

    Returns:
      Array of shape (batch_size, sequence_length, embedding_size)
    """

    (batch_size, sequence_length, _) = xs.shape
    if attn_ys is not None:
      assert attn_ys.shape == (batch_size, sequence_length,
                               self.num_heads, self.head_size)
    if cross_attn_ys is not None:
      assert cross_attn_ys.shape == (batch_size, sequence_length,
                                     self.num_heads, self.head_size)
    no_dropout = not self.is_training()

    # Combine results from attention and cross-attention if need be.
    if attn_ys is not None and cross_attn_ys is not None:
      # Concatenate self-attention and cross-attention results along the
      # heads axis, before applying the projection layer.
      logging.info("tbase: using self-attention and cross-attention.")
      attn_ys = jnp.concatenate([attn_ys, cross_attn_ys], axis=2)
      attn_ys_edim = 2 * self.num_heads * self.head_size
    elif attn_ys is not None:
      # Only use self-attention.
      logging.info("tbase: using self-attention only.")
      attn_ys_edim = self.num_heads * self.head_size
    elif cross_attn_ys is not None:
      # Only use cross-attention.
      logging.info("tbase: using cross-attention only.")
      attn_ys = cross_attn_ys
      attn_ys_edim = self.num_heads * self.head_size
    else:
      raise ValueError("Both attn_ys and cross_attn_ys are None.")

    # Reshape from num_heads * head_size
    attn_ys = attn_ys.reshape(batch_size, sequence_length, attn_ys_edim)
    logging.info("tbase: attn_ys = %s", vshape(attn_ys))

    logging.info("tbase: post-attention MLP.")
    # Standard transformer archicture.
    # The post-attention MLP applies a linear projection to project attn_ys
    # to the embedding space.  It then uses a residual connection or gate to
    # combine the projection with gate_xs.  Post-attention dropout is applied
    # before the residual/gate.
    post_attn_ys = self.post_attn_mlp(
        attn_ys, xs,
        apply_dropout=self.post_attn_dropout and not no_dropout,
        dropout_rate=self.dropout_rate,
        drop_tile_shape=(1, 128, self.embedding_size),
        rng_function=self._get_dropout_rng)
    logging.info("tbase: post_attn_ys = %s", vshape(post_attn_ys))

    # The input to the FFN; Layernorm is applied before the FFN.
    ffn_in = self.pre_ffn_layernorm(post_attn_ys)
    logging.info("tbase: pre-FFN layernorm = %s", vshape(ffn_in))

    # Pre-FFN dropout.
    if self.pre_ffn_dropout:
      logging.info("tbase: pre-FFN dropout.")
      ffn_in = nn_components.tiled_dropout(
          ffn_in, (1, 128, self.embedding_size), self.dropout_rate,
          rng_function=self._get_dropout_rng, deterministic=no_dropout)

    # FFN layer.
    # Large MLP with hidden layers followed by residual connection or gate.
    # The MLP will apply post-ffn dropout before the gate.
    logging.info("tbase: final FFN")
    ys = self.ffn(ffn_in, post_attn_ys,
                  apply_dropout=self.post_ffn_dropout and not no_dropout,
                  dropout_rate=self.dropout_rate,
                  drop_tile_shape=(1, 128, self.embedding_size),
                  rng_function=self._get_dropout_rng)

    logging.info("tbase: ys = %s", vshape(ys))
    return ys


def normalize_kq(kq: Array, dtype: Any) -> Array:
  """Normalize function for keys and queries."""
  epsilon = jnp.array(1.0e-6, dtype=dtype)
  kq_sum_sqr = jnp.sum(jnp.square(kq), axis=-1, keepdims=True)
  norm_kq = kq * jax.lax.rsqrt(kq_sum_sqr + epsilon)
  return jnp.asarray(norm_kq, dtype=dtype)
