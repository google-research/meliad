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

from typing import Any, Optional, Tuple

from absl import logging

from flax import linen as nn
import gin
import jax
import jax.numpy as jnp

from transformer import nn_components


Array = Any

# Tuple of scale factors
AttnScaleTuple = Tuple[Optional[Array], Optional[Array]]

# Tuple of keys,values,queries
KVQTuple = Tuple[Array, Array, Optional[Array], Optional[Array]]


vshape = nn_components.vshape


@gin.configurable
class KVQLayer(nn.Module):
  """Generate keys, values, and queries for attention."""

  embedding_size: int
  num_heads: int
  head_size: int
  has_queries: bool = True
  has_queries2: bool = False  # For cross-attention, e.g. decoder or recurrence.

  normalize_keys: bool = True  # Normalize keys and queries.
  num_position_embeddings: int = 0  # Learned absolute position embeddings.
  pre_attn_dropout: bool = True
  dropout_rate: float = 0.0
  dtype: Any = jnp.float32

  def setup(self):
    kernel_init = nn.initializers.variance_scaling(
        scale=1.0, mode="fan_in", distribution="truncated_normal")

    # Project to keys,values,queries
    # Disable bias.  This prevents a failure mode whereby the attention matrix
    # can become filled with very large uniform values, due to high bias.
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
    if self.has_queries2:
      self.queries2_layer = nn.Dense(
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

    # Learned position embeddings for absolute positions.
    if self.num_position_embeddings > 0:
      # Embeddings for query elements.
      self.position_embeddings = self.param(
          "position_embeddings",
          jax.nn.initializers.normal(stddev=1.0),
          (self.num_position_embeddings, self.embedding_size),
          jnp.float32)

    # Layernorm
    self.pre_attn_layernorm = nn_components.LayerNorm()

  def attention_scale_factor(self) -> Optional[Array]:
    """Returns the attention scale, when keys and queries are normalized."""
    if self.normalize_keys:
      # Take absolute value to ensure if the attention_scale becomes negative,
      # the temperature will still be positive and learnable.
      return jnp.abs(jnp.asarray(self.attention_scale, dtype=self.dtype))
    else:
      return None

  def _get_dropout_rng(self):
    return self.make_rng("dropout")

  def __call__(self, xs: Array, deterministic: bool = False) -> KVQTuple:
    """Takes a sequence of embeddings as input, and returns keys,values,queries.

    First apply pre_attn layernorm, and pre_attn dropout.
    Then add learned positional embeddings, if any.
    Return (keys, values, queries, queries2).

    Args:
      xs: input sequence of shape (batch_size, sequence_length, embedding_size)
      deterministic: if False, apply dropout.

    Returns:
      (keys, values, queries, queries2) of shape
          (batch_size, sequence_length, num_heads, head_size)
    """

    # Project inputs to (keys, values, queries).
    (batch_size, num_keys, _) = xs.shape
    drop_tile_shape = (1, 128, self.embedding_size)

    # Apply layernorm to input, rather than the output.
    # This provides better gradients through the resnet, and also avoids
    # the need for a prolonged warmup phase (https://arxiv.org/abs/2002.04745)

    # Layernorm for self-attention.
    logging.info("kvq: pre_attn xs = %s", vshape(xs))
    xs = jnp.asarray(xs, dtype=self.dtype)
    xs = self.pre_attn_layernorm(xs)

    # Add (optional) learned position embeddings.
    if self.num_position_embeddings > 0:
      assert xs.ndim == 3   # (b, sequence_length, embedding_size)
      assert xs.shape[-2] == self.num_position_embeddings
      logging.info("kvq: learned positions.")
      xs_pos = jnp.asarray(self.position_embeddings, dtype=self.dtype)
      xs_pos = jnp.expand_dims(xs_pos, 0)  # Add batch dimension.
      xs = xs + xs_pos

    # Pre-attention dropout.
    if self.pre_attn_dropout:
      logging.info("kvq: pre_attn dropout.")
      xs = nn_components.tiled_dropout(xs, drop_tile_shape, self.dropout_rate,
                                       rng_function=self._get_dropout_rng,
                                       deterministic=deterministic)

    # Compute keys and values.
    keys = self.keys_layer(xs)  # (b, num_keys, num_heads * head_size)
    values = self.values_layer(xs)

    # Compute queries and cross-attention queries if necessary.
    if self.has_queries:
      queries = self.queries_layer(xs)  # (b, num_keys, n_heads * head_size)
      logging.info("kvq: queries = %s", vshape(queries))
    else:
      queries = None
    if self.has_queries2:
      queries2 = self.queries2_layer(xs)  # (b, num_keys, n_heads * head_size)
      logging.info("kvq: queries2 = %s", vshape(queries2))
    else:
      queries2 = None

    # Reshape to split num_heads, head_size into separate dimensions.
    kv_shape = (batch_size, num_keys, self.num_heads, self.head_size)
    keys = jnp.reshape(keys, kv_shape)
    values = jnp.reshape(values, kv_shape)
    if queries is not None:
      queries = jnp.reshape(queries, kv_shape)
    if queries2 is not None:
      queries2 = jnp.reshape(queries2, kv_shape)

    if self.normalize_keys:
      # Normalize both keys and queries.
      # The learned attention_scale_factors() will return non-None.
      logging.info("kvq: normalize keys, queries.")
      keys = normalize_kq(keys, self.dtype)
      if queries is not None:
        queries = normalize_kq(queries, self.dtype)
      if queries2 is not None:
        queries2 = normalize_kq(queries2, self.dtype)
    else:
      # Scale queries by 1 / sqrt(d) when using unnormalized keys,queries.
      d_scale = jax.lax.rsqrt(float(self.head_size)).astype(self.dtype)
      logging.info("kvq: scale queries by 1/sqrt(d).")
      if queries is not None:
        queries = queries * d_scale
      if queries2 is not None:
        queries2 = queries2 * d_scale

    # Return keys, values, and queries.
    return (keys, values, queries, queries2)


@gin.configurable
class MemTransformerBase(nn.Module):
  """MemTransformerBase implements everything except attention.

  It handles:
    - Projection to (keys, values, queries) before attention.
    - Projection MLP back to embedding_size after attention.
    - Final FFN layer.
    - layernorm, dropout, and normalization of keys and queries.

  This functionality is ecapsulated here so that it can be reused with more
  complicated attention mechanisms.
  """

  # Options set by parent module.
  mode: str
  embedding_size: int
  output_embedding_size: int
  num_heads: int
  head_size: int
  mlp_dim: int

  cross_attention_q: bool = False         # Additional q for cross-attention.
  cross_attention_kv: bool = False        # Additional kv for cross-attention.
  num_position_embeddings: int = 0        # Learned position embeddings.
  num_cross_position_embeddings: int = 0  # Learned position embeddings.

  # Configurable hyperparameters.
  attn_mlp_factory: Any = gin.REQUIRED
  ffn_factory: Any = gin.REQUIRED
  gate_type: str = "residual"
  single_gate: bool = False
  skip_ffn: bool = False

  normalize_keys: bool = True
  dropout_rate: float = 0.0
  pre_attn_dropout: bool = True
  post_attn_dropout: bool = False
  pre_ffn_dropout: bool = False
  post_ffn_dropout: bool = True

  dtype: Any = jnp.float32

  def is_training(self) -> bool:
    return self.mode == "train"

  def _get_dropout_rng(self):
    return self.make_rng("dropout")

  def setup(self):
    # Keys,values,queries for self-attention; queries for cross-attention.
    self._kvq = KVQLayer(self.embedding_size, self.num_heads, self.head_size,
                         has_queries=True,
                         has_queries2=self.cross_attention_q,
                         num_position_embeddings=self.num_position_embeddings,
                         normalize_keys=self.normalize_keys,
                         pre_attn_dropout=self.pre_attn_dropout,
                         dropout_rate=self.dropout_rate,
                         dtype=self.dtype)

    # Keys,values, attention_scale for cross-attention.
    if self.cross_attention_kv:
      # Use a full kvq layer, with layernorm and attention scale.
      self._cross_kv = KVQLayer(
          self.embedding_size, self.num_heads, self.head_size,
          has_queries=False,
          has_queries2=False,
          num_position_embeddings=self.num_cross_position_embeddings,
          normalize_keys=self.normalize_keys,
          pre_attn_dropout=self.pre_attn_dropout,
          dropout_rate=self.dropout_rate,
          dtype=self.dtype)
    elif self.cross_attention_q:
      # No separate keys,values for cross-attention, but we may still need
      # cross-attention-scale, so we create our own.
      assert self.num_cross_position_embeddings == 0
      if self.normalize_keys:
        self.attention_scale2 = self.param("attention_scale2",
                                           jax.nn.initializers.ones,
                                           (self.num_heads,), jnp.float32)
    # Post-attention linear projection.
    if not self.single_gate:
      if ((self.output_embedding_size == self.embedding_size) or
          (self.output_embedding_size == self.embedding_size // 2)):
        # Normal residual connection or gate.
        post_attn_gate_type = self.gate_type
      elif self.output_embedding_size == self.embedding_size * 2:
        # If we are doubling the embedding_size, then don't use a gate.
        # The result of attention will be concatenated instead.
        post_attn_gate_type = None
      else:
        logging.fatal("Incompatible embedding dimensions %d, %d",
                      self.embedding_size, self.output_embedding_size)

      self.post_attn_mlp = self.attn_mlp_factory(
          self.embedding_size,
          num_hidden_units=0,
          gate_type=post_attn_gate_type,
          final_activation=None,
          dtype=self.dtype)  # pytype: disable=wrong-keyword-args  # trace-all-classes

    # Final FNN.
    if not self.skip_ffn:
      self.ffn = self.ffn_factory(
          self.output_embedding_size,
          num_hidden_units=self.mlp_dim,
          gate_type=self.gate_type,
          final_activation=("tanh" if self.single_gate else None),
          dtype=self.dtype)  # pytype: disable=wrong-keyword-args  # trace-all-classes

    # Layernorm.
    self.pre_ffn_layernorm = nn_components.LayerNorm()

  def force_init(self, xs: Array):
    """Force flax initialization of self, prior to use with lax.scan.

    Args:
      xs: The input sequence that the module will be called with.
    """
    logging.info("tbase: Begin forced initialization.")
    _ = self.kvq(xs)
    batch_size = xs.shape[0]
    seq_len = xs.shape[1]
    attn_ys_shape = (batch_size, seq_len, self.num_heads, self.head_size)
    dummy_attn_ys = jnp.zeros(attn_ys_shape, dtype=self.dtype)
    if self.cross_attention_kv or self.cross_attention_q:
      dummy_cross_attn_ys = dummy_attn_ys
    else:
      dummy_cross_attn_ys = None
    _ = self.post_attn_ffn(xs, dummy_attn_ys, dummy_cross_attn_ys)
    logging.info("tbase: End forced initialization.")

  def attention_scale_factors(self) -> AttnScaleTuple:
    """Returns the attention scales, when keys and queries are normalized.

    Returns: (scale for kv (i.e. queries), scale for cross_kv (i.e queries2))
    """
    sfactor = self._kvq.attention_scale_factor()
    if self.cross_attention_kv:
      cross_sfactor = self._cross_kv.attention_scale_factor()
    elif self.cross_attention_q and self.normalize_keys:
      # Take absolute value to ensure if the attention_scale2 becomes negative,
      # the temperature will still be positive and learnable.
      cross_sfactor = jnp.abs(
          jnp.asarray(self.attention_scale2, dtype=self.dtype))
    else:
      cross_sfactor = None
    return (sfactor, cross_sfactor)

  def kvq(self, xs: Array) -> KVQTuple:
    enable_dropout = self.pre_attn_dropout and self.is_training()
    return self._kvq(xs, deterministic=not enable_dropout)

  def cross_kv(self, xs: Array) -> Tuple[Array, Array]:
    assert self.cross_attention_kv
    enable_dropout = self.pre_attn_dropout and self.is_training()
    (k, v, _, _) = self._cross_kv(xs, deterministic=not enable_dropout)
    return (k, v)

  def post_attn_ffn(self, xs: Array, attn_ys: Array,
                    cross_attn_ys: Optional[Array]) -> Array:
    """Combines the output of attention with the original input sequence.

    Post-attn MLP on attn_ys, followed by resnet/gate.
    Pre-FFN layernorm and dropout, then the FFN layer, followed by resnet/gate.

    Args:
      xs: Original input sequence of shape
          (batch_size, sequence_length, embedding_size)
      attn_ys: Output of the self-attention module, of shape
          (batch_size, sequence_length, num_heads, head_size)
      cross_attn_ys: Output of the cross-attention module, of shape
          (batch_size, sequence_length, num_heads, head_size)

    Returns:
      Array of shape (batch_size, sequence_length, embedding_size)
    """

    (batch_size, sequence_length, _) = xs.shape
    assert attn_ys.shape == (batch_size, sequence_length,
                             self.num_heads, self.head_size)
    no_dropout = not self.is_training()

    # Concatenate cross-attention and self-attention results.
    if cross_attn_ys is not None:
      # Concatenate self-attention and cross-attention results, before
      # applying the projection layer.
      logging.info("tbase: using cross-attention.")
      assert attn_ys.shape == (batch_size, sequence_length,
                               self.num_heads, self.head_size)
      attn_ys = jnp.concatenate([attn_ys, cross_attn_ys], axis=2)
      att_ys_num_heads = self.num_heads * 2
    else:
      # Only use self-attention.
      att_ys_num_heads = self.num_heads

    logging.info("tbase: attn_ys = %s", vshape(attn_ys))
    attn_ys = attn_ys.reshape(
        (batch_size, sequence_length, att_ys_num_heads * self.head_size))

    if self.single_gate:
      logging.info("tbase: single gate.")
      assert not self.skip_ffn
      # Skip post-attention linear projection and residual connection.
      assert self.embedding_size == self.output_embedding_size
      ys_hidden = xs    # The FFN (below) will be gated onto xs (the input).
      ffn_in = attn_ys  # The input to the FFN is the output of attention.
    else:
      # Set up the residual or gate for a possible expansion/contraction.
      if self.output_embedding_size == self.embedding_size:
        gate_xs = xs    # Normal transformer: residual or gate onto xs.
      elif self.output_embedding_size == self.embedding_size * 2:
        gate_xs = None  # Expansion: no residual, concat with xs instead.
      elif self.output_embedding_size == self.embedding_size // 2:
        gate_xs = xs    # Contraction: residual or gate onto xs, then contract.
      else:
        logging.fatal("Incompatible embedding dimensions %d, %d",
                      self.embedding_size, self.output_embedding_size)

      logging.info("tbase: post-attention MLP.")
      # Standard transformer archicture.
      # The post-attention MLP applies a linear projection to project attn_ys
      # to the embedding space.  It then uses a residual connection or gate to
      # combine the projection with gate_xs.  Post-attention dropout is applied
      # before the residual/gate.
      post_attn_ys = self.post_attn_mlp(
          attn_ys, gate_xs,
          apply_dropout=self.post_attn_dropout and not no_dropout,
          dropout_rate=self.dropout_rate,
          drop_tile_shape=(1, 128, self.embedding_size),
          rng_function=self._get_dropout_rng)
      logging.info("tbase: post_attn_ys = %s", vshape(post_attn_ys))

      # Double or halve the embedding_size if we need to.
      if self.output_embedding_size == self.embedding_size:
        # Normal transformer.
        # The FFN (below) will be gated onto post_attn_ys.
        # The value of post_attn_ys (above), was gated onto xs.
        ys_hidden = post_attn_ys
      elif self.output_embedding_size == self.embedding_size * 2:
        # For an expansion, we double embedding_size by concatenating xs
        # and post_attn_ys, instead of gating them with a residual connection.
        logging.info("tbase: expanding embedding_dim from %d to %d",
                     self.embedding_size, self.output_embedding_size)
        ys_hidden = jnp.concatenate([xs, post_attn_ys], axis=-1)
      elif self.output_embedding_size == self.embedding_size // 2:
        # For a contraction, we halve the embedding_size by adding the
        # (previously concanated) halves together.  When an expansion is paired
        # with a contraction, the concatenation doesn't eliminate the residual
        # connection, it merely shifts it to a later (contracting) layer.
        logging.info("tbase: contracting embedding_dim from %d to %d",
                     self.embedding_size, self.output_embedding_size)
        (ysh1, ysh2) = jnp.split(post_attn_ys, 2, axis=-1)
        ys_hidden = ysh1 + ysh2
      else:
        logging.fatal("Inconceivable!")  # Can't happen: see fatal error above.

      logging.info("tbase: ys_hidden = %s", vshape(ys_hidden))

      if self.skip_ffn:
        logging.info("tbase: skip final FFN. ys = %s", vshape(ys_hidden))
        return ys_hidden

      # The input to the FFN; Layernorm is applied before the FFN.
      ffn_in = self.pre_ffn_layernorm(ys_hidden)
      logging.info("tbase: pre-FFN layernorm = %s", vshape(ffn_in))

      # Pre-FFN dropout.
      if self.pre_ffn_dropout:
        logging.info("tbase: pre-FFN dropout.")
        ffn_in = nn_components.tiled_dropout(
            ffn_in, (1, 128, self.output_embedding_size), self.dropout_rate,
            rng_function=self._get_dropout_rng, deterministic=no_dropout)

    # FFN layer.
    # Large MLP with hidden layers followed by residual connection or gate.
    # The MLP will apply post-ffn dropout before the gate.
    logging.info("tbase: final FFN")
    ys = self.ffn(ffn_in, ys_hidden,
                  apply_dropout=self.post_ffn_dropout and not no_dropout,
                  dropout_rate=self.dropout_rate,
                  drop_tile_shape=(1, 128, self.output_embedding_size),
                  rng_function=self._get_dropout_rng)

    logging.info("tbase: ys = %s", vshape(ys))
    return ys


def normalize_kq(kq: Array, dtype: Any) -> Array:
  """Normalize function for keys and queries."""
  epsilon = jnp.array(1.0e-6, dtype=dtype)
  kq_sum_sqr = jnp.sum(jnp.square(kq), axis=-1, keepdims=True)
  norm_kq = kq * jax.lax.rsqrt(kq_sum_sqr + epsilon)
  return jnp.asarray(norm_kq, dtype=dtype)
