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

"""Memory components used in Melodi.
"""

from typing import Any, Callable, Literal, Optional, Tuple
from absl import logging

from flax import linen as nn
import gin
import jax
import jax.numpy as jnp

from transformer import attention
from transformer import nn_components
from transformer import position
from transformer import transformer_base
from transformer import transformer_layer

KVITupleFlaxVars = Tuple[Any, Any, Any]
MemTupleFlaxVars = Tuple[Any, Any]
vshape = nn_components.vshape


@gin.configurable
class TokenMixing(nn.Module):
  """Token mixing class.

  Token mixing is a simple linear transformation that mixes the input sequence
  and summary sequence (along the sequence dimension).

  output_size: the number of tokens in the output. This could be an arbitrary
    number of tokens. e.g. mixing 512 context tokens xs and 64 short-term memory
    tokens and output 32 tokens (with the same embedding dimension) as the 
    new long-term memory increment.
  use_norm_and_residual: whether to use normalization and residual.
  dtype: the dtype of the token mixing.
  """
  output_size: int
  use_norm_and_residual: bool = True
  dtype: Any = jnp.float32

  def setup(self):
    if self.use_norm_and_residual:
      self.layernorm1 = nn_components.LayerNorm()
      self.layernorm2 = nn_components.LayerNorm()

    init = jax.nn.initializers.variance_scaling(
        scale=1.0, mode="fan_in", distribution="truncated_normal")

    self.lin_mix = nn.Dense(self.output_size,
                            use_bias=True,
                            kernel_init=init,
                            dtype=self.dtype)

  def __call__(self,
               xs: jax.Array,
               us: jax.Array) -> jax.Array:
    """Token mixing.

    Args:
      xs: input sequence of shape:
          (batch_size, sequence_length, embedding_dim)
      us: summary sequence of shape:
          (batch_size, summary_length, embedding_dim)

    Returns:
      mixed sequence of shape
          (batch_size, output_size, embedding_dim)
    """

    if self.use_norm_and_residual:
      # check that the summary sequence length is equal to the output size when
      # using norm and residual.
      (_, us_len, _) = us.shape
      assert us_len == self.output_size

      xs_ln = self.layernorm1(xs)
      us_ln = self.layernorm2(us)
    else:
      xs_ln = xs
      us_ln = us

    xus_ln = jnp.concatenate([xs_ln, us_ln], axis=1)

    # Swap the sequence and embedding dimensions
    # before: xus_lin has shape: (batch_size, sequence_length, output_size)
    # after: xus_swap has shape: (batch_size, output_size, sequence_length)
    # Thus, the following linear transformation is applied to each token in the
    # sequence dimension.
    xus_swap = jnp.swapaxes(xus_ln, 1, 2)
    res_swap = self.lin_mix(xus_swap)
    out = jnp.swapaxes(res_swap, 1, 2)

    if self.use_norm_and_residual:
      out = out + us

    logging.info("token-mixing: out = %s", vshape(out))
    return out


@gin.configurable
class MelodiBaseLayer(nn.Module):
  """Melodi Base Layer.

  Extending a transformer base layer by adding token mixing.
  """
  mode: str
  batch_size: int
  embedding_dim: int

  num_heads: int
  head_size: int
  mlp_dim: int
  window_length: int  # this is only used for setting relative position in setup

  relative_position_type: Literal[
      None, "fourier", "t5", "nn", "rotary", "orthogonal", "alibi"
  ] = "fourier"
  attn_dropout_rate: float = 0.0

  short_term_memory_size: int = 64
  summary_size: int = 64
  post_ffn_dropout: bool = True
  token_mixing: bool = True
  use_cross_attention: bool = False
  cross_attention_dedicated_kv: bool = False
  dtype: Any = jnp.float32

  def is_training(self) -> bool:
    return self.mode == "train"

  def _get_dropout_rng(self):
    return self.make_rng("dropout")

  def setup(self):

    # standard transformer base layer
    # cross_attention_dedicated_kv is True to use separate k,v.
    self.tbase = transformer_base.TransformerBase(
        mode=self.mode,
        embedding_size=self.embedding_dim,
        num_heads=self.num_heads,
        head_size=self.head_size,
        mlp_dim=self.mlp_dim,
        use_cross_attention=self.use_cross_attention,
        cross_attention_dedicated_kv=self.cross_attention_dedicated_kv,
        num_position_embeddings=0,
        dtype=self.dtype)

    # relative position
    self.relative_positions = transformer_layer.get_relative_positions(
        self.relative_position_type,
        self.num_heads,
        self.window_length,
        self.mode,
        self.dtype
    )

    # token mixing
    self.token_mixing_summary = None
    self.token_mixing_short = None
    if self.token_mixing:
      self.token_mixing_summary = TokenMixing(
          output_size=self.summary_size,
          use_norm_and_residual=True,
          dtype=self.dtype
      )

      use_norm_and_residual = (self.summary_size == self.short_term_memory_size)
      self.token_mixing_short = TokenMixing(
          output_size=self.short_term_memory_size,
          use_norm_and_residual=use_norm_and_residual,
          dtype=self.dtype
      )
    else:
      # when not using token mixing, summary_size must be equal to
      # short_term_memory_size.
      assert self.summary_size == self.short_term_memory_size

  def pre_attention(self, xs: jax.Array) -> jax.Array:
    return self.tbase.pre_attention(xs)

  def kvq(self, xs: jax.Array) -> transformer_base.KVQTuple:
    return self.tbase.kvq(xs)

  def pre_cross_attention(self, xs: jax.Array) -> jax.Array:
    return self.tbase.pre_cross_attention(xs)

  def self_attention(self, xs: jax.Array,
                     us: jax.Array,
                     short_ms: jax.Array,
                     kvq_xs: KVITupleFlaxVars,
                     kvq_us: KVITupleFlaxVars,
                     dropout_mask_function:
                     Optional[Callable[[int, int], Any]] = None
                     ) -> jax.Array:
    """Self-attention for the context and summary sequences.

    It concats the short-term memory sequence to the context (xs) and 
    summary (us) sequences, applies self-attention, and then outputs 
    the update of context and summary.

    Args:
      xs: The context sequence.
      us: The summary sequence.
      short_ms: The short-term memory sequence.
      kvq_xs: The keys, values, and queries for the input sequence.
      kvq_us: The keys, values, and queries for the summary sequence.
      dropout_mask_function: The dropout mask function.

    Returns:
      The output of the self-attention.
    """

    _, xs_len, _ = xs.shape
    _, us_len, _ = us.shape

    (keys_x, values_x, queries_x) = kvq_xs
    (keys_u, values_u, queries_u) = kvq_us
    logging.info("melodi-base: keys_x = %s, values_x = %s, queries_x = %s",
                 vshape(keys_x), vshape(values_x), vshape(queries_x))
    logging.info("melodi-base: keys_u = %s, values_u = %s, queries_u = %s",
                 vshape(keys_u), vshape(values_u), vshape(queries_u))

    if short_ms is None:
      short_ms_len = 0
      keys_w = jnp.concatenate([keys_x, keys_u], axis=1)
      values_w = jnp.concatenate([values_x, values_u], axis=1)
    else:
      _, short_ms_len, _ = short_ms.shape
      short_pre_attn = self.pre_attention(short_ms)
      keys_short, values_short, _ = self.kvq(short_pre_attn)
      keys_w = jnp.concatenate([keys_short, keys_x, keys_u], axis=1)
      values_w = jnp.concatenate([values_short, values_x, values_u], axis=1)

    # connect context x and summary u
    queries_w = jnp.concatenate([queries_x, queries_u], axis=1)

    # relative position
    num_queries = xs_len + us_len
    num_keys = short_ms_len + xs_len + us_len
    rel_position_bias = self.relative_positions(
        num_queries, num_keys, bidirectional=False)
    assert rel_position_bias is not None

    # attention scale factor
    attention_scale_factor = self.tbase.self_attention_scale_factor()

    # causal mask
    kqpos = position.relative_positions(num_queries, num_keys)
    causal_mask = (kqpos < 0)

    # dropout multiplier
    dropout_multiplier = dropout_mask_function(num_queries, num_keys)

    # apply attention
    importance_w = None
    logging.info("melodi-base: single window attention.")
    attn_xu = attention.simple_attention(
        keys_w, values_w, queries_w, importance_w,
        relative_position_bias=rel_position_bias,
        scale_factor=attention_scale_factor,
        causal_mask=causal_mask,
        dropout_multiplier=dropout_multiplier,
        dtype=self.dtype)
    logging.info("melodi-base: attn_xu = %s", vshape(attn_xu))

    return attn_xu

  def cross_attention(self, xs: jax.Array,
                      us: jax.Array,
                      kv_long: Tuple[Any, Any],
                      kvq_xs: KVITupleFlaxVars,
                      kvq_us: KVITupleFlaxVars,
                      dropout_mask_function:
                      Optional[Callable[[int, int], Any]] = None
                      ) -> jax.Array:
    """Cross-attention for the context and summary sequences.

    Both context and summary sequences attend to the long-term memory.

    Args:
      xs: The context sequence.
      us: The summary sequence.
      kv_long: The keys and values of the long-term memory.
      kvq_xs: The keys, values, and queries for the input sequence.
      kvq_us: The keys, values, and queries for the summary sequence.
      dropout_mask_function: The dropout mask function.

    Returns:
      The output of the cross-attention.
    """

    (_, xs_len, _) = xs.shape
    (_, us_len, _) = us.shape
    (_, long_ms_len, _, _) = kv_long[0].shape
    logging.info("melodi-base: cross attention")
    logging.info("melodi-base: xs = %s, us = %s, long_ms = %s",
                 vshape(xs), vshape(us), vshape(kv_long[0]))

    keys_long, values_long = kv_long

    (_, _, queries_x) = kvq_xs
    (_, _, queries_u) = kvq_us

    # connect context x and summary u
    queries_w = jnp.concatenate([queries_x, queries_u], axis=1)

    # relative position
    num_queries = xs_len + us_len
    num_keys = long_ms_len

    # attention scale factor
    attention_scale_factor = self.tbase.self_attention_scale_factor()

    # dropout multiplier
    dropout_multiplier = dropout_mask_function(num_queries, num_keys)

    # apply attention
    importance_w = None
    logging.info("melodi-base: cross attention to long-term memory.")
    attn_cross = attention.simple_attention(
        keys_long, values_long, queries_w, importance_w,
        scale_factor=attention_scale_factor,
        dropout_multiplier=dropout_multiplier,
        dtype=self.dtype)
    logging.info("melodi-base: attn_cross = %s", vshape(attn_cross))

    return attn_cross


@gin.configurable
class MelodiShortTermMemoryLayer(MelodiBaseLayer):
  """Melodi short term memory Layer.

  The short-term memory layer takes the context and summary sequences and the
  short-term memory as input, and outputs (a) the updated context and summary
  sequences for the next layer, and (b) the updated short-term memory sequence
  for the next window. The process has three steps as follows:

  (1) self-attention between the context and summary sequences and the
      short-term memory sequence.
  (2) post-attention MLP.
  (3) token mixing between the context and summary sequences to update summary
  for the next layer and short-term memory for the next window.

  """

  def forward_window(
      self,
      xs: jax.Array,
      us: jax.Array,
      ms: Tuple[Any, Any],
      kvq_xs: KVITupleFlaxVars,
      kvq_us: KVITupleFlaxVars,
      dropout_mask_function: Optional[Callable[[int, int], Any]] = None,
  ) -> Tuple[jax.Array, Any, Any, Any]:
    """Forward pass for a context window.

    Args:
      xs: The input sequence.
      us: The summary sequence.
      ms: The memory sequence (short, long).
      kvq_xs: The keys, values, and queries for the input sequence.
      kvq_us: The keys, values, and queries for the summary sequence.
      dropout_mask_function: The dropout mask function.

    Returns:
      A tuple of the output sequence, the output summary sequence, the output
      short-term memory sequence, None (for long-term memory).
    """

    logging.info("melodi-short: forward window")
    logging.info("melodi-short: xs = %s, us = %s",
                 vshape(xs), vshape(us))

    short_ms, _ = ms
    if short_ms is None:
      logging.info("melodi-short: short_ms is None")
    else:
      logging.info("melodi-short: short_ms = %s", vshape(short_ms))

    (_, xs_len, _) = xs.shape

    # self-attention
    attn_xu = self.self_attention(xs, us, short_ms, kvq_xs, kvq_us,
                                  dropout_mask_function)
    logging.info("melodi-short: attn_xu = %s", vshape(attn_xu))

    # post-attention MLP
    xus = jnp.concatenate([xs, us], axis=1)
    xus_out = self.tbase.post_attn_ffn(
        xus, attn_xu, None
    )
    logging.info("melodi-short: xus after post-attn-ffn = %s", vshape(xus_out))

    # token mixing
    xs_out = xus_out[:, :xs_len, :]
    us_out = xus_out[:, xs_len:, :]

    short_out = None
    if short_ms is not None:
      short_out = us_out
      if self.token_mixing:
        short_out = self.token_mixing_short(xs_out, us_out)
        us_out = self.token_mixing_summary(xs_out, us_out)

    return xs_out, us_out, short_out, None


@gin.configurable
class MelodiLongTermMemoryLayer(MelodiBaseLayer):
  """Melodi long term memory Layer.

  Compared to the short-term memory layer (above), the long-term memory layer
  add three components:

  (1) Context and summary tokens cross attention to long-term memory.

  (2) Gate over self and cross attention with a learned parameter.

  (3) linear mixing of context and summary tokens to generate long-term memory
  increment.
  """

  long_term_memory_increment_size: int = 0

  def setup(self):
    super().setup()

    self.long_mem_token_mixing = TokenMixing(
        output_size=self.long_term_memory_increment_size,
        use_norm_and_residual=False,
        dtype=self.dtype
    )

    gate_shape = (self.num_heads, self.head_size)
    self.attn_gate = self.param(
        "attn_gate",
        jax.nn.initializers.truncated_normal(stddev=0.25),
        gate_shape,
        self.dtype
    )

  def _apply_gate_attn(self,
                       attn1: jax.Array,
                       attn2: jax.Array,
                       ) -> jax.Array:
    """Apply a gate combine self-attention and cross-attention.

    The self-attention also attends to the short-term memory.
    The cross-attention attends to the long-term memory.
    self.attn_gate is a learned parameter.

    Args:
      attn1: self-attention output.
      attn2: cross-attention output.

    Returns:
      The gated attention output.
    """

    alpha = jax.nn.sigmoid(self.attn_gate)
    alpha = jnp.expand_dims(alpha, axis=0)
    alpha = jnp.expand_dims(alpha, axis=0)

    logging.info("melodi-long: gate alpha after reshape = %s", vshape(alpha))
    attn_out = attn1 * alpha + attn2 * (1.0-alpha)
    logging.info("melodi-long: gate attn_out = %s", vshape(attn_out))

    return attn_out

  def forward_window(self,
                     xs: jax.Array,
                     us: jax.Array,
                     ms: Tuple[Any, Any],
                     kvq_xs: KVITupleFlaxVars,
                     kvq_us: KVITupleFlaxVars,
                     dropout_mask_function:
                     Optional[Callable[[int, int], Any]] = None) -> (
                         Tuple[jax.Array, Any, Any, Any]):
    """Forward pass for a context window.

    Args:
      xs: The input sequence.
      us: The summary sequence.
      ms: The memory sequence (short, (long_k, long_v)).
      kvq_xs: The keys, values, and queries for the input sequence.
      kvq_us: The keys, values, and queries for the summary sequence.
      dropout_mask_function: The dropout mask function.

    Returns:
      A tuple of the output sequence, the output summary sequence, the output
      short-term memory sequence, the new tokens added to the long-term memory .
    """

    logging.info("melodi-long: forward window")

    short_ms, long_kv = ms
    logging.info("melodi-long: xs = %s, us = %s, short_ms = %s, long_kv = %s",
                 vshape(xs), vshape(us), vshape(short_ms), vshape(long_kv))
    logging.info("melodi-long: long_term_memory_increment_size = %s",
                 self.long_term_memory_increment_size)

    (_, xs_len, _) = xs.shape

    # self-attention
    attn_xu = self.self_attention(
        xs, us, short_ms, kvq_xs, kvq_us, dropout_mask_function
    )
    logging.info("melodi-long: attn_xu = %s", vshape(attn_xu))

    # cross attention
    attn_cross = self.cross_attention(
        xs, us, long_kv, kvq_xs, kvq_us, dropout_mask_function
    )
    logging.info("melodi-long: attn_cross = %s", vshape(attn_cross))

    # gate attention
    attn_xu = self._apply_gate_attn(attn_xu, attn_cross)

    # post-attention MLP
    xus = jnp.concatenate([xs, us], axis=1)
    xus_out = self.tbase.post_attn_ffn(
        xus, attn_xu, None
    )
    logging.info("melodi-long: xus after post-attn-ffn = %s", vshape(xus_out))

    xs_out = xus_out[:, :xs_len, :]
    us_out = xus_out[:, xs_len:, :]

    # long-term memory increment
    long_increment = None
    if self.long_term_memory_increment_size > 0:
      long_increment = self.long_mem_token_mixing(xs_out, us_out)
      logging.info("melodi-long: long_increment = %s", vshape(long_increment))

    # token mixing (short and summary)
    short_out = us_out
    if self.token_mixing:
      short_out = self.token_mixing_short(xs_out, us_out)
      us_out = self.token_mixing_summary(xs_out, us_out)

    return xs_out, us_out, short_out, long_increment
