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

"""Implements a melodi (memory with low dimensions) layer."""
from typing import Any, Dict, Literal, Optional, Sequence, Tuple
from absl import logging

from flax import linen as nn
import gin
import jax
import jax.numpy as jnp

from transformer import attention
from transformer import kv_cache
from transformer import nn_components
from transformer.memory2 import melodi_memory

Metrics = Dict[str, Any]
KVITuple = attention.KVITuple
KVITupleFlaxVars = Tuple[Any, Any, Any]
MemTupleFlaxVars = Tuple[Any, Any]
Shape = Tuple[int, ...]
vshape = nn_components.vshape


@gin.configurable
class MelodiSequenceLayer(nn.Module):
  """Melodi layer that processes a sequence of multiple context windows.

     q, k, v: processed in parallel for all windows.
     attention and FFN: processed sequentially for each window.
     each window has two options: short-term memory or long-term memory.
  """

  mode: str
  batch_size: int
  embedding_dim: int

  # Configurable hyper-parameters
  num_heads: int = gin.REQUIRED
  head_size: int = gin.REQUIRED
  mlp_dim: int = gin.REQUIRED

  window_length: int = gin.REQUIRED
  relative_position_type: Literal[
      None, "fourier", "t5", "nn", "rotary", "orthogonal", "alibi"
  ] = gin.REQUIRED
  attn_dropout_rate: float = gin.REQUIRED

  # Memory
  memory_type: Literal[None, "short", "long"] = "short"
  long_term_memory_size: int = 0
  long_term_memory_increment_size: int = 0
  short_term_memory_size: int = 128
  memory_embedding_dim: int = 0  # 0: use embedding_dim
  summary_size: int = 128

  dtype: Any = jnp.float32

  def is_training(self) -> bool:
    return self.mode == "train"

  def _get_dropout_rng(self):
    return self.make_rng("dropout")

  def _get_memory_embedding_dim(self):
    if self.memory_embedding_dim == 0:
      return self.embedding_dim  # default
    else:
      return self.memory_embedding_dim

  def _get_number_of_windows(self, sequence_length: int) -> int:
    """Returns the number of windows or blocks in the sequence."""
    if sequence_length < self.window_length:
      num_windows = 1  # Shouldn't happen, but it's not an error.
    elif sequence_length == self.window_length:
      num_windows = 1
    else:
      num_windows = sequence_length // self.window_length
      if (num_windows * self.window_length) != sequence_length:
        raise ValueError(f"Window length {self.window_length} must be a " +
                         f"multiple of sequence length {sequence_length}")
    logging.info("melodi-seq-layer: num_windows = %d.", num_windows)
    return num_windows

  def _get_attn_dropout_mask(self,
                             num_queries: int,
                             num_keys: int) -> Optional[jax.Array]:
    """Returns a mask that applies dropout to the attention matrix."""
    # The mask is supplied as floating-point values, not boolean.
    # The mask will be broadcast across batches and windows.
    is_training = (self.mode == "train")
    if self.attn_dropout_rate > 0.0 and is_training:
      dropout_rng = self.make_rng("dropout")
      attn_shape = (self.num_heads, num_queries, num_keys)
      dropout_multiplier = nn_components.dropout_multiplier_mask(
          dropout_rng, self.attn_dropout_rate, attn_shape, self.dtype
      )
      logging.info(
          "melodi-seq-layer: attn_dropout = %s", vshape(dropout_multiplier)
      )
    else:
      dropout_multiplier = None
      logging.info(
          "melodi-seq-layer: dropout_multiplier = None (is_training = %s)",
          is_training
      )
    return dropout_multiplier

  def setup(self):

    # memory cache
    self.short_memory = None
    if self.short_term_memory_size > 0:
      self.short_memory = kv_cache.MemoryCache(
          mode=self.mode,
          batch_size=self.batch_size,
          memory_size=self.short_term_memory_size,
          memory_embedding_dim=self._get_memory_embedding_dim(),
          init_from_zero=True,
          dtype=self.dtype,
      )

    self.long_memory = None
    if self.memory_type == "long" and self.long_term_memory_size > 0:
      self.long_memory = kv_cache.MemoryCache(
          mode=self.mode,
          batch_size=self.batch_size,
          memory_size=self.long_term_memory_size,
          memory_embedding_dim=self._get_memory_embedding_dim(),
          init_from_zero=True,
          dtype=self.dtype,
      )

    # melodi_layer
    if self.memory_type == "short":
      self.melodi_layer = melodi_memory.MelodiShortTermMemoryLayer(
          mode=self.mode,
          batch_size=self.batch_size,
          embedding_dim=self.embedding_dim,
          num_heads=self.num_heads,
          head_size=self.head_size,
          mlp_dim=self.mlp_dim,
          window_length=self.window_length,
          relative_position_type=self.relative_position_type,
          attn_dropout_rate=self.attn_dropout_rate,
          short_term_memory_size=self.short_term_memory_size,
          summary_size=self.summary_size,
          post_ffn_dropout=True,
          dtype=self.dtype
      )
    elif self.memory_type == "long":
      self.melodi_layer = melodi_memory.MelodiLongTermMemoryLayer(
          mode=self.mode,
          batch_size=self.batch_size,
          embedding_dim=self.embedding_dim,
          num_heads=self.num_heads,
          head_size=self.head_size,
          mlp_dim=self.mlp_dim,
          window_length=self.window_length,
          relative_position_type=self.relative_position_type,
          attn_dropout_rate=self.attn_dropout_rate,
          short_term_memory_size=self.short_term_memory_size,
          summary_size=self.summary_size,
          post_ffn_dropout=True,
          dtype=self.dtype,
          long_term_memory_increment_size=self.long_term_memory_increment_size,
      )

  def __call__(self,
               xs: jax.Array,
               us: jax.Array,
               start_of_sequence: jax.Array) -> (
                   Tuple[jax.Array, Any]):
    """Takes context xs and summary us as input, and returns updated xs and us.

    Args:
      xs: input sequence of shape:
          (batch_size, sequence_length, embedding_dim)
      us: summary sequence of shape:
          (batch_size, summary_size * num_windows, embedding_dim)
      start_of_sequence:

    Returns:
      (xs, us) of shape
          (batch_size, sequence_length, embedding_dim)
    """
    logging.info("melodi-seq-layer: memory_size = Long: %s Short: %s",
                 self.long_term_memory_size,
                 self.short_term_memory_size)
    logging.info("melodi-seq-layer: summary_size = %s", self.summary_size)
    logging.info("melodi-seq-layer: xs = %s, us = %s", vshape(xs), vshape(us))

    # Get number of windows for the sequence.
    _, sequence_length, _ = xs.shape
    number_of_windows = self._get_number_of_windows(sequence_length)
    logging.info("melodi-seq-layer: number_of_windows = %s", number_of_windows)

    # Load short-term memory
    short_ms = None
    if self.short_memory is not None:
      logging.info("melodi-seq-layer: short ms mode = %s, cache mode = %s",
                   self.mode, self.short_memory.mode)
      prev_mem = self.short_memory.load_prev_cache(start_of_sequence)
      if prev_mem is not None:
        short_ms = prev_mem
        logging.info("melodi-seq-layer: short_ms = %s", vshape(short_ms))

    # Load long-term memory
    long_ms = None
    long_k = None
    long_v = None
    if self.long_memory is not None:
      logging.info("melodi-seq-layer: long ms mode = %s, cache mode = %s",
                   self.mode, self.long_memory.mode)
      prev_mem = self.long_memory.load_prev_cache(start_of_sequence)
      if prev_mem is not None:
        long_ms = prev_mem
        logging.info("melodi-seq-layer: long_ms = %s", vshape(long_ms))
        long_pre_attn = self.melodi_layer.pre_attention(long_ms)
        (long_k, long_v, _) = self.melodi_layer.kvq(long_pre_attn)
        logging.info("melodi-seq-layer: long_k = %s, long_v = %s",
                     vshape(long_k),
                     vshape(long_v))
        logging.info("melodi-seq-layer: long_increment = %s",
                     self.long_term_memory_increment_size)

    # compute q, k, v for all windows
    xs_pre_attn = self.melodi_layer.pre_attention(xs)
    us_pre_attn = self.melodi_layer.pre_attention(us)

    (keys_x, values_x, queries_x) = self.melodi_layer.kvq(xs_pre_attn)
    (keys_u, values_u, queries_u) = self.melodi_layer.kvq(us_pre_attn)
    logging.info(
        "melodi-seq-layer: pre_attn xs = %s, us = %s",
        vshape(xs_pre_attn),
        vshape(us),
    )
    logging.info(
        "melodi-seq-layer: keys_x = %s, values_x = %s, queries_x = %s",
        vshape(keys_x),
        vshape(values_x),
        vshape(queries_x),
    )
    logging.info(
        "melodi-seq-layer: keys_u = %s, values_u = %s, queries_u = %s",
        vshape(keys_u),
        vshape(values_u),
        vshape(queries_u),
    )

    # split xs, us into windows
    xs_list = jnp.split(xs, number_of_windows, axis=1)
    us_list = jnp.split(us, number_of_windows, axis=1)

    # split kvq for xs, us into windows
    keys_x_list = jnp.split(keys_x, number_of_windows, axis=1)
    values_x_list = jnp.split(values_x, number_of_windows, axis=1)
    queries_x_list = jnp.split(queries_x, number_of_windows, axis=1)
    logging.info("melodi-seq-layer: keys_xs len = %s x %s",
                 len(keys_x_list),
                 vshape(keys_x_list[1]))

    keys_u_list = jnp.split(keys_u, number_of_windows, axis=1)
    values_u_list = jnp.split(values_u, number_of_windows, axis=1)
    queries_u_list = jnp.split(queries_u, number_of_windows, axis=1)
    logging.info("melodi-seq-layer: keys_us len = %s x %s",
                 len(keys_u_list),
                 vshape(keys_u_list[1]))

    # loop over windows
    xs_out_list = []
    us_out_list = []
    for k in range(number_of_windows):
      xs_k, us_k, short_ms, long_increment = self.melodi_layer.forward_window(
          xs_list[k],
          us_list[k],
          (short_ms, (long_k, long_v)),
          (keys_x_list[k], values_x_list[k], queries_x_list[k]),
          (keys_u_list[k], values_u_list[k], queries_u_list[k]),
          self._get_attn_dropout_mask
      )

      if long_ms is not None and long_increment is not None:
        logging.info("melodi-seq-layer: enqueue long increment = %s",
                     vshape(long_increment))

        inc_pre_attn = self.melodi_layer.pre_attention(long_increment)
        (inc_k, inc_v, _) = self.melodi_layer.kvq(inc_pre_attn)

        long_keep = long_ms[:, self.long_term_memory_increment_size:, :]
        long_ms = jnp.concatenate([long_keep, long_increment], axis=1)

        long_k_keep = long_k[:, self.long_term_memory_increment_size:, :, :]
        long_k = jnp.concatenate([long_k_keep, inc_k], axis=1)

        long_v_keep = long_v[:, self.long_term_memory_increment_size:, :, :]
        long_v = jnp.concatenate([long_v_keep, inc_v], axis=1)

      # add output of xs and zs to lists
      xs_out_list.append(xs_k)
      us_out_list.append(us_k)

    # store ms
    if self.short_memory is not None:
      logging.info("melodi-seq-layer: store short ms = %s", vshape(short_ms))
      self.short_memory.store_next_cache(short_ms)

    if self.long_memory is not None:
      logging.info("melodi-seq-layer: store long ms = %s", vshape(long_ms))
      self.long_memory.store_next_cache(long_ms)

    # concat xs_out, us_out
    xs_out = jnp.concatenate(xs_out_list, axis=1)
    us_out = jnp.concatenate(us_out_list, axis=1)

    logging.info("melodi-seq-layer: xs_out = %s", vshape(xs_out))
    logging.info("melodi-seq-layer: us_out = %s", vshape(us_out))

    return xs_out, us_out


@gin.configurable
class MelodiStack(nn.Module):
  """A stack of Melodi layers.

     Different with DecoderStack, it adds initialized summary embeddings.
     see 'us' argument in __call__ function.
  """

  # Mode: e.g test or train.  Set by caller.
  mode: str

  # Configured by either the caller or by gin.
  embedding_dim: int = gin.REQUIRED
  batch_size: int = gin.REQUIRED

  # Usually Configured by gin.
  num_layers: int = gin.REQUIRED
  layer_factory: Any = gin.REQUIRED
  window_length: int = gin.REQUIRED

  # Memory
  long_term_memory_layer_indices: Sequence[int] = ()
  no_memory_layer_indices: Sequence[int] = ()  # default ()
  long_term_memory_size: int = gin.REQUIRED
  long_term_memory_increment_size: int = gin.REQUIRED
  short_term_memory_size: int = gin.REQUIRED
  memory_embedding_dim: int = gin.REQUIRED  # 0: use embedding_dim
  summary_size: int = gin.REQUIRED

  # Optional aux output from middle of stack.  -1 means no output.
  aux_output_layer_num: int = -1

  # Usually passed by caller.
  dtype: Any = jnp.float32

  def is_training(self):
    return self.mode == "train"

  def get_memory_type(self, layer_index: int):
    if layer_index in self.long_term_memory_layer_indices:
      return "long"
    elif layer_index in self.no_memory_layer_indices:
      return None
    else:
      return "short"

  def get_memory_size(self, layer_index: int):
    memory_type = self.get_memory_type(layer_index)
    if memory_type == "long":
      return (self.long_term_memory_size, self.short_term_memory_size)
    elif memory_type == "short":
      return (0, self.short_term_memory_size)
    else:
      return (0, 0)

  def setup(self):
    # A stack of transformer layers.
    layer_factory = self.layer_factory

    # Layer factor is instantiated num_layers times.
    layers = []
    for i in range(0, self.num_layers):
      memory_type = self.get_memory_type(i)
      (long_mem_size, short_mem_size) = self.get_memory_size(i)
      logging.info("MelodiStack: layer %d, mem-type=%s, long=%d, short=%d",
                   i, memory_type, long_mem_size, short_mem_size)
      long_mem_increment_size = (
          self.long_term_memory_increment_size if memory_type == "long" else 0
      )

      layer = layer_factory(
          mode=self.mode,
          batch_size=self.batch_size,
          embedding_dim=self.embedding_dim,
          memory_type=memory_type,
          long_term_memory_size=long_mem_size,
          long_term_memory_increment_size=long_mem_increment_size,
          short_term_memory_size=short_mem_size,
          memory_embedding_dim=self.memory_embedding_dim,
          summary_size=self.summary_size,
          dtype=self.dtype,
      )
      layers.append(layer)
    self.melodi_layers = layers

    # Final layernorm.
    self.final_layernorm = nn_components.LayerNorm()

    # Summary embedding
    # Using stddev=0.25 is good enough to provide good results.
    #    (see https://arxiv.org/pdf/2410.03156)
    # Ablation on stddev is worth investigating.
    logging.info("MelodiStack: summary_size = %s", self.summary_size)
    self.summary_embeddings = self.param(
        "summary_embeddings",
        jax.nn.initializers.truncated_normal(stddev=0.25),
        (self.summary_size, self.embedding_dim),
        self.dtype)

  def __call__(self,
               xs: jax.Array,
               start_of_sequence: jax.Array
              ) -> Tuple[jax.Array, Optional[jax.Array], Metrics]:
    """Apply a stack of melodi layers to xs.

    Note that the basic TransformerStack does not return any metrics, but the
    interface is compatible with other modules that potentially do.

    Args:
      xs: Array of input embeddings of shape (batch_size, seq_len, embed_dim).
      start_of_sequence: Boolean values of shape (batch_size,), which
          indicate whether the input is the start of a new sequence.
          Used for long-context Transformer-XL models.

    Returns:
      (ys: Array of output embeddings of shape (batch_size, seq_len, embed_dim),
       ys_aux: Optional Array of embeddings from an intermediate layer, used
               when extracting auxiliary signals
       d_metrics: Dictionary of additional metrics.
      )
    """
    xs = jnp.asarray(xs, dtype=self.dtype)
    logging.info("MelodiStack: xs = %s", vshape(xs))

    # Expand summary embeddings
    #   from: (summary_size, embedding_dim)
    #     to: (bs, summary_size * num_windows, emb_dim)
    logging.info("MelodiStack: summary_embeddings = %s",
                 vshape(self.summary_embeddings))
    us = jnp.expand_dims(self.summary_embeddings, axis=0)
    bs, seq_size, _ = xs.shape
    us = jnp.tile(us, (bs, seq_size//self.window_length, 1))
    logging.info("MelodiStack: us = %s", vshape(us))

    # Stack of transformer layers.
    ys = xs
    for (i, layer) in enumerate(self.melodi_layers):
      logging.info("MelodiStack: ---- Layer %d ----", i)
      (ys, us) = layer(ys, us, start_of_sequence=start_of_sequence)

    # Apply layernorm before token prediction.
    ys = self.final_layernorm(ys)

    aux_ys = None
    d_metrics = {}
    return (ys, aux_ys, d_metrics)
