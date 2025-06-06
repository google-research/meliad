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

"""A single transformer layer."""

import functools
from typing import Any, Callable, Dict, Literal, Mapping, NewType, Optional, Sequence, Tuple

from absl import logging
from flax import linen
import gin

import jax
import jax.numpy as jnp

from transformer import attention
from transformer import kv_cache
from transformer import nn_components
from transformer import position
from transformer import position_alibi
from transformer import position_fourier
from transformer import position_nn
from transformer import position_t5


Array = jax.Array
DecoderState = NewType("DecoderState", Mapping[str, Array])
KVITuple = attention.KVITuple
KVQITuple = attention.KVQITuple
KVITupleFlaxVars = Tuple[Any, Any, Any]   # Tuple of flax Variables.
ArraySeq = Sequence[Array]
OptArray = Optional[Array]

vshape = nn_components.vshape


@gin.configurable
class TransformerLayer(linen.Module):
  """Full transformer layer, with attention."""

  # Set by parent (TransformerStack)
  mode: str
  batch_size: int
  embedding_size: int

  # Configurable hyper-parameters
  transformer_base_factory: Any = gin.REQUIRED

  num_heads: int = gin.REQUIRED
  head_size: int = gin.REQUIRED
  mlp_dim: int = gin.REQUIRED

  window_length: int = gin.REQUIRED
  use_long_xl_architecture: bool = True
  relative_position_type: Literal[
      None, "fourier", "t5", "nn", "rotary", "orthogonal", "alibi"
  ] = "t5"
  use_causal_mask: bool = True
  attn_dropout_rate: float = 0.0

  use_self_attention: bool = True              # Perform self-attention.
  use_cross_attention: bool = False            # Perform cross-attention.
  use_importance: bool = False                 # Takes importance as an input
  parallel_windowed_attention: bool = True

  # If true, use relative positions between decoder (xs) and encoder (cross_xs).
  cross_attention_aligned_positions: bool = True

  # Usually set by parent.
  dtype: Any = gin.REQUIRED

  def supports_generate(self) -> bool:
    return self.use_long_xl_architecture

  def setup(self):
    # Basic transformer functionality: everything except attention.
    self.tbase = self.transformer_base_factory(
        mode=self.mode,
        embedding_size=self.embedding_size,
        num_heads=self.num_heads,
        head_size=self.head_size,
        mlp_dim=self.mlp_dim,
        use_cross_attention=self.use_cross_attention,
        cross_attention_dedicated_kv=True,
        num_position_embeddings=0,
        dtype=self.dtype)

    # Set up relative position encoding.
    self.relative_positions = get_relative_positions(
        position_type=self.relative_position_type,
        num_heads=self.num_heads,
        window_length=self.window_length,
        mode=self.mode,
        dtype=self.dtype
    )

    # Set up cache for Transformer-XL style architectures.
    # A separate cache is created for each each mode (e.g. train, test)
    self.kvi_cache = kv_cache.KVICache(
        mode=self.mode,
        enable_cache=self.use_long_xl_architecture,
        batch_size=self.batch_size,
        window_length=self.window_length,
        num_heads=self.num_heads,
        head_size=self.head_size,
        use_importance=self.use_importance,
        dtype=self.dtype
        )

  def _get_rel_position_bias(self,
                             num_queries: int,
                             num_keys: int) -> Optional[Array]:
    """Returns the relative position bias, if any."""
    # The bias doesn't depend on the query content, and so can be precomputed.
    bidirectional = not self.use_causal_mask
    if self.relative_positions is not None:
      rel_position_bias = self.relative_positions(num_queries, num_keys,
                                                  bidirectional=bidirectional)
      logging.info("tlayer: %s relative bias = %s",
                   self.relative_position_type, vshape(rel_position_bias))
      logging.info("tlayer: relative bias -- bidirectional = %s",
                   bidirectional)
    else:
      logging.info("tlayer: no relative position bias.")
      rel_position_bias = None
    return rel_position_bias

  def _get_causal_mask(self,
                       num_queries: int,
                       num_keys: int) -> Optional[Array]:
    """Returns the causal mask, if any."""
    # Get causal mask.
    if self.use_causal_mask:
      causal_mask = position.causal_mask(num_queries, num_keys,
                                         window_length=self.window_length)
      logging.info("tlayer: causal mask = %s", vshape(causal_mask))
    else:
      logging.info("tlayer: no causal mask.")
      causal_mask = None
    return causal_mask

  def _get_attn_dropout_mask(self,
                             num_queries: int,
                             num_keys: int):
    """Returns a mask that applies dropout to the attention matrix."""
    # The mask is supplied as floating-point values, not boolean.
    # The mask will be broadcast across batches and windows.
    is_training = (self.mode == "train")
    if self.attn_dropout_rate > 0.0 and is_training:
      dropout_rng = self.make_rng("dropout")
      attn_shape = (self.num_heads, num_queries, num_keys)
      dropout_multiplier = nn_components.dropout_multiplier_mask(
          dropout_rng, self.attn_dropout_rate, attn_shape, self.dtype)
      logging.info("tlayer: attn_dropout = %s", vshape(dropout_multiplier))
    else:
      dropout_multiplier = None
    return dropout_multiplier

  def _get_number_of_windows(self, sequence_length: int, prev_kvi: Any) -> int:
    """Returns the number of windows or blocks in the sequence."""
    if sequence_length < self.window_length:
      num_windows = 1  # Shouldn't happen, but it's not an error.
    elif sequence_length == self.window_length:
      num_windows = 1
      if self.use_long_xl_architecture:
        assert prev_kvi is not None
    else:
      if self.use_long_xl_architecture:
        logging.info("tlayer: Using sliding window with Transformer XL.")
        assert prev_kvi is not None
      else:
        logging.info("tlayer: Using sliding window without Transformer XL.")
        assert prev_kvi is None
      num_windows = sequence_length // self.window_length
      if (num_windows * self.window_length) != sequence_length:
        raise ValueError(f"Window length {self.window_length} must be a " +
                         f"multiple of sequence length {sequence_length}")
    logging.info("tlayer: num_windows = %d.", num_windows)
    return num_windows

  def single_window_attention(self,
                              kvqi_w: KVQITuple,
                              *,
                              rel_position_bias: OptArray,
                              causal_mask: OptArray,
                              kq_relative_offset: int,
                              dropout_multiplier: OptArray,
                              attention_scale_factor: OptArray) -> Array:
    """Does attention within a single window."""
    (keys_w, values_w, queries_w, _, importance_w) = kvqi_w

    # If using RoPE, keys and queries are rotated before self-attention.
    if self.relative_position_type == "rotary":
      logging.info("tlayer: Rotary position encodings (RoPE), offset = %d",
                   kq_relative_offset)
      (keys_w, queries_w) = position.rotate_kq(keys_w, queries_w,
                                               max_wavelength=10_000,
                                               offset=kq_relative_offset)
    # Self-attention over input tokens.
    logging.info("tlayer: single window attention.")
    attn_ys_w = attention.simple_attention(
        keys_w, values_w, queries_w, importance_w,
        relative_position_bias=rel_position_bias,
        scale_factor=attention_scale_factor,
        causal_mask=causal_mask,
        dropout_multiplier=dropout_multiplier,
        dtype=self.dtype)
    return attn_ys_w

  def windowed_attention(self,
                         single_window_attn_fn: Callable[[KVQITuple], Array],
                         kvqi: KVQITuple,
                         prev_kvi: Optional[KVITuple],
                         *,
                         start_of_sequence: Array,
                         attention_scale_factors: Dict[str, OptArray],
                         num_windows: int) -> Tuple[Array, Optional[KVITuple]]:
    """Peform sliding window attention."""
    # Unused in default implementation
    del start_of_sequence
    del attention_scale_factors  # folded into single_window_attn_fn

    logging.info("tlayer: windowed attention.")
    if self.parallel_windowed_attention or num_windows == 1:
      (attn_ys, next_kvi) = attention.parallel_sliding_window_attention(
          single_window_attn_fn,
          kvqi,
          prev_kvi,
          num_windows)
    else:
      (attn_ys, next_kvi) = attention.sequential_sliding_window_attention(
          single_window_attn_fn,
          kvqi,
          prev_kvi,
          num_windows)
    return (attn_ys, next_kvi)

  def self_attention(self,
                     xs: Array,
                     prev_kvi: Optional[KVITuple],
                     *,
                     start_of_sequence: Array,
                     importance: OptArray) -> Tuple[Array, Optional[KVITuple]]:
    """Perform sliding-window self-attention over the sequence xs.

    Note that layer-norm, dropout etc. should have already been applied to xs.

    Args:
      xs: An array of shape (batch_size, seq_length, embed_dim),
          which will be used to construct the queries, keys, and values.
      prev_kvi: Cached keys and values for the last block of xs from the
          previous step, when using Transformer-XL.
      start_of_sequence: Boolean array of shape (batch_size,) which is true
          if the current element is starting a new sequence.
      importance: Optional array of shape (batch_size, seq_length,) which
          contains the importance of token in the sequence.

    Returns:
      (attn_ys: of shape (batch_size, seq_length, num_heads, head_dim)
       next_kvi: Keys and values from the last of window of xs, for caching.)
    """
    (_, sequence_length, _) = xs.shape

    # Compute keys, values, and queries.
    logging.info("tlayer: self-attention -- compute keys,values,queries.")
    (keys, values, queries) = self.tbase.kvq(xs)

    # Compute masks and position info for the sliding window.
    (num_queries, num_keys) = attention.sliding_attention_window_shape(
        (keys, values, importance), prev_kvi, queries,
        window_length=self.window_length)
    num_windows = self._get_number_of_windows(sequence_length, prev_kvi)
    rel_pos_bias = self._get_rel_position_bias(num_queries, num_keys)
    causal_mask = self._get_causal_mask(num_queries, num_keys)
    dropout_mul = self._get_attn_dropout_mask(num_queries, num_keys)

    # Do sliding window self-attention.
    single_window_attn_fn = functools.partial(
        self.single_window_attention,
        rel_position_bias=rel_pos_bias,
        causal_mask=causal_mask,
        kq_relative_offset=num_keys - num_queries,
        dropout_multiplier=dropout_mul,
        attention_scale_factor=self.tbase.self_attention_scale_factor())

    kvqi = (keys, values, queries, None, importance)
    (attn_ys, next_kvi) = self.windowed_attention(
        single_window_attn_fn,
        kvqi,
        prev_kvi,
        start_of_sequence=start_of_sequence,
        attention_scale_factors=self.tbase.attention_scale_factors(),
        num_windows=num_windows)

    return (attn_ys, next_kvi)

  def aligned_cross_attention(self,
                              xs: Array,
                              cross_xs: Array,
                              prev_cross_kvi: Optional[KVITuple],
                             ) -> Tuple[Array, Optional[KVITuple]]:
    """Perform positionally aligned cross-attention from xs to cross_xs.

    Aligned cross-attention is a variant of cross-attention in which the
    tokens from the encoder (cross_xs) and tokens from the decoder (xs) are
    part of the same sequence, and thus have positions relative to each other.
    The current implementation assumes an exact positional alignment between
    xs and cross_xs, which happens when a later layer attends to an earlier
    layer using skip connections.

    Aligned cross-attention works just like sliding window self-attention,
    except that the elements of xs attends to cross_xs, rather than to itself.
    Because the two sequences are perfectly aligned, the implementation uses
    relative positions and sliding windows in exactly the same way as
    self-attention.

    Note that layer-norm, dropout etc. should have already been applied to xs
    and cross_xs.

    Args:
      xs: An array of shape (batch_size, seq_length, embed_dim),
          which will be used to construct the queries.
      cross_xs: An array of shape (batch_size, encoder_seq_length, embed_dim)
          which will be used to construct the keys and values.
      prev_cross_kvi: Cached keys and values for the last block of cross_xs
          from the previous step, when using Transformer-XL.

    Returns:
      (cross_attn_ys: of shape (batch_size, seq_length, num_heads, head_dim)
       next_kvi: Keys and values from the last window of cross_xs.)
    """
    (_, sequence_length, _) = xs.shape
    # TODO(delesley): enable importance for cross-attention.
    importance = None

    # TODO(delesley): Add support for cases in which the two sequences are
    # not perfectly aligned.

    # TODO(delesley): Consider moving this method to a derived class, since
    # aligned cross-attention is not exactly a standard operation.

    # Compute keys, values, and queries for cross-attention.
    logging.info("tlayer: cross-attention; compute keys,values,queries.")
    (keys, values, queries) = self.tbase.cross_kvq(xs, cross_xs)

    # Compute masks and position info for the sliding window.
    (num_queries, num_keys) = attention.sliding_attention_window_shape(
        (keys, values, importance), prev_cross_kvi, queries,
        window_length=self.window_length)
    num_windows = self._get_number_of_windows(sequence_length,
                                              prev_cross_kvi)
    rel_pos_bias = self._get_rel_position_bias(num_queries, num_keys)
    causal_mask = self._get_causal_mask(num_queries, num_keys)
    dropout_mul = self._get_attn_dropout_mask(num_queries, num_keys)

    # Do sliding window cross-attention.
    single_window_attn_fn = functools.partial(
        self.single_window_attention,
        rel_position_bias=rel_pos_bias,
        causal_mask=causal_mask,
        kq_relative_offset=num_keys - num_queries,
        dropout_multiplier=dropout_mul,
        attention_scale_factor=self.tbase.cross_attention_scale_factor())

    kvqi = (keys, values, queries, None, importance)
    (attn_ys, next_kvi) = attention.parallel_sliding_window_attention(
        single_window_attn_fn,
        kvqi,
        prev_cross_kvi,
        num_windows)

    return (attn_ys, next_kvi)

  def __call__(self,
               xs: Array,
               start_of_sequence: Array,
               *,
               cross_xs: Optional[Array] = None,
               importance: Optional[Array] = None) -> Array:
    """Computes attention over a sequence of inputs.

    Args:
      xs: input sequence of shape (batch_size, sequence_length, num_hidden)
      start_of_sequence: An input array of shape (batch_size)

      --- The following must be passed by keyword only. ---
      cross_xs: Additional inputs for cross-attention from the encoder (if any).
          The embedding vectors in xs will cross-attend to those in cross_xs.
      importance: Array of shape (batch_size, sequence_length).
                  An importance bias for attention.

    Returns:
      ys: outputs of shape (batch_size, sequence_length, num_hidden)
    """

    xs = jnp.asarray(xs, dtype=self.dtype)
    logging.info("tlayer: xs = %s", vshape(xs))
    logging.info("tlayer: use_importance = %r", self.use_importance)
    if importance is not None:
      logging.info("tlayer: importance = %r", vshape(importance))

    logging.info("tlayer: pre-attention (layernorm, dropout, etc.)")
    xs_pre_attn = self.tbase.pre_attention(xs)

    # ==== Self attention. ====
    attn_ys = None
    if self.use_self_attention:
      prev_kvi = self.kvi_cache.load_prev_cache(start_of_sequence)

      (attn_ys, next_kvi) = self.self_attention(
          xs_pre_attn,
          prev_kvi,
          start_of_sequence=start_of_sequence,
          importance=importance,
      )

      self.kvi_cache.store_next_cache(next_kvi)

    # ==== Cross attention. ====
    cross_attn_ys = None
    if self.use_cross_attention:
      # TODO(delesley): Implement normal cross-attention as well.
      assert self.cross_attention_aligned_positions

      cross_xs_pre_attn = self.tbase.pre_cross_attention(cross_xs)
      del cross_xs

      # TODO(delesley): Enable multiple KV caches.
      # There's only one KV-cache, so it can't be used for both self-attention
      # and cross-attention at the same time.  The cache needs to be factored
      # into a separate class.
      assert not self.use_self_attention
      prev_kvi = self.kvi_cache.load_prev_cache(start_of_sequence)

      (cross_attn_ys, next_kvi) = self.aligned_cross_attention(
          xs_pre_attn,
          cross_xs_pre_attn,
          prev_kvi)

      self.kvi_cache.store_next_cache(next_kvi)

    # Post-attention MLP, resnet, and FFN.
    logging.info("tlayer: final FFN.")
    ys = self.tbase.post_attn_ffn(xs, attn_ys, cross_attn_ys)
    return ys

  def decode_token(self,
                   xs: Array,
                   decoder_state: DecoderState,
                   *,
                   importance: OptArray = None) -> Tuple[Array, DecoderState]:
    """Implements inference for a single token."""

    # When decoding, prior keys,values are loaded from the decoder state.
    # Other values are precomputed, and loaded from the decoder state.
    # The decoder state will be updated with the current token.
    xs = jnp.asarray(xs, dtype=self.dtype)
    logging.info("tlayer: xs = %s", vshape(xs))
    logging.info("tlayer: use_importance = %r", self.use_importance)
    if importance is not None:
      logging.info("tlayer: importance = %r", vshape(importance))

    # Compute keys, values and queries for current input token(s).
    logging.info("tlayer: compute keys,values,queries.")
    (keys, values, queries, _) = self.tbase.kvq(xs)

    # Load a full window of prior keys/values from the decoder_state,
    # and update decoder_state by writing the current key,value to the state.
    logging.info("tlayer: using autoregressive decoder.")
    (decoder_state, keys, values) = self._next_decoder_state(
        decoder_state, keys, values)

    # Each query attends to window_length prior keys.
    assert queries.shape[1] == 1
    assert keys.shape[1] == self.window_length

    kvqi = (keys, values, queries, None, importance)
    attn_ys = self.single_window_attention(
        kvqi,
        rel_position_bias=decoder_state["relative_position_bias"],
        causal_mask=None,
        kq_relative_offset=self.window_length,
        dropout_multiplier=None,
        attention_scale_factor=self.tbase.self_attention_scale_factor())

    # Post-attention MLP, residual connection, and FFN.
    logging.info("tlayer: final FFN.")
    ys = self.tbase.post_attn_ffn(xs, attn_ys, None)
    return (ys, decoder_state)

  def init_decoder_state(self,
                         sequence_length: int,
                         start_of_sequence: Array) -> DecoderState:
    """Initialize decoder state for autoregressive generation.

    Args:
      sequence_length: The maximum length of the sequence to generate.
      start_of_sequence: Array of boolean of shape (batch_size,)
                         True if starting a new sequence (with no prefix).

    Returns:
      A state object that can be passed to __call__.
    """

    # Note that generate always uses a local context of size window_length.
    # Training should be set up appropriately.
    if not self.use_long_xl_architecture:
      raise ValueError("Generation is only supported for transformer XL.")
    if not self.use_causal_mask:
      raise ValueError("Generator must have been trained with a causal mask.")

    assert self.use_causal_mask

    # Get relative position bias.
    if self.relative_positions is not None:
      # Relative positions for all tokens *prior* to the current token.
      # The causal mask prevents each token from attending to itself.
      rel_position_bias = self.relative_positions(1, self.window_length,
                                                  offset=self.window_length,
                                                  bidirectional=False)
    else:
      rel_position_bias = None

    # Initialize autoregressive storage for (key, value) pairs.
    # Include space for a prefix of window_length tokens.
    num_keys = sequence_length + self.window_length
    stored_shape = (self.batch_size, num_keys, self.num_heads, self.head_size)
    stored_keys = jnp.zeros(stored_shape, dtype=self.dtype)
    stored_values = jnp.zeros(stored_shape, dtype=self.dtype)
    start_index = self.window_length

    # Copy keys,values from cache into storage, for use as a prefix.
    prev_kvi = self.kvi_cache.load_prev_cache(start_of_sequence)

    if prev_kvi is not None:
      (pkeys, pvals, prev_imps) = prev_kvi
      assert prev_imps is None  # Not yet supported.
      assert pkeys.ndim == 4
      assert pkeys.shape[1] == self.window_length  # (b, wlen, num_heads, d)

      stored_keys = jax.lax.dynamic_update_slice_in_dim(
          stored_keys, pkeys, 0, axis=1)
      stored_values = jax.lax.dynamic_update_slice_in_dim(
          stored_values, pvals, 0, axis=1)

    decoder_state_dict = {
        "keys": stored_keys,
        "values": stored_values,
        "current_index": start_index,
        "relative_position_bias": rel_position_bias,
    }
    return DecoderState(decoder_state_dict)

  def _next_decoder_state(self,
                          decoder_state: DecoderState,
                          keys: Array,
                          values: Array) -> Tuple[DecoderState, Array, Array]:
    """Compute the next decoder state, and return keys,values to attend to.

    The keys,values returned from this function are drawn from the prior
    decoding state, and comprise a full window of local context.

    Args:
      decoder_state: The current decoder state, initially created using
          init_decoder_state().
      keys: The key for the current token, of shape (batch_size, 1, dim)
      values: The value for the current token of shape (batch_size, 1, dim)

    Returns:
      (next_decoder_state,
       window of keys of shape (batch_size, window_length, dim),
       window of values of shape (batch_size, window_length, dim))
    """

    assert keys.shape[1] == 1   # single-token autoregressive decoding.

    logging.info("attn_layer: next decoder state; key = %s", vshape(keys))

    # Unpack decoder_state
    stored_keys = decoder_state["keys"]
    stored_values = decoder_state["values"]
    curr_index = decoder_state["current_index"]

    # Slice to get window_length-sized chunk of previous keys,values.
    out_decoder_state = {}
    curr_win_index = curr_index - self.window_length
    out_keys = jax.lax.dynamic_slice_in_dim(
        stored_keys, curr_win_index, self.window_length, axis=1)
    out_values = jax.lax.dynamic_slice_in_dim(
        stored_values, curr_win_index, self.window_length, axis=1)

    # Write current keys,values to stored keys, values.
    stored_keys = jax.lax.dynamic_update_slice_in_dim(
        stored_keys, keys, curr_index, axis=1)
    stored_values = jax.lax.dynamic_update_slice_in_dim(
        stored_values, values, curr_index, axis=1)
    curr_index = curr_index + 1

    # Pack a new decoder_state object.
    out_decoder_state["keys"] = stored_keys
    out_decoder_state["values"] = stored_values
    out_decoder_state["current_index"] = curr_index
    out_decoder_state["relative_position_bias"] = (
        decoder_state["relative_position_bias"])

    return (DecoderState(out_decoder_state), out_keys, out_values)


def get_relative_positions(
    position_type: Literal[
        None, "fourier", "t5", "nn", "rotary", "orthogonal", "alibi"
    ],
    num_heads: int,
    window_length: int,
    mode: str,
    dtype: Any = jnp.float32):
  """Return a relative position bias layer for the given named type.

  Args:
    position_type: The type of relative position encoding to use.
    num_heads: The number of attention heads.
    window_length: The window length of the model.
    mode: The mode of the model.
    dtype: The data type of the model.
  Returns:
    relative_positions: The relative position bias layer.
  """

  if position_type == "fourier":
    relative_positions = position_fourier.RelativeFourierPositions(
        num_heads=num_heads,
        max_number_of_keys=window_length,
        dtype=dtype)
  elif position_type == "t5":
    relative_positions = position_t5.T5RelativePositionBiases(
        num_buckets=32,   # TODO(delesley): Let Gin configure these.
        max_distance=128,
        num_heads=num_heads,
        dtype=dtype)
  elif position_type == "nn":
    relative_positions = position_nn.NNRelativePositionBiases(
        num_heads=num_heads,
        dtype=dtype,
    )
  elif position_type == "orthogonal":
    relative_positions = position_nn.OrthogonalBasisPositionBias(
        mode=mode,
        num_heads=num_heads,
        dtype=dtype,
    )
  elif position_type == "alibi":
    relative_positions = position_alibi.BoundedALiBiIntegerPositions(
        num_heads=num_heads,
    )
  elif position_type == "rotary":
    # Rotary position encodings (RoPE).  No learned bias parameters.
    relative_positions = None
  else:
    assert position_type is None
    relative_positions = None

  return relative_positions

