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

from typing import Any, Literal, Mapping, NewType, Optional, Sequence, Tuple

from absl import logging
from flax import linen as nn
import gin

import jax
import jax.numpy as jnp

from transformer import attention
from transformer import nn_components
from transformer import position
from transformer import position_alibi
from transformer import position_fourier
from transformer import position_nn
from transformer import position_t5
from transformer.memory import memory_factory
from transformer.memory import transformer_base


Array = jnp.ndarray
DecoderState = NewType("DecoderState", Mapping[str, Array])
WindowState = Optional[Tuple[Optional[attention.KVITuple], Optional[Array]]]
KVITuple = attention.KVITuple
CombineOption = Literal["ADD", "TRAINABLE_WEIGHTED_MEAN", "STOP_FORWARD"]

vshape = nn_components.vshape


@gin.configurable
class TransformerLayer(nn.Module):
  """Full transformer layer, with attention."""

  # Set by DecoderStack
  mode: str
  batch_size: int
  embedding_size: int
  cross_attention: bool = False
  recurrent_attention: bool = False
  memory: Optional[memory_factory.MemoryManager] = None

  # Configurable hyper-parameters
  num_heads: int = gin.REQUIRED
  head_size: int = gin.REQUIRED
  mlp_dim: int = gin.REQUIRED

  window_length: int = gin.REQUIRED
  use_long_xl_architecture: bool = True
  max_unrolled_windows: int = -1    # Always unroll.
  relative_position_type: Literal[
      None, "fourier", "t5", "nn", "rotary", "orthogonal", "alibi"
  ] = "fourier"
  use_causal_mask: bool = True
  attn_dropout_rate: float = 0.0

  recurrent_num_states: int = 0
  recurrent_gate_type: str = "bias"
  recurrent_single_gate: bool = False
  recurrent_skip_ffn: bool = False
  recurrent_always_clear_state: bool = False  # Always clear the rec. state.

  use_importance: bool = False       # Takes importance as an input
  compute_importance: bool = False   # Produces importance as an output
  output_embedding_size: Optional[int] = None  # Defaults to embedding_size
  memory_num_neighbors: int = 0
  memory_reset_on_new_doc: bool = True
  memory_combine_with_local: CombineOption = "TRAINABLE_WEIGHTED_MEAN"

  dtype: Any = jnp.float32

  # Modes which support caching of previous keys and values.
  supported_modes_for_cache: Sequence[str] = ("train", "test")
  update_memory_modes: Sequence[str] = ("train", "test")

  def supports_generate(self) -> bool:
    return self.use_long_xl_architecture

  def _get_cache_name_from_mode(self, mode: str) -> Tuple[str, bool, bool]:
    """Get the name of the cache, and whether to update the cache, from mode."""
    # This is a hack to ensure that "generate" steps generate text as a
    # continuation of the text that is stored in the "test" cache,
    # but it does not update the "test" cache.
    if mode == "generate":
      assert "test" in self.supported_modes_for_cache
      return ("test", False, False)   # Use test cache, but don't update it.
    elif mode == "init":
      return ("train", False, False)   # Use training cache for initialization.
    else:
      return (mode, True, mode in self.update_memory_modes)

  def _allocate_cached_kvi(self, mode: str) -> KVITuple:
    """Allocate (keys, values, importance) which can be cached between steps."""

    kv_shape = [self.batch_size, self.window_length,
                self.num_heads, self.head_size]
    imp_shape = [self.batch_size, self.window_length]

    def kv_initializer(shape):
      return jnp.zeros(shape, dtype=self.dtype)

    def imp_initializer(shape):
      return jnp.zeros(shape, dtype=self.dtype)

    pkeys = self.variable("state", "previous_keys_" + mode,
                          kv_initializer, kv_shape)
    pvals = self.variable("state", "previous_values_" + mode,
                          kv_initializer, kv_shape)
    if self.use_importance:
      pimportance = self.variable("state", "previous_importance_" + mode,
                                  imp_initializer, imp_shape)
    else:
      pimportance = None
    return (pkeys, pvals, pimportance)  # pytype: disable=bad-return-type  # jax-ndarray

  def _allocate_cached_recurrent_state(self, mode: str):
    rec_num_states = self.recurrent_num_states
    st_shape = [self.batch_size, rec_num_states, self.embedding_size]

    def st_initializer(shape):
      return jnp.zeros(shape, dtype=self.dtype)

    return self.variable("state", "recurrent_state_" + mode,
                         st_initializer, st_shape)

  def _get_output_embedding_size(self):
    if self.output_embedding_size is None:
      return self.embedding_size  # default
    else:
      return self.output_embedding_size

  def setup(self):
    # Basic transformer functionality: everything except attention.
    self.tbase = transformer_base.MemTransformerBase(
        mode=self.mode,
        embedding_size=self.embedding_size,
        output_embedding_size=self._get_output_embedding_size(),
        num_heads=self.num_heads,
        head_size=self.head_size,
        mlp_dim=self.mlp_dim,
        cross_attention_q=self.recurrent_attention or self.cross_attention,
        cross_attention_kv=False,         # or True to use separate k,v.
        num_position_embeddings=0,
        num_cross_position_embeddings=0,  # or self.recurrent_num_states w/ k,v.
        dtype=self.dtype)

    # Recurrent transformer functionality.
    self.recurrent_tbase = None
    if self.recurrent_attention:
      # Recurrent transformer layer.
      # We use a learned position embedding so that each element of the state
      # can learn to query and compute different summaries.
      self.recurrent_tbase = transformer_base.MemTransformerBase(
          mode="pure",  # Disable dropout, which breaks jax.lax.scan.
          embedding_size=self.embedding_size,
          output_embedding_size=self.embedding_size,
          num_heads=self.num_heads,
          head_size=self.head_size,
          mlp_dim=self.mlp_dim,
          cross_attention_q=True,
          cross_attention_kv=False,          # or True to use separate k,v.
          num_position_embeddings=self.recurrent_num_states,
          num_cross_position_embeddings=0,   # or self.window_length w/ k,v.
          gate_type=self.recurrent_gate_type,
          single_gate=self.recurrent_single_gate,
          skip_ffn=self.recurrent_skip_ffn,
          dtype=self.dtype)

      # Initial state at start of document.
      # We want this to be initially small, but large enough that adafactor
      # will scale updates to a reasonable value.
      self.recurrent_initial_state = self.param(
          "recurrent_initial_state",
          jax.nn.initializers.normal(stddev=0.1),
          (self.recurrent_num_states, self.embedding_size), jnp.float32)

      # Cached state from previous step for BPTT.
      rec_state = {}
      for mkey in self.supported_modes_for_cache:
        rec_state[mkey] = self._allocate_cached_recurrent_state(mkey)
      self.cached_recurrent_state = rec_state

    # Compute importance scores for the outputs.
    if self.compute_importance:
      imp_kernel_init = jax.nn.initializers.variance_scaling(
          scale=0.1, mode="fan_in",
          distribution="truncated_normal")
      self.importance_layer = nn.Dense(1, use_bias=True,
                                       kernel_init=imp_kernel_init,
                                       dtype=self.dtype)

    # Set up relative position encoding.
    if self.relative_position_type == "fourier":
      self.relative_positions = position_fourier.RelativeFourierPositions(
          num_heads=self.num_heads,
          max_number_of_keys=self.window_length,
          dtype=self.dtype)
    elif self.relative_position_type == "t5":
      self.relative_positions = position_t5.T5RelativePositionBiases(
          num_buckets=32,      # TODO(delesley): Let Gin configure these.
          max_distance=128,
          num_heads=self.num_heads,
          dtype=self.dtype)
    elif self.relative_position_type == "nn":
      self.relative_positions = position_nn.NNRelativePositionBiases(
          num_heads=self.num_heads,
          dtype=self.dtype,
      )
    elif self.relative_position_type == "orthogonal":
      self.relative_positions = position_nn.OrthogonalBasisPositionBias(
          mode=self.mode,
          num_heads=self.num_heads,
          dtype=self.dtype,
      )
    elif self.relative_position_type == "alibi":
      self.relative_positions = position_alibi.BoundedALiBiIntegerPositions(
          num_heads=self.num_heads,
      )
    elif self.relative_position_type == "rotary":
      # Rotary position encodings (RoPE).  No learned bias parameters.
      self.relative_positions = None
    else:
      assert self.relative_position_type is None
      self.relative_positions = None

    # Set up cache for Transformer-XL style architectures.
    # A separate cache is created for each each mode (e.g. train, test)
    cached_kvi = {}
    if self.use_long_xl_architecture:
      for mkey in self.supported_modes_for_cache:
        cached_kvi[mkey] = self._allocate_cached_kvi(mkey)
    self.cached_kvi = cached_kvi

    # Set up external memory.
    # A separate memory will be created for each mode (e.g. train, test)
    mem_layers = {}
    if self.memory is not None:
      self.memory_bias = self.param("external_memory_bias", nn.zeros,
                                    (self.num_heads,), "float32")
      for mkey in self.supported_modes_for_cache:
        mlayer = self.memory.create_memory_layer()
        # Use setattr to setup the name and module containership hierarchy.
        setattr(self, "mem_layer_" + mkey, mlayer)
        mem_layers[mkey] = mlayer
    self.mem_layers = mem_layers

    # Verify memory_combine_with_local is among allowed options:
    if self.memory_combine_with_local not in ("ADD", "TRAINABLE_WEIGHTED_MEAN",
                                              "STOP_FORWARD"):
      raise ValueError(
          f"Unrecognized setting: {self.memory_combine_with_local = }")

  def _get_cached_kvi(self, start_of_sequence: Array,
                      mode: str) -> Optional[KVITuple]:
    """Returns cached (keys, values, importance) from the previous step."""
    if not self.use_long_xl_architecture:
      return None
    if mode not in self.cached_kvi:
      # No cache, but we're using XL / sliding window, so return zeros.
      logging.info("tlayer: using zero as initial XL cache value.")
      kvi_shape = (self.batch_size, self.window_length,
                   self.num_heads, self.head_size)
      return attention.initial_kvi(kvi_shape, self.use_importance,
                                   dtype=self.dtype)

    # New documents start with zero_kv.
    # Continuing the same document will attend to previous keys/vals.
    logging.info("tlayer: window_length = %d", self.window_length)
    (pkeys, pvals, pimportance) = self.cached_kvi[mode]
    (zkeys, zvals, zimportance) = attention.initial_kvi(
        pkeys.value.shape, self.use_importance, dtype=self.dtype)  # pytype: disable=attribute-error  # jax-ndarray

    # Broadcast start_of_sequence over non-batch dims.
    b = self.batch_size
    start_of_sequence_kv = jnp.reshape(start_of_sequence, [b, 1, 1, 1])
    prev_keys = jnp.where(start_of_sequence_kv, zkeys, pkeys.value)  # pytype: disable=attribute-error  # jax-ndarray
    prev_vals = jnp.where(start_of_sequence_kv, zvals, pvals.value)  # pytype: disable=attribute-error  # jax-ndarray
    if self.use_importance:
      start_of_sequence_imp = jnp.reshape(start_of_sequence, [b, 1])
      prev_importance = jnp.where(start_of_sequence_imp, zimportance,
                                  pimportance.value)  # pytype: disable=attribute-error  # jax-ndarray
    else:
      prev_importance = None
    logging.debug("tlayer: start_of_sequence = %s", vshape(start_of_sequence))
    logging.info("tlayer: prev_keys[%s] = %s", mode, vshape(prev_keys))
    logging.info("tlayer: prev_importance[%s] = %s", mode,
                 vshape(prev_importance))
    return (prev_keys, prev_vals, prev_importance)

  def _set_cached_kvi(self, next_kvi: KVITuple, mode: str):
    """Caches the last (keys, values, importance) from the current step."""
    if not self.use_long_xl_architecture:
      return
    if mode not in self.cached_kvi:
      return

    (pkeys, pvals, pimportance) = self.cached_kvi[mode]
    (nkeys, nvals, nimportance) = next_kvi   # From last window
    if pkeys.value.shape != nkeys.shape:  # pytype: disable=attribute-error  # jax-ndarray
      raise ValueError(
          "Shape mismatch for keys on write to cache: "
          + f"{pkeys.value.shape} != {nkeys.shape}"  # pytype: disable=attribute-error  # jax-ndarray
          + "\nThis could indicate that TransformerTaskConfig.sequence_length"
          " is less than TranformerLayer.window_length, which is a"
          " configuration error."
      )
    if pvals.value.shape != nvals.shape:  # pytype: disable=attribute-error  # jax-ndarray
      raise ValueError(
          "Shape mismatch for values on write to cache: " +
          f"{pvals.value.shape} != {nvals.shape}")  # pytype: disable=attribute-error  # jax-ndarray

    logging.info("tlayer: next_keys[%s] = %s", mode, vshape(nkeys))
    pkeys.value = nkeys
    pvals.value = nvals
    if self.use_importance:
      assert pimportance is not None  # for ptype
      assert nimportance is not None  # for ptype
      logging.info("tlayer: next_importance[%s] = %s", mode,
                   vshape(nimportance))
      if pimportance.value.shape != nimportance.shape:
        raise ValueError(
            "Shape mismatch for importance on write to cache: " +
            f"{pimportance.value.shape} != {nimportance.shape}")
      pimportance.value = nimportance

  def _get_cached_recurrent_state(self, start_of_sequence: Array,
                                  mode: str) -> Optional[Array]:
    """Returns cached recurrent state from the previous step."""
    if not self.recurrent_attention:
      return None
    if mode not in self.cached_recurrent_state:
      return None

    b = self.batch_size
    rstate = self.cached_recurrent_state[mode].value
    istate = jnp.asarray(self.recurrent_initial_state, dtype=self.dtype)
    istate = istate[jnp.newaxis, :, :]   # Add batch dimension for broadcast.
    logging.info("tlayer: get_cached_recurrent_state %s, %s",
                 vshape(istate), vshape(rstate))

    if self.recurrent_always_clear_state:
      return jnp.broadcast_to(istate, rstate.shape)

    start_of_sequence_st = jnp.reshape(start_of_sequence, (b, 1, 1))
    return jnp.where(start_of_sequence_st, istate, rstate)

  def _set_cached_recurrent_state(self, next_state: Array, mode: str):
    """Store the next recurrent state in the cache."""
    if not self.recurrent_attention:
      return
    if mode not in self.cached_recurrent_state:
      return
    if self.recurrent_always_clear_state:
      return

    logging.info("tlayer: set_cached_recurrent_state %s", vshape(next_state))
    rstate = self.cached_recurrent_state[mode]
    rstate.value = next_state

  def _query_external_memory(self, keys: Array, values: Array, queries: Array,
                             start_of_sequence: Array,
                             mode: str, update_memory: bool):
    """Query and update external memory."""
    if self.memory is None:
      return None

    # Make sure we initialize (allocate) the external memories for all modes.
    # Per the flax lazy module initialization scheme, setup() will not be
    # invoked on a submodule until that module is actually used.
    if mode == "init":
      for (_, mlayer) in self.mem_layers.items():
        (_, _) = mlayer.topk_retrieval(queries, self.memory_num_neighbors)
        mode = "train"  # Pretend we're in training mode during initialization.

    if mode not in self.mem_layers:
      return None
    if self.memory_num_neighbors == 0:
      raise ValueError("Using memory, but num_neighbors == 0")

    # Grab the appropriate memory layer for the current mode.
    memory_layer = self.mem_layers[mode]

    # Clear the relevant memories at the start of each new document.
    if update_memory and self.memory_reset_on_new_doc:
      # The number of "datasets" is batch_dim * num_heads.
      # jnp.repeat will "broadcast" start_of_sequence over num_heads.
      # E.g. if start_of_sequence = [True, False] and 4 heads,
      # jnp.repeat will yield [T, T, T, T, F, F, F, F]
      memory_layer.reset(jnp.repeat(start_of_sequence, self.num_heads))

    # Query external memory, with queries.
    (rkeys, rvals) = memory_layer.topk_retrieval(queries,
                                                 self.memory_num_neighbors)
    logging.info("tlayer: query external memory (%s): rvals = %s",
                 mode, vshape(rvals))

    # Sanity check all dimensions are as expected.
    assert rkeys.ndim == 5   # (b, seq_len, num_heads, num_neigh, head_dim)
    assert rvals.ndim == 5
    assert rkeys.shape == rvals.shape
    assert rkeys.shape[0] == queries.shape[0]  # batch size
    assert rkeys.shape[1] == queries.shape[1]  # sequence length
    assert rkeys.shape[2] == self.num_heads
    assert rkeys.shape[3] == self.memory_num_neighbors
    assert rkeys.shape[4] == self.head_size

    # Update external memory, with (keys, values).
    if update_memory:
      memory_layer.update(keys, values)
    return (rkeys, rvals)

  def __call__(self, xs: Array, start_of_sequence: Array,
               *,
               global_info: dict[str, Any],
               importance: Optional[Array] = None,
               cross_attention_kv: Optional[Tuple[Array, Array]] = None,
               window_state: Optional[WindowState] = None,
               decoder_state: Optional[DecoderState] = None) -> (
                   Tuple[Array, Optional[Array], Optional[WindowState],
                         Optional[DecoderState], Any]):
    """Computes attention over a sequence of inputs.

    Args:
      xs: input sequence of shape (batch_size, sequence_length, num_hidden)
      start_of_sequence: An input array of shape (batch_size)

      --- The following must be passed by keyword only. ---
      global_info:  A dictionary of global information (e.g., current training
        step as "step").
      importance: Array of shape (batch_size, sequence_length).
                  An importance bias for attention.
      cross_attention_kv: Keys and values from encoder for cross-attention.
      window_state: State object which contains context from the prior
                    window when using a transformer-XL or sliding window.
                    Initially created with load_window_state().
      decoder_state: State object for autoregressive decoding, initially
                     created with from init_decoder_state().

    Returns:
      (ys: outputs of shape (batch_size, sequence_length, num_hidden),
       importance_score: importance score for the next layer,
       next_window_state: state to pass to the next window,
       next_decoder_state: next decoder state for autoregressive decoding,
       viz_dict: dictionary of visualizations
      )
    """

    xs = jnp.asarray(xs, dtype=self.dtype)
    logging.info("tlayer: xs = %s", vshape(xs))
    logging.info("tlayer: recurrent = %r", self.recurrent_attention)
    logging.info("tlayer: use_importance = %r", self.use_importance)
    logging.info("tlayer: compute_importance = %r", self.compute_importance)
    if cross_attention_kv is not None:
      logging.info("tlayer: cross-attention = %r", vshape(cross_attention_kv))
    if importance is not None:
      logging.info("tlayer: importance = %r", vshape(importance))

    is_training = (self.mode == "train")

    # Compute keys, values and queries.
    # ---------------------------------
    logging.info("tlayer: compute keys,values,queries.")
    (keys, values, queries, queries2) = self.tbase.kvq(xs)
    attention_scale_factors = self.tbase.attention_scale_factors()
    (_, sequence_length, num_heads, _) = queries.shape  # (b, k, h, d)

    # Get biases and masks that are shared across windows.
    # ----------------------------------------------------
    if decoder_state is not None:
      logging.info("tlayer: using autoregressive decoder.")
      # When decoding, prior keys,values are loaded from the decoder state.
      # Other values are precomputed, and loaded from the decoder state.
      # The decoder state will be updated with the current token.
      assert window_state is None

      prev_kvi = None
      recurrent_state = None   # Use precomputed recurrent_kvq.
      cross_attention_kv = None
      rel_position_bias = decoder_state["relative_position_bias"]
      causal_mask = None
      dropout_multiplier = None

      # Reuse cached recurrent keys,values for each token.
      cached_recurrent_kvq = decoder_state["recurrent_kvq"]
      if cached_recurrent_kvq is not None:
        assert cross_attention_kv is None
        cross_attention_kv = (cached_recurrent_kvq[0], cached_recurrent_kvq[1])
      del cached_recurrent_kvq

      # Get a full window of keys,values and update decoder state.
      (decoder_state, keys, values) = self._next_decoder_state(
          decoder_state, keys, values)

      # Each query attends to window_length prior keys.
      assert keys.shape[1] == self.window_length
      kq_relative_offset = self.window_length
    else:
      logging.info("tlayer: windowed attention.")
      # When training, attention is done using windows or chunks, and prior
      # context (e.g. keys,values from the previous window) is stored in the
      # window_state object.
      (prev_kvi, recurrent_state) = window_state  # pytype: disable=attribute-error

      # Get the size of the sliding window for pos bias, dropout, & causal mask.
      (num_queries, num_keys) = attention.sliding_attention_window_shape(
          (keys, values, importance), prev_kvi, queries,
          window_length=self.window_length)
      kq_relative_offset = num_keys - num_queries

      # Get the relative position bias.
      # The bias doesn't depend on the query content, and so can be precomputed.
      if self.relative_positions is not None:
        rel_position_bias = self.relative_positions(num_queries, num_keys,
                                                    bidirectional=False)
        logging.info("tlayer: %s relative bias = %s",
                     self.relative_position_type, vshape(rel_position_bias))
      else:
        rel_position_bias = None

      # Get causal mask.
      if self.use_causal_mask:
        causal_mask = position.causal_mask(num_queries, num_keys,
                                           window_length=self.window_length)
        logging.info("tlayer: causal mask = %s", vshape(causal_mask))
      else:
        causal_mask = None

      # Apply dropout to the attention matrix.
      # The mask will be broadcast across batches and windows.
      if self.attn_dropout_rate > 0.0 and is_training:
        dropout_rng = self.make_rng("dropout")
        attn_shape = (self.num_heads, num_queries, num_keys)
        dropout_multiplier = nn_components.dropout_multiplier_mask(
            dropout_rng, self.attn_dropout_rate, attn_shape, self.dtype)
        logging.info("tlayer: attn_dropout = %s", vshape(dropout_multiplier))
      else:
        dropout_multiplier = None

    # Load and store values into external memory, if memory is not None.
    # ------------------------------------------------------------------
    (mode, _, update_memory) = self._get_cache_name_from_mode(self.mode)
    external_kv = self._query_external_memory(
        keys, values, queries,
        start_of_sequence=start_of_sequence, mode=mode,
        update_memory=decoder_state is None and update_memory)

    if (self.memory is not None and
        self.memory_combine_with_local == "TRAINABLE_WEIGHTED_MEAN"):
      external_memory_bias = jnp.asarray(self.memory_bias, dtype=self.dtype)
      external_memory_bias = jnp.reshape(external_memory_bias,
                                         (1, 1, num_heads, 1))
      external_memory_bias = jax.nn.sigmoid(external_memory_bias)
    else:
      external_memory_bias = None

    # Compute the number of windows.
    # ------------------------------
    if sequence_length < self.window_length:
      num_windows = 1  # Happens with autoregressive decoding.
    elif sequence_length == self.window_length:
      num_windows = 1
      if self.use_long_xl_architecture:
        assert prev_kvi is not None
    else:
      if not self.use_long_xl_architecture:
        raise ValueError("Can only use sliding window with Transformer XL.")
      num_windows = sequence_length // self.window_length
      if (num_windows * self.window_length) != sequence_length:
        raise ValueError(f"Window length {self.window_length} must be a " +
                         f"multiple of sequence length {sequence_length}")
    logging.info("tlayer: num_windows = %d.", num_windows)

    # Define the function to do attention within a single window.
    # ---------------------------------------------------------
    def single_window_attention(carry, inputs_w):
      # This function uses the following variables from the outer scope.
      # They are listed here for clarity.
      nonlocal rel_position_bias
      nonlocal causal_mask
      nonlocal kq_relative_offset
      nonlocal dropout_multiplier
      nonlocal attention_scale_factors
      nonlocal external_memory_bias
      nonlocal cross_attention_kv  # externally supplied.

      # keys,values,queries over the whole sequence will be split into chunks.
      # xs_w, kvqi_w, etc. are the chunk for the current window.
      (prev_kvi_w, rec_state) = carry  # carried from one window to the next.
      (kvqi_w, external_kv_w) = inputs_w  # inputs to the current window.
      # (keys_curr_w, values_curr_w, _, _, importance_curr_w) = kvqi_w

      # Concatenate keys,values from the previous window with the current
      # window to implement sliding window attention.
      (kvqi_w, next_kvi_w) = attention.concat_kvqi(kvqi_w, prev_kvi_w)
      (keys_w, values_w, queries_w, queries2_w, importance_w) = kvqi_w

      # Perform recurrent attention within the current window to get the next
      # recurrent state, and set up cross attention.
      if rec_state is not None:
        logging.info("tlayer: recurrent attention.")

        # NOTE -- recurrent states and input tokens are handled separately,
        # because they have separate learned positional embeddings.  Due to
        # the way TransformerBase does cross-attention, this means that we use
        # separate key,value layers for rec_state and tokens_w.

        # Keys, values, queries from recurrent state.
        logging.info("tlayer: recurrent kvq.")
        assert self.recurrent_tbase is not None  # for pytype
        rec_kvq = self.recurrent_tbase.kvq(rec_state)
        r_scale_factors = self.recurrent_tbase.attention_scale_factors()
        (r_keys, r_values, r_queries, r_queries2) = rec_kvq

        # Joint attention over both recurrent states and input tokens.
        logging.info("tlayer: recurrent self-attention.")
        r_attn_ys = attention.simple_attention(
            r_keys, r_values, r_queries, None,
            scale_factor=r_scale_factors[0],
            dtype=self.dtype)

        logging.info("tlayer: recurrent cross-attention.")
        r_cross_attn_ys = attention.simple_attention(
            keys_w, values_w, r_queries2, importance_w,
            scale_factor=r_scale_factors[1],
            dtype=self.dtype)

        # Recurrent post-attention FFN.
        logging.info("tlayer: recurrent ffn.")
        assert self.recurrent_tbase is not None  # for pytype
        next_rec_state = self.recurrent_tbase.post_attn_ffn(
            rec_state, r_attn_ys, r_cross_attn_ys)

        # Get keys and values for cross-attention from recurrent state.
        assert cross_attention_kv is None
        local_cross_attention_kv = (r_keys, r_values)
      else:
        # Get keys and values for cross-attention from external argument.
        next_rec_state = None
        local_cross_attention_kv = cross_attention_kv

      # If using RoPE, keys and queries are rotated before self-attention.
      if self.relative_position_type == "rotary":
        logging.info("Using rotary position encodings (RoPE), offset = %d",
                     kq_relative_offset)
        (keys_w, queries_w) = position.rotate_kq(keys_w, queries_w,
                                                 max_wavelength=10_000,
                                                 offset=kq_relative_offset)

      # Self-attention over input tokens.
      logging.info("tlayer: self-attention.")
      attn_ys_w = attention.simple_attention(
          keys_w, values_w, queries_w, importance_w,
          relative_position_bias=rel_position_bias,
          scale_factor=attention_scale_factors[0],
          causal_mask=causal_mask,
          dropout_multiplier=dropout_multiplier,
          dtype=self.dtype)

      # Attention over external memory.
      if external_kv_w is not None:
        (external_keys_w, external_values_w) = external_kv_w
        y_ext = attention.external_attention(
            external_keys_w, external_values_w, queries_w,
            scale_factor=attention_scale_factors[0])
        if external_memory_bias is not None:
          ebias = external_memory_bias
          logging.info("tlayer: using external memory bias = %s", vshape(ebias))
          attn_ys_w = (attn_ys_w * (1 - ebias)) + (y_ext * ebias)
        elif self.memory_combine_with_local == "ADD":
          attn_ys_w += y_ext
        elif self.memory_combine_with_local == "STOP_FORWARD":
          attn_ys_w = y_ext + (attn_ys_w - jax.lax.stop_gradient(attn_ys_w))
        else:
          raise ValueError(
              f"Unexpected setting: {self.memory_combine_with_local = }")

      # Cross attention from input tokens to encoder or recurrent state.
      if local_cross_attention_kv is not None:
        logging.info("tlayer: cross-attention.")
        (c_keys, c_values) = local_cross_attention_kv

        # Cross-attention using queries2.
        cross_attn_ys_w = attention.simple_attention(
            c_keys, c_values, queries2_w, None,
            scale_factor=attention_scale_factors[1],
            dtype=self.dtype)
      else:
        cross_attn_ys_w = None

      # End function single_window_attention(...)
      return ((next_kvi_w, next_rec_state),
              (attn_ys_w, cross_attn_ys_w))

    # Initialize recurrent_tbase before calling jax.lax.scan.
    # Otherwise flax will throw a tantrum.
    if (self.recurrent_attention and 0 <= self.max_unrolled_windows and
        self.max_unrolled_windows < num_windows):
      assert self.recurrent_tbase is not None  # for ptype
      logging.info("tlayer: force initialization of recurrent_tbase.")
      self.recurrent_tbase.force_init(recurrent_state)

    # Perform sliding window attention over all keys,values,queries.
    # --------------------------------------------------------------
    initial_carry = (prev_kvi, recurrent_state)  # window state.
    kvqi = (keys, values, queries, queries2, importance)
    attn_inputs = (kvqi, external_kv)
    (next_carry, attn_outputs) = attention.split_and_scan(
        single_window_attention,
        initial_carry,
        attn_inputs,
        sections=num_windows,
        axis=1,
        max_unrolled_windows=self.max_unrolled_windows)
    (attn_ys, cross_attn_ys) = attn_outputs

    logging.info("tlayer: End windows.")

    # Post-attention MLP, resnet, and FFN.
    # ------------------------------------
    logging.info("tlayer: final FFN.")
    ys = self.tbase.post_attn_ffn(xs, attn_ys, cross_attn_ys)

    # Compute importance scores for each token if requested.
    if self.compute_importance:
      (batch_size, sequence_length, _) = ys.shape
      importance_score = self.importance_layer(ys)
      logging.debug("tlayer: importance_score_i = %s", vshape(importance_score))
      importance_score = importance_score.reshape(
          (batch_size, sequence_length))
      logging.info("tlayer: importance_score = %s", vshape(importance_score))
    else:
      importance_score = None

    next_window_state = next_carry if window_state is not None else None
    viz_dict = {}  # Visualizations, not currently enabled.
    return (ys, importance_score, next_window_state, decoder_state, viz_dict)  # pytype: disable=bad-return-type  # jax-ndarray

  def load_window_state(self, start_of_sequence: Array) -> WindowState:
    """Load cached state that is passed from one window to the next."""

    (mode, _, _) = self._get_cache_name_from_mode(self.mode)
    prev_kvi = self._get_cached_kvi(start_of_sequence, mode)
    rec_state = self._get_cached_recurrent_state(start_of_sequence, mode)
    if prev_kvi is not None:
      logging.info("tlayer: Loaded keys,values for mode %s from cache %s",
                   self.mode, mode)
    else:
      logging.info("tlayer: Skipping XL cache for mode %s.", self.mode)
    if rec_state is not None:
      logging.info("tlayer: Loaded recurrent state for mode %s from cache %s.",
                   self.mode, mode)
    return (prev_kvi, rec_state)

  def store_window_state(self, window_state: WindowState):
    """Write window state to the cache."""

    (mode, update_cache, _) = self._get_cache_name_from_mode(self.mode)
    (next_kvi, next_rec_state) = window_state  # pytype: disable=attribute-error
    if update_cache and next_kvi is not None:
      logging.info("tlayer: Storing keys,values for mode %s in cache %s.",
                   self.mode, mode)
      self._set_cached_kvi(next_kvi, mode)
    else:
      logging.info("tlayer: Skipping XL cache update for mode %s.", self.mode)
    if update_cache and next_rec_state is not None:
      logging.info("tlayer: Storing recurrent state for mode %s in cache %s.",
                   self.mode, mode)
      self._set_cached_recurrent_state(next_rec_state, mode)

  def get_recurrent_kv(self, window_state: WindowState):
    """Get the recurrent keys,values from window_state."""

    # TODO(delesley): optimize.
    # This isn't ideal, because we wind up computing the recurrent keys,values
    # twice -- once within the sliding window above, and again in the
    # DecoderStack, so they can be passed to other layers.  However, the
    # plumbing is a lot simpler this way.
    if window_state is None:
      return None
    (_, rec_state) = window_state
    if rec_state is None:
      return None
    logging.info("tlayer: get_recurrent_kv.")
    assert self.recurrent_tbase is not None  # for ptype
    (r_keys, r_values, _, _) = self.recurrent_tbase.kvq(rec_state)
    return (r_keys, r_values)

  def init_decoder_state(self, sequence_length: int,
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

    (mode, _, _) = self._get_cache_name_from_mode(self.mode)

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
    prev_kvi = self._get_cached_kvi(start_of_sequence, mode)
    if prev_kvi is not None:
      (pkeys, pvals, prev_imps) = prev_kvi
      assert prev_imps is None  # Not yet supported.
      assert pkeys.ndim == 4
      assert pkeys.shape[1] == self.window_length  # (b, wlen, num_heads, d)

      stored_keys = jax.lax.dynamic_update_slice_in_dim(
          stored_keys, pkeys, 0, axis=1)
      stored_values = jax.lax.dynamic_update_slice_in_dim(
          stored_values, pvals, 0, axis=1)

    # Grab the current recurrent_state, and precompute keys,values,queries.
    rstate = self._get_cached_recurrent_state(start_of_sequence, mode)
    if rstate is not None:
      assert self.recurrent_tbase is not None  # for ptype
      recurrent_kvq = self.recurrent_tbase.kvq(rstate)
    else:
      recurrent_kvq = None

    decoder_state_dict = {
        "keys": stored_keys,
        "values": stored_values,
        "current_index": start_index,
        "relative_position_bias": rel_position_bias,
        "recurrent_kvq": recurrent_kvq
    }
    return DecoderState(decoder_state_dict)

  def _next_decoder_state(self, decoder_state: DecoderState,
                          keys: Array, values: Array) -> Tuple[
                              DecoderState, Array, Array]:
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
    out_decoder_state["recurrent_kvq"] = decoder_state["recurrent_kvq"]

    return (DecoderState(out_decoder_state), out_keys, out_values)
