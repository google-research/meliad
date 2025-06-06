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

"""Hierarchical transformer."""

import functools
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from absl import logging

from flax import linen
import gin
import jax
import jax.numpy as jnp
from transformer import attention
from transformer import language_model
from transformer import nn_components
from transformer import position
from transformer.memory import transformer_layer


Array = jax.Array
vshape = nn_components.vshape


DStackDecoderState = Tuple[transformer_layer.DecoderState, ...]
DStackWindowState = Tuple[transformer_layer.WindowState, ...]


@gin.configurable
class DecoderStack(linen.Module):
  """Stack of transformer decoder layers."""

  # Supplied by DecoderOnlyLanguageModel
  mode: str
  task_config: language_model.TransformerTaskConfig

  # Configurable hyperparameters.
  num_layers: int = gin.REQUIRED
  embedding_size: int = gin.REQUIRED
  embedding_stddev: float = 1.0

  # The class to use for an individual transformer layer.
  layer_factory: Any = gin.REQUIRED

  # Window length to use for the decoder stack.
  # If nonzero, use this instead of TransformerLayer.window_length.
  dstack_window_length: int = 0
  use_absolute_positions: bool = False
  use_final_layernorm: bool = True
  final_dropout_rate: float = 0.0
  final_mlp_factory: Optional[Callable[[int], linen.Module]] = None

  # Enable recurrence on particular layers.
  recurrent_layer_indices: Sequence[int] = ()
  feedback_recurrence: bool = True

  # The factory function which creates a MemoryManager, or None.
  memory_factory: Any = None
  # Layers to equip with external memory.
  memory_layer_indices: Sequence[int] = ()
  # Disable position encoding on layers with external memory?
  disable_position_encoding_for_memory_layers: bool = False

  dtype: Any = jnp.float32

  def is_training(self):
    return self.mode == "train"

  def supports_generate(self) -> bool:
    return all([lyr.supports_generate() for lyr in self.transformer_layers])

  def accepts_global_info(self) -> bool:
    # Declare that __call__ takes a global_info named parameter
    return True

  def setup(self):
    task_config = self.task_config

    embed_init = linen.initializers.normal(stddev=self.embedding_stddev,
                                           dtype=jnp.float32)
    self.embed = linen.Embed(num_embeddings=task_config.vocab_size,
                             features=self.embedding_size,
                             embedding_init=embed_init)

    # Create a memory_factory.MemoryManager object, which is shared among
    # all transformer layers.  Each layer will use the MemoryManager object
    # to instantiate a block of memory for that layer.
    memory = None
    if self.memory_factory is not None:
      if self.memory_layer_indices:
        memory = self.memory_factory(batch_size=task_config.batch_size,   # pylint: disable=not-callable
                                     mode=self.mode)
      else:
        logging.warning(
            "Memory factory specified, but memory_layer_indices is empty.")

    # Allow negative numbers in memory_layer_indices.
    # Negative numbers refer to layers at the top of the stack.
    for k in self.memory_layer_indices:
      if k < -self.num_layers or k >= self.num_layers:
        raise ValueError(f"Invalid memory layer index {k}")
    # The % operator will convert negative k to self.num_layers + k.
    mem_layer_indices = [
        idx % self.num_layers for idx in self.memory_layer_indices
    ]

    # Allow negative numbers in recurrent_layer_indices.
    for k in self.recurrent_layer_indices:
      if k < -self.num_layers or k >= self.num_layers:
        raise ValueError(f"Invalid recurrent layer index {k}")
    recurrent_layer_indices = [
        idx % self.num_layers for idx in self.recurrent_layer_indices
    ]
    # Turn on cross attention if there are recurrent layers with feedback.
    enable_cross_attn = (self.feedback_recurrence and
                         self.recurrent_layer_indices and
                         self.dstack_window_length > 0)

    layers = []
    for i in range(0, self.num_layers):
      mem = memory if (i in mem_layer_indices) else None
      rec_i = i in recurrent_layer_indices
      layer_fn = functools.partial(
          self.layer_factory,
          mode=self.mode,
          batch_size=self.task_config.batch_size,
          embedding_size=self.embedding_size,
          name=f"transformer{i}",
          recurrent_attention=rec_i,
          cross_attention=enable_cross_attn and not rec_i)
      if mem:
        logging.info("Using external memory with transformer layer %d.", i)
        layer_fn = functools.partial(
            layer_fn,
            memory=mem,
            # We use partial function applications here only to avoid
            # overwriting the head size unless memory is involved.
            head_size=mem.key_size,
            num_heads=mem.num_heads,
        )
        if self.disable_position_encoding_for_memory_layers:
          layer_fn = functools.partial(layer_fn, relative_position_type=None)
      layers.append(layer_fn())
    self.transformer_layers = layers

    if self.use_final_layernorm:
      self.final_layernorm = nn_components.LayerNorm()

    if self.final_mlp_factory is not None:
      self.final_mlp = self.final_mlp_factory(self.embedding_size)  # pylint: disable=not-callable

  def init_decoder_state(self, sequence_length: int,
                         start_of_sequence: Array) -> DStackDecoderState:
    """Return initial state for autoregressive generation."""
    return tuple([
        layer.init_decoder_state(sequence_length, start_of_sequence)
        for layer in self.transformer_layers
    ])

  def load_window_state(self, start_of_sequence: Array) -> DStackWindowState:
    """Load cached state that is passed from one window to the next."""
    return tuple([
        layer.load_window_state(start_of_sequence)
        for layer in self.transformer_layers
    ])

  def store_window_state(self, window_state: DStackWindowState):
    """Write window state for each layer to the cache."""
    for (layer, wstate) in zip(self.transformer_layers, window_state):
      layer.store_window_state(wstate)

  def _eval_layer_stack(self, xs: Array, start_of_sequence: Array,
                        window_state: Optional[DStackWindowState],
                        decoder_state: Optional[DStackDecoderState],
                        *,
                        global_info: Dict[str, Any]) -> (
                            Tuple[Array, Optional[DStackWindowState],
                                  Optional[DStackDecoderState], Any]):
    """Evaluate a stack of transformer layers on an input."""

    ys = xs  # (batch_size, seq_len, num_hidden)
    importance = None  # (batch_size, sequence_length)
    next_window_states = []
    next_decoder_states = []
    attn_viz_dicts = []

    # If we have a recurrent layer, grab the keys and values from it.
    # All other layers can then cross-attend to the recurrent keys and values.
    recurrent_kv = None
    enable_cross_attn = (self.feedback_recurrence and
                         self.recurrent_layer_indices and
                         self.dstack_window_length > 0)
    if enable_cross_attn and window_state is not None:
      # TODO(delesley): fix this so it works with the autoregressive decoder.
      assert decoder_state is None
      logging.info("dstack: using recurrent cross attention on all layers.")
      for (layer, wstate_i) in zip(self.transformer_layers, window_state):
        rkv = layer.get_recurrent_kv(wstate_i)
        if rkv is not None:
          recurrent_kv = rkv

    # Apply transformer layers.
    for (i, layer) in enumerate(self.transformer_layers):
      if layer.recurrent_attention:
        cross_kv = None  # The recurrent layer handles rkv internally.
      else:
        cross_kv = recurrent_kv  # Other layers cross-attend to recurrent one.

      logging.info("dstack: ---- Layer %d ----", i)
      wstate_i = None if window_state is None else window_state[i]
      dstate_i = None if decoder_state is None else decoder_state[i]
      (ys, importance, n_wstate_i, n_dstate_i, viz_dict) = layer(
          ys,
          start_of_sequence,
          importance=importance,
          cross_attention_kv=cross_kv,  # cross-attend to recurrent_kv.
          window_state=wstate_i,
          decoder_state=dstate_i,
          global_info=global_info,
      )
      next_window_states.append(n_wstate_i)
      next_decoder_states.append(n_dstate_i)
      attn_viz_dicts.append(viz_dict)

    window_state = tuple(next_window_states)
    decoder_state = tuple(next_decoder_states)
    return (ys, window_state, decoder_state, attn_viz_dicts)

  def __call__(self,
               inputs: Dict[str, Array],
               *,
               global_info: Dict[str, Any],
               decoder_state: Optional[DStackDecoderState] = None) -> Any:
    """Call the decoder stack.

    This function will embed tokens, run the embeddings through a stack of
    decoder layers, and then compute logits for the target tokens using the
    transpose of the embeddings.  It returns un-normalized (pre-softmax)
    logits.

    Args:
      inputs: Dictionary of shape {
          "input_tokens": Integer array of shape [batch_size, sequence_length]
          "start_of_sequence": Boolean array of shape [batch_size],
              which indicates whether a sequence is at the start of sequence.
      }
      global_info:  A dictionary of global information (e.g., current training
        step as "step").
      decoder_state: State object for autoregressive decoding,
          created from init_decoder_state.

    Returns:
       (logits, of shape [batch_size, sequence_length, vocab_size],
        next_decoder_state: for autoregressive decoding,
        viz_dict: dictionary of visualizations,
       )
    """

    input_tokens = inputs["input_tokens"]
    target_tokens = inputs["target_tokens"]
    start_of_sequence = inputs["start_of_sequence"]

    del target_tokens
    task_config = self.task_config

    # Embed tokens.
    embeddings = self.embed(input_tokens)  # (batch_size, seq_len, num_hidden)
    embeddings = embeddings.astype(self.dtype)
    sequence_length = embeddings.shape[1]
    logging.info("dstack: embeddings = %s", vshape(embeddings))

    # Add absolute position encodings if necessary.
    if self.use_absolute_positions:
      # Use a large max_wavelength so that only part of the input vector
      # is used for positions.
      positions = position.position_encoding(
          num_positions=task_config.sequence_length,
          input_dim=self.embedding_size,
          max_wavelength=10_000)
      positions = jnp.asarray(positions, dtype=self.dtype)
      positions = jnp.expand_dims(positions, 0)  # Add batch dimension.
      logging.info("dstack: absolute positions = %s", vshape(positions))
      embeddings = embeddings + positions

    # Function to run the whole transformer stack on a single window.
    # ---------------------------------------------------------------
    def single_window_stack(carry, inputs_w):
      (window_state_w, start_of_seq_w) = carry
      (outputs_w, window_state_w, _, _) = self._eval_layer_stack(
          inputs_w,
          start_of_seq_w,
          window_state=window_state_w,
          decoder_state=None,
          global_info=global_info,
      )

      # start_of_sequence is false after the first window.
      bsize = self.task_config.batch_size
      next_start_of_seq = jnp.asarray([False] * bsize, dtype=jnp.bool_)
      return ((window_state_w, next_start_of_seq), outputs_w)

    # Find the number of windows.  A sequence may be split into multiple
    # windows here, or alternatively, it may be split (or further split) within
    # TransformerLayer, depending on configuration.
    if (self.dstack_window_length == 0 or
        self.dstack_window_length >= sequence_length):
      num_windows = 1
    else:
      num_windows = sequence_length // self.dstack_window_length
      assert (num_windows * self.dstack_window_length) == sequence_length

    # Evaluate the stack of layers, scanning over windows if configured.
    # ------------------------------------------------------------------
    if decoder_state is None:
      logging.info("dstack: scanning over %d windows.", num_windows)
      # Load cached state from the previous training step, for truncated BPTT.
      window_state = self.load_window_state(start_of_sequence)

      # Scan single_window_stack over the sequence.
      cstate = (window_state, start_of_sequence)
      (cstate, ys) = attention.split_and_scan(
          single_window_stack, cstate, embeddings, sections=num_windows, axis=1
      )  # pytype: disable=wrong-arg-types
      (window_state, _) = cstate

      # Cache state for the next training step, for truncated BPTT.
      self.store_window_state(window_state)
    else:
      logging.info("dstack: autoregressive generator.")
      # Run as an autoregressive decoder: evaluate the whole stack on a token.
      # Do not load or store window_state; decoder_state is used instead.
      (ys, _, decoder_state, _) = self._eval_layer_stack(
          embeddings,
          start_of_sequence,
          window_state=None,
          decoder_state=decoder_state,
          global_info=global_info,
      )

    # Apply layernorm to the final output, before calculating logits.
    # With a pre-layernorm architecture, this has to be done here.
    if self.use_final_layernorm:
      logging.info("dstack: Final layernorm.")
      ys = self.final_layernorm(ys)

    # Final dropout before token prediction.
    drop_tile_shape = (1, 128, self.embedding_size)
    get_dropout_rng = lambda: self.make_rng("dropout")
    ys = nn_components.tiled_dropout(ys, drop_tile_shape,
                                     self.final_dropout_rate,
                                     rng_function=get_dropout_rng,
                                     deterministic=not self.is_training())

    # Apply an MLP at the very end to convert the output of the transformer
    # into a vector to look up target tokens in the embedding table.
    # This final layer allows the NN to distinguish between the "input context",
    # which is returned by the transformer resnet, and the "predicted target".
    if self.final_mlp_factory is not None:
      logging.info("dstack: Final MLP layer.")
      ys = self.final_mlp(ys)

    # Reverse embedding to generate logits which predict the output tokens.
    logits = self.embed.attend(ys)  # (..., seq_len, vocab_size)
    logging.info("dstack: logits = %s", vshape(logits))

    # Normalize so that the range of logits is reasonable.
    logits = logits / jnp.sqrt(ys.shape[-1]).astype(self.dtype)

    d_metrics = {}  # No additional metrics
    if decoder_state is None:
      return (logits, d_metrics)
    else:
      # TODO(delesley): fix generation.
      return (logits, decoder_state, d_metrics)   # For use by generate() only.

  def generate(self, inputs: Any, sequence_length: int) -> Array:
    """Generate an output sequence.

    Args:
      inputs: the same as argument to _call_.
      sequence_length: the length of sequence to generate.

    Returns:
      An array of generated tokens of shape (batch_size, sequence_length).
    """
    # TODO(delesley): Add support for passing the prefix as an argument.
    # TODO(delesley): Add support for temperature, gumbel softmax, beam search.

    batch_size = self.task_config.batch_size
    input_tokens = inputs["targets"]                  # [b,seq_len]
    start_of_sequence = inputs["start_of_sequence"]   # [b]

    # Initialize decoder.
    dstate = self.init_decoder_state(sequence_length,
                                     start_of_sequence)

    # TODO(delesley): Handle start-of-sequence in a better way.
    # There is no special token for start of sequence, so we grab the first
    # one from the ground-truth input data.
    first_token = input_tokens[:, 0:1]
    no_start_of_seq = jnp.array([False] * batch_size, dtype=jnp.bool_)
    sample_method = self.sample_method
    sample_prng = self.make_rng("sample")

    # Greedy autoregressive decoder function.
    def loop_fn(scan_state: Any, i: Any) -> Tuple[Any, Array]:
      prng = jax.random.fold_in(sample_prng, i)
      (dstate, input_token) = scan_state
      del i
      (logits, dstate, _) = self.__call__(
          inputs={
              "input_tokens": input_token,
              "start_of_sequence": no_start_of_seq,
          },
          decoder_state=dstate,
          global_info={},
      )
      if sample_method == "sample":
        logging.info("Using categorical sampling.")
        output_token = jax.random.categorical(prng, logits, axis=-1)
      elif sample_method == "greedy":
        logging.info("Using greedy sampling.")
        output_token = jnp.argmax(logits, axis=-1)
      else:
        raise ValueError(f"Invalid sampling method: {sample_method}")
      logging.info("generate_loop_fn: output_token = %s", vshape(output_token))
      return ((dstate, output_token), output_token)

    # Scan over the sequence length.
    iterations = jnp.arange(sequence_length)
    initial_scan_state = (dstate, first_token)
    (_, output_tokens) = jax.lax.scan(loop_fn, initial_scan_state, iterations)
    logging.info("generate: output_tokens = %s", vshape(output_tokens))

    # Output_tokens has shape (sequence_length, batch_size, 1)
    assert output_tokens.shape == (sequence_length, batch_size, 1)
    output_tokens = jnp.reshape(
        output_tokens, (sequence_length, self.task_config.batch_size))
    output_tokens = output_tokens.transpose([1, 0])
    return output_tokens
