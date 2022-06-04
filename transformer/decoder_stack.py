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

"""Hierarchical transformer."""

import functools
from typing import Any, Callable, Optional, Sequence, Tuple

from absl import logging

from flax import linen as nn
from flax import struct
import gin
import jax.numpy as jnp
from transformer import attention
from transformer import metric_utils
from transformer import nn_components
from transformer import position
from transformer import transformer_layer


Array = Any


# Basic task options are shared among multiple classes.
@gin.configurable
@struct.dataclass
class TransformerTaskConfig:
  """Configuration hyperparameters for sequence-to-sequence tasks."""

  dataset_name: str = "synthetic"
  train_split: str = "train"
  test_split: str = "test"
  sequential_chunks: bool = True  # Process chunks of text in sequential order.

  sequence_length: int = 4096
  batch_size: int = 1  # per device batch size
  vocab_size: int = 256


DStackDecoderState = Tuple[transformer_layer.DecoderState, ...]
DStackWindowState = Tuple[transformer_layer.WindowState, ...]


@gin.configurable
class DecoderStack(nn.Module):
  """Stack of transformer decoder layers."""

  mode: str
  task_config: TransformerTaskConfig = gin.REQUIRED

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
  final_mlp_factory: Optional[Callable[[int], nn.Module]] = None

  # Enable recurrence on particular layers.
  recurrent_layer_indices: Sequence[int] = ()
  feedback_recurrence: bool = True

  # The factory function which creates a MemoryManager, or None.
  memory_factory: Any = None
  # Layers to equip with external memory.
  memory_layer_indices: Sequence[int] = ()

  dtype: Any = jnp.float32

  def is_training(self):
    return self.mode == "train"

  def supports_generate(self) -> bool:
    return all([lyr.supports_generate() for lyr in self.transformer_layers])

  def setup(self):
    task_config = self.task_config

    embed_init = nn.initializers.normal(stddev=self.embedding_stddev,
                                        dtype=jnp.float32)
    self.embed = nn.Embed(num_embeddings=task_config.vocab_size,
                          features=self.embedding_size,
                          embedding_init=embed_init)

    # Create a memory_factory.MemoryManager object, which is shared among
    # all transformer layers.  Each layer will use the MemoryManager object
    # to instantiate a block of memory for that layer.
    memory = None
    if self.memory_factory is not None:
      if self.memory_layer_indices:
        memory = self.memory_factory(batch_size=task_config.batch_size,
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
            num_heads=mem.num_heads)
      layers.append(layer_fn())
    self.transformer_layers = layers

    if self.use_final_layernorm:
      self.final_layernorm = nn_components.LayerNorm()

    if self.final_mlp_factory is not None:
      self.final_mlp = self.final_mlp_factory(self.embedding_size)

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
    """Write window state to the cache."""
    for (layer, wstate) in zip(self.transformer_layers, window_state):
      layer.store_window_state(wstate)

  def _eval_layer_stack(self, xs: Array, start_of_sequence: Array,
                        window_state: Optional[DStackWindowState],
                        decoder_state: Optional[DStackDecoderState]) -> (
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
          ys, start_of_sequence,
          importance=importance,
          cross_attention_kv=cross_kv,   # cross-attend to recurrent_kv.
          window_state=wstate_i,
          decoder_state=dstate_i)
      next_window_states.append(n_wstate_i)
      next_decoder_states.append(n_dstate_i)
      attn_viz_dicts.append(viz_dict)

    window_state = tuple(next_window_states)
    decoder_state = tuple(next_decoder_states)
    return (ys, window_state, decoder_state, attn_viz_dicts)

  def __call__(self,
               input_tokens: Array,
               target_tokens: Array,
               start_of_sequence: Array,
               decoder_state: Optional[DStackDecoderState] = None) -> (
                   Tuple[Array, Optional[DStackDecoderState], Any]):
    """Call the decoder stack.

    This function will embed tokens, run the embeddings through a stack of
    decoder layers, and then compute logits for the target tokens using the
    transpose of the embeddings.  It returns un-normalized (pre-softmax)
    logits.

    Args:
      input_tokens: Integer array of shape [batch_size, sequence_length]
      target_tokens: For compatibility.  Ignored by this class.
      start_of_sequence: Boolean array of shape [batch_size],
          which indicates whether a sequence is at the start of sequence.
      decoder_state: State object for autoregressive decoding,
          created from init_decoder_state.

    Returns:
       (logits, of shape [batch_size, sequence_length, vocab_size],
        next_decoder_state: for autoregressive decoding,
        viz_dict: dictionary of visualizations,
       )
    """
    del target_tokens
    task_config = self.task_config

    # Embed tokens.
    embeddings = self.embed(input_tokens)  # (batch_size, seq_len, num_hidden)
    embeddings = embeddings.astype(self.dtype)
    sequence_length = embeddings.shape[1]
    logging.info("dstack: embeddings = %r", embeddings)

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
      logging.info("dstack: absolute positions = %r", positions)
      embeddings = embeddings + positions

    # Function to run the whole transformer stack on a single window.
    # ---------------------------------------------------------------
    def single_window_stack(carry, inputs_w):
      (window_state_w, start_of_seq_w) = carry
      (outputs_w, window_state_w, _, _) = self._eval_layer_stack(
          inputs_w, start_of_seq_w,
          window_state=window_state_w, decoder_state=None)

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
      (cstate, ys) = attention.split_and_scan(single_window_stack,
                                              cstate,
                                              embeddings,
                                              sections=num_windows,
                                              axis=1)
      (window_state, _) = cstate

      # Cache state for the next training step, for truncated BPTT.
      self.store_window_state(window_state)
      attn_viz_dicts = {}  # Temporarily disabled.
    else:
      logging.info("dstack: autoregressive generator.")
      # Run as an autoregressive decoder: evaluate the whole stack on a token.
      # Do not load or store window_state; decoder_state is used instead.
      (ys, _, decoder_state, _) = self._eval_layer_stack(
          embeddings, start_of_sequence,
          window_state=None, decoder_state=decoder_state)
      attn_viz_dicts = {}

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
    logging.info("dstack: logits = %r", logits)

    # Normalize so that the range of logits is reasonable.
    logits = logits / jnp.sqrt(logits.shape[-1]).astype(self.dtype)

    # Produce various visualizations in generate mode.
    # TODO(delesley): Too many visualizations crashes the summary writer.
    if self.mode == "generate":
      img_dict = self._make_images(attn_viz_dicts, [])
      hist_dict = {}  # metric_utils.make_histograms(attn_viz_dicts)
      info_dict = {**img_dict, **hist_dict}
    else:
      info_dict = {}  # Don't output any visualizations.

    return (logits, decoder_state, info_dict)

  def _make_importance_image(self, importance_list, scaled=True) -> Array:
    rows = []
    for imp in importance_list:
      rows += [imp] * 8  # Rows are 8 pixels high for better visability.
    image = jnp.stack(rows)
    if scaled:
      image = jnp.exp(image)
    image = metric_utils.normalize_image(image, True)
    return metric_utils.reshape_image(image)

  def _make_images(self, viz_dicts, importance_list):
    image_dict = {}
    for (i, viz_dict) in enumerate(viz_dicts):
      if "attn_importance_gate" in viz_dict:
        imp_gate = viz_dict["attn_importance_gate"][0]  # First item in batch.
        imp_strip = metric_utils.normalize_image(imp_gate[:, 0:8, :], True)
      else:
        imp_strip = None

      for (k, attn_images) in viz_dict.items():
        if k not in {"attn_content",
                     "attn_pre_softmax",
                     "attn_log",
                     "attn",
                     "attn_position_bias",
                     "attn_importance_bias",
                     "attn_importance_gate"}:
          continue

        attn_img = attn_images[0]  # Grab the first item in the batch.
        attn_img = metric_utils.normalize_image(attn_img,
                                                as_group=(k != "attn"))
        if imp_strip is not None and k in {"attn_log", "attn"}:
          # Show importance bias in a strip at the bottom of the image.
          attn_img = metric_utils.overlay_images(attn_img, imp_strip)
        attn_img = metric_utils.reshape_image(attn_img)  # Returns None on fail.
        if attn_img is not None:
          image_dict[k + "_" + str(i)] = attn_img

    if importance_list:
      # Create an image out of the importance for each layer.
      image_dict["importance_gate"] = self._make_importance_image(
          importance_list, scaled=True)
      image_dict["importance_raw"] = self._make_importance_image(
          importance_list, scaled=False)
    return image_dict
