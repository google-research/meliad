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

"""A stack of transformer layers."""

from typing import Any, Dict, Optional, Sequence, Tuple

from absl import logging

from flax import linen
import gin
import jax
import jax.numpy as jnp
from transformer import language_model
from transformer import nn_components


Array = jax.Array
Metrics = Dict[str, Any]
vshape = nn_components.vshape


@gin.configurable
class TransformerStack(linen.Module):
  """A stack of transformer layers."""

  # Mode: e.g test or train.  Set by caller.
  mode: str

  # Configured by either the caller or by gin.
  embedding_dim: int = gin.REQUIRED
  batch_size: int = gin.REQUIRED

  # Usually Configured by gin.
  num_layers: int = gin.REQUIRED
  layer_factory: Any = gin.REQUIRED

  # Instantiates an additional layer, and uses it at one or more points.
  # The interface should be similar to nn_components.MLPWrapper or MLP.
  side_layer_factory: Any = None

  # Selects where the side_layer will be inserted.  If multiple indices are
  # used, the side layer will be applied (with shared weights) at multiple
  # locations.
  side_layer_indices: Sequence[int] = ()

  # A side layer must follow either the nn_components.SimpleMLP interface,
  # or the transformer_layer.TransformerLayer interface.  This boolean
  # should be set along with side_layer_factory to choose the right interface.
  # TODO(delesley): Possibly generalize this mechanism?
  side_layer_is_mlp: bool = True

  # Optional aux output from middle of stack.  -1 means no output.
  aux_output_layer_num: int = -1

  # Apply a final layernorm after the last layer.
  use_final_layernorm: bool = True

  # Usually set by parent.
  dtype: Any = gin.REQUIRED

  def is_training(self):
    return self.mode == "train"

  def setup(self):
    # A stack of transformer layers.
    layer_factory = self.layer_factory

    # Layer factor is instantiated num_layers times.
    layers = []
    for i in range(0, self.num_layers):
      layer = layer_factory(
          mode=self.mode,
          batch_size=self.batch_size,
          embedding_size=self.embedding_dim,
          name=f"transformer{i}",
          use_importance=False,
          dtype=self.dtype)
      layers.append(layer)
    self.transformer_layers = layers

    # The optional side layer is instantiated once, and used one or more times.
    # This can be used to implement, e.g., external memory.
    if self.side_layer_factory is not None:
      self._setup_side_layer()
    elif self.side_layer_indices:
      raise ValueError("Indices specified, but no side_layer_factory.")

    # Final layernorm.
    if self.use_final_layernorm:
      self.final_layernorm = nn_components.LayerNorm()

    # Aux layernorm -- for models that use the aux output.
    if self.aux_output_layer_num >= 0:
      self.aux_layernorm = nn_components.LayerNorm()

  def _setup_side_layer(self):
    """Set up the side layer, if configured."""
    if not self.side_layer_indices:
      raise ValueError("Indices should be specified when using side_layer.")
    # Check that indices are within range.
    for i in self.side_layer_indices:
      if not (i >= 0 and i < self.num_layers):
        raise ValueError(f"Index {i} for side_layer_indices is out of bounds.")
    if self.side_layer_is_mlp:
      # nn_components.SimpleMLP or MultiHeadMLP
      self.side_layer = self.side_layer_factory(
          num_output_features=self.embedding_dim,
          dtype=self.dtype
      )
      self.side_layer_norm = nn_components.LayerNorm()
    else:
      # transformer_layer.TransformerLayer
      self.side_layer = self.side_layer_factory(
          mode=self.mode,
          batch_size=self.batch_size,
          embedding_size=self.embedding_dim,
          use_importance=False,
          dtype=self.dtype
      )
      self.side_layer_norm = None

  def maybe_call_side_layer(self,
                            i: int,
                            xs: Array,
                            start_of_sequence: Array) -> Array:
    """Call the side layer, if configured."""
    if (self.side_layer_factory is None) or (i not in self.side_layer_indices):
      return xs

    logging.info("Simple TransformerStack: --- Side Layer (%d) ---", i)
    if self.side_layer_is_mlp:
      side_xs = self.side_layer_norm(xs)
      xs += self.side_layer(side_xs)  # residual connection
    else:
      xs += self.side_layer(xs, start_of_sequence)  # residual connection
    return xs

  def __call__(self,
               xs: Array,
               start_of_sequence: Array
              ) -> Tuple[Array, Optional[Array], Metrics]:
    """Apply a stack of transformer layers to xs.

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
    logging.info("Simple TransformerStack: xs = %s", vshape(xs))

    # Stack of transformer layers.
    ys = xs
    aux_ys = None
    for (i, layer) in enumerate(self.transformer_layers):
      logging.info("Simple TransformerStack: ---- Layer %d ----", i)
      ys = layer(ys, start_of_sequence)
      ys = self.maybe_call_side_layer(i, ys, start_of_sequence)

      # Possibly retrieve aux outputs from the middle of the stack.
      if i == self.aux_output_layer_num:
        aux_ys = ys

    # Apply layernorm before token prediction.
    if self.use_final_layernorm:
      ys = self.final_layernorm(ys)
    if aux_ys is not None:
      aux_ys = self.aux_layernorm(aux_ys)

    logging.info("Simple TransformerStack: ys = %s", vshape(ys))
    logging.info("Simple TransformerStack: aux_ys = %s", vshape(aux_ys))
    d_metrics = {}
    return (ys, aux_ys, d_metrics)


# TODO(delesley): Rename this to something more generic, like TokenLM.
@gin.configurable
class DecoderStack(linen.Module):
  """This is a simpler version of memory.decoder_stack.DecoderStack.

  Decoder stack implements a language model over a token vocabulary.
  It has an embedding table which maps token IDs --> embedding vectors,
  and uses the same table to map embedding vectors --> logits at the end.
  The language model itself is usually a TransformerStack.
  """

  # Mode: e.g test or train.  Set by caller.
  mode: str
  task_config: language_model.TransformerTaskConfig

  # Configured by gin.
  embedding_dim: int = gin.REQUIRED

  # A language model which operates on embedding vectors.
  # Usually TransformerStack, or something which obeys the same interface.
  stack_factory: Any = gin.REQUIRED

  final_dropout_rate: float = 0.0
  dtype: Any = jnp.float32

  def is_training(self):
    return self.mode == "train"

  def accepts_global_info(self) -> bool:
    # Declare that __call__ takes a global_info named parameter
    return True

  def setup(self):
    embed_init = linen.initializers.normal(stddev=1.0, dtype=jnp.float32)
    self.embed = linen.Embed(num_embeddings=self.task_config.vocab_size,
                             features=self.embedding_dim,
                             embedding_init=embed_init)
    self.layer_stack = self.stack_factory(
        mode=self.mode,
        embedding_dim=self.embedding_dim,
        batch_size=self.task_config.batch_size,
        dtype=self.dtype)

  def __call__(self,
               inputs: Dict[str, Array],
               *,
               global_info: Optional[Dict[str, Any]] = None
              ) -> Tuple[Array, Metrics]:
    """Call the decoder stack.

    This function will embed tokens, run the embeddings through a language
    model (usually a stack of transformer layers), and then compute logits for
    the target tokens using the transpose of the embeddings.  It returns
    un-normalized (pre-softmax) logits.

    Args:
      inputs: Dictionary of shape {
          "input_tokens": Integer array of shape [batch_size, sequence_length]
          "start_of_sequence": Boolean array of shape [batch_size],
              which indicates whether a sequence is at the start of sequence.
      }
      global_info:  A dictionary of global information (e.g., current training
        step as "step").

    Returns:
       (logits, of shape [batch_size, sequence_length, vocab_size],
        d_metrics: dictionary of metrics,
       )
    """

    input_tokens = inputs["input_tokens"]
    target_tokens = inputs["target_tokens"]
    start_of_sequence = inputs["start_of_sequence"]
    del target_tokens
    del global_info

    # Embed tokens.
    embeddings = self.embed(input_tokens)  # (batch_size, seq_len, num_hidden)
    embeddings = embeddings.astype(self.dtype)
    logging.info("DecoderStack: embeddings = %s", vshape(embeddings))

    # Run Transformer layers.
    (ys, _, d_metrics) = self.layer_stack(embeddings,
                                          start_of_sequence)
    logging.info("DecoderStack: ys = %s", vshape(ys))

    # Final dropout before token prediction.
    logging.info("DecoderStack: final_dropout = %s", self.final_dropout_rate)
    drop_tile_shape = (1, 128, self.embedding_dim)
    get_dropout_rng = lambda: self.make_rng("dropout")
    ys = nn_components.tiled_dropout(ys,
                                     drop_tile_shape,
                                     self.final_dropout_rate,
                                     rng_function=get_dropout_rng,
                                     deterministic=not self.is_training())

    # Reverse embedding to generate logits which predict the output tokens.
    logits = self.embed.attend(ys)  # (..., seq_len, vocab_size)
    logging.info("DecoderStack: logits = %s", vshape(logits))

    # Normalize so that the range of logits is reasonable.
    logits = logits / jnp.sqrt(ys.shape[-1]).astype(self.dtype)

    return (logits, d_metrics)
