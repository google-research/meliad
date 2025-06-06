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

"""Sequence to sequence model."""

from typing import Any, Dict, Optional, Tuple

from absl import logging
import flax
import gin
import jax.numpy as jnp
import  metrics_summary
import  model_base
from transformer import language_model
from transformer import nn_components
from transformer import text_dataset


Array = model_base.Array
Metrics = model_base.Metrics
PRNGKey = model_base.PRNGKey
CollectionFilter = model_base.CollectionFilter
FrozenVariableDict = model_base.FrozenVariableDict
VariableDict = model_base.VariableDict
TransformerTaskConfig = language_model.TransformerTaskConfig


vshape = nn_components.vshape
average_metric = metrics_summary.average_metric
output_value_metric = metrics_summary.output_value_metric


@gin.configurable
class DecoderOnlyLanguageModelRunner(model_base.ModelBase):
  """Decoder only language modeling.

  Can be used for either autoregressive decoder-only models, or for masked
  language modeling.

  Like model_base.WrappedModel, this class is a wrapper around another model.
  """

  def __init__(self,
               mode: str,
               task_config: TransformerTaskConfig = gin.REQUIRED,
               decoder_factory: Any = gin.REQUIRED,
               autoregressive: bool = True,
               sample_method: str = "sample",   # Can be {"sample", "greedy"}
               output_token_losses: bool = False):
    """Implements a DecoderOnlyLanguageModel.

    Args:
      mode:            One of {"train", "test", "generate"}.
      task_config:     Information about the langauge modelling task.
      decoder_factory: A function to instantiate the underlying decoder model.
      autoregressive:  If true, the model will be configured to predict the
          next token, if false, it will predict the current token, which is
          is appropriate for autoencoders or masked language models.
          (Note that decoder_factory must be separately configured to be a
          causal model in order to do autoregressive decoding.)
      sample_method:   The method to use for sampling during inference.
          Can be one of {"sample", "greedy"}
      output_token_losses: If true, record individual per-token losses.
    """
    self.mode = mode
    self.task_config = task_config
    self.autoregressive = autoregressive
    self.sample_method = sample_method
    self.output_token_losses = output_token_losses
    self.decoder = decoder_factory(mode=mode, task_config=task_config)

  def get_fake_input(self, batch_size: int):
    """Returns a fake input of the appropriate shape for initialization.

    Args:
      batch_size: The batch size of the input to create.

    Returns:
      A set of dummy inputs, with the given batch size.
    """
    b = batch_size
    fake_input_dict = {
        "targets": jnp.ones([b, self.task_config.sequence_length],
                            dtype=jnp.int32),
        "start_of_sequence": jnp.ones([b], dtype=jnp.bool_),
        "epoch": jnp.ones([b], dtype=jnp.int32),
    }

    datapipe = text_dataset.DataPipelineConfig()
    if datapipe.compute_token_boundaries:
      # Add token boundaries if requested.
      fake_input_dict["token_boundaries"] = jnp.ones(
          [b, self.task_config.sequence_length], dtype=jnp.float32)

    if datapipe.loss_mask_fn is not None:
      # We are not adding the loss mask to the dummy input by default as it can
      # cause a slowdown during evaluation and perhaps inference.
      fake_input_dict["loss_mask"] = jnp.ones(
          [b, self.task_config.sequence_length], dtype=jnp.bool_)
    return fake_input_dict

  def unpack_inputs(self, inputs) -> Tuple[Array, Array, Array,
                                           Optional[Array], Array]:
    """Unpack the dictionary of inputs."""

    # Unpack input dictionary.
    input_tokens = jnp.asarray(inputs["targets"])   # [b, seq_len]
    start_of_sequence = jnp.asarray(inputs["start_of_sequence"])  # [b]
    if "loss_mask" in inputs:
      loss_mask = jnp.asarray(inputs["loss_mask"])  # [b, seq_len]
    else:
      loss_mask = jnp.ones(input_tokens.shape, dtype=jnp.bool_)

    # Check that inputs are valid.
    task_config = self.task_config
    assert input_tokens.ndim == 2
    assert input_tokens.shape[1] == task_config.sequence_length
    assert start_of_sequence.shape[0] == input_tokens.shape[0]

    # Sanity check to avoid out-of-bounds on token lookup.
    input_tokens = input_tokens % task_config.vocab_size

    # Compute target_tokens.
    if self.autoregressive:
      # Predict the next token -- targets are inputs shifted over by 1.
      (target_tokens, loss_mask) = language_model.targets_from_input_tokens(
          input_tokens, loss_mask)
    else:
      # For autoencoder or masked language models.
      target_tokens = input_tokens

    # Update loss mask.
    loss_mask = language_model.add_padding_to_loss_mask(
        loss_mask, input_tokens, target_tokens)

    logging.info("langmodel: input_tokens = %s", vshape(input_tokens))
    logging.info("langmodel: target_tokens = %s", vshape(target_tokens))
    logging.info("langmodel: loss_mask = %s", vshape(loss_mask))
    logging.info("langmodel: start_of_sequece = %s", vshape(start_of_sequence))

    if "token_boundaries" in inputs:
      token_boundaries = jnp.asarray(inputs["token_boundaries"])
      logging.info("langmodel: token_boundaries = %r", vshape(token_boundaries))
      assert token_boundaries.shape == input_tokens.shape
    else:
      token_boundaries = None

    return (input_tokens,
            target_tokens,
            loss_mask,
            token_boundaries,
            start_of_sequence)

  def model_init(self,
                 prng_dict: Dict[str, PRNGKey],
                 dummy_input: Any) -> FrozenVariableDict:
    logging.info("==== Initializing model with mode %s ====", self.mode)
    dummy_decoder_inputs = {
        "input_tokens": dummy_input["targets"],
        "target_tokens": dummy_input["targets"],
        "start_of_sequence": dummy_input["start_of_sequence"]
    }
    if "loss_mask" in dummy_input:
      dummy_decoder_inputs["loss_mask"] = dummy_input["loss_mask"]
    if "token_boundaries" in dummy_input:
      dummy_decoder_inputs["token_boundaries"] = dummy_input["token_boundaries"]
    return self.decoder.model_init(prng_dict, dummy_decoder_inputs)

  def model_apply(
      self,
      params_state: VariableDict,
      inputs: Any,
      prng_dict: Dict[str, PRNGKey],
      mutable_keys: CollectionFilter,
      *,
      global_info: Dict[str, Any],
  ) -> Tuple[Tuple[Array, Metrics], FrozenVariableDict]:
    logging.info("==== Compiling model for mode %s ====", self.mode)

    if self.mode == "generate":
      # TODO(delesley): re-enable autoregressive generation.
      logging.info("langmodel: "
                   "Autoregressive generation is currently disabled.")
      mutable_state = {k: params_state[k] for k in mutable_keys}
      return ((jnp.array(0.0), {}), flax.core.freeze(mutable_state))

    # Unpack inputs from the training data.
    # The inputs may vary from task to task.
    (input_tokens,
     target_tokens,
     loss_mask,
     token_boundaries,
     start_of_sequence) = self.unpack_inputs(inputs)

    # Package relevant inputs back into a dictionary for the wrapped model.
    model_inputs = {
        "input_tokens": input_tokens,
        "target_tokens": target_tokens,
        "start_of_sequence": start_of_sequence,
    }
    if token_boundaries is not None:
      model_inputs["token_boundaries"] = token_boundaries

    # Invoke the actual language model.
    # The language model will return pre-softmax logits for the targets.
    next_state: FrozenVariableDict
    ((logits, d_metrics), next_state) = self.decoder.model_apply(
        params_state,
        model_inputs,
        prng_dict,
        mutable_keys,
        global_info=global_info,
    )
    logging.info("langmodel: d_metrics = %s", d_metrics.keys())

    # Softmax cross-entropy loss for each target token.
    (token_losses, token_entropy) = language_model.softmax_cross_entropy_loss(
        logits, target_tokens)
    # Apply loss mask.
    loss_mask = loss_mask.astype(token_losses.dtype)
    token_losses *= loss_mask
    # Add up all the losses to get a final training loss.
    training_loss = jnp.sum(token_losses)

    if "aux_loss" in d_metrics:
      # Models may choose to return auxiliary losses.
      aux_loss_metric = d_metrics["aux_loss"]
      assert isinstance(aux_loss_metric, metrics_summary.Average)
      aux_loss = aux_loss_metric.total
      logging.info("langmodel: found auxiliary loss: %s", vshape(aux_loss))
      assert aux_loss.shape == ()
      training_loss += aux_loss

    # Compute metrics.
    metrics = language_model.compute_token_metrics(inputs,
                                                   token_losses=token_losses,
                                                   loss_mask=loss_mask,
                                                   logits=logits,
                                                   target_tokens=target_tokens)
    metrics["loss"] = average_metric(training_loss)
    metrics.update(d_metrics)

    if self.output_token_losses:
      metrics["token_losses"] = output_value_metric(token_losses)
      metrics["token_entropy"] = output_value_metric(token_entropy)

    return ((training_loss, metrics), next_state)
