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

"""Sequence to sequence model."""

from typing import Any, Callable, Dict, Tuple

from absl import logging

from flax import linen as nn
from flax.training import common_utils
import gin

import jax
import jax.numpy as jnp
import numpy as np
import seqio

import  metrics_summary
from transformer import decoder_stack
from transformer import metric_utils
from transformer import text_dataset


Array = jnp.ndarray
MetricsSummary = metrics_summary.MetricsSummary


# TODO(mrabe): Remove this function and find a better way to turn text metrics
# into text on tensorboard.
def process_summaries(vocab: seqio.Vocabulary,
                      met_summary: MetricsSummary,
                      mode: str) -> MetricsSummary:
  """Compute some additional summaries, and convert tokens to text.

  Args:
    vocab: The vocabulary to detokenize generated text.
    met_summary: The summary object to process.
    mode: The mode of the summary (e.g. "test", "train")

  Returns:
    The modified summary dictionary.
  """

  mdict = met_summary.current_metric_dict()

  # Calculate perplexity from the average nats_per_token over all replicas.
  # This has to be done here, because the perplexities themselves can't be
  # averaged in the usual way.
  if "nats_per_token" in mdict:
    nats_per_token = mdict["nats_per_token"].to_value()
    met_summary.add({"perplexity": np.exp(nats_per_token)})

  if mode == "generate" and "gen_tokens" in mdict:
    # Convert output tokens to example output text.
    # Write text to both the summary, and pretty-print to the log file.
    gen_toks = mdict["gen_tokens"].to_value()
    if np.ndim(gen_toks) != 2:
      raise ValueError("Unsupported shape for gen_tokens: %s" % gen_toks.shape)

    ntoks = gen_toks.shape[-1]
    gen_text = text_dataset.decode_tokens(gen_toks, vocab, max_length=ntoks)
    logging.info("Generated text = %s", gen_text)
    met_summary.add_text({"gen_text": gen_text})
    del mdict["gen_tokens"]   # Otherwise it will turn into a histogram.

  return met_summary


@gin.configurable
def process_summaries_function(vocab: seqio.Vocabulary) -> Callable[
    [MetricsSummary, str], MetricsSummary]:
  """Return a function that processes summaries with the given vocabulary."""
  # For use with training_loop.process_summaries_function
  def process_fn(met_summary: MetricsSummary, mode: str):
    return process_summaries(vocab, met_summary, mode)
  return process_fn


@gin.configurable
class DecoderOnlyLanguageModel(nn.Module):
  """Decoder only language modeling."""

  mode: str
  task_config: decoder_stack.TransformerTaskConfig = gin.REQUIRED
  decoder_factory: Callable[[], Any] = gin.REQUIRED

  sample_method: str = "sample"   # Can be {"sample", "greedy"}
  output_token_losses: bool = False

  def get_fake_input(self):
    """Returns a fake input for initialization of the appropriate shape."""
    b = self.task_config.batch_size
    fake_input_dict = {
        "targets": jnp.ones([b, self.task_config.sequence_length],
                            dtype=jnp.int32),
        "start_of_sequence": jnp.ones([b], dtype=jnp.bool_),
        "epoch": jnp.ones([b], dtype=jnp.int32),
    }
    if text_dataset.get_loss_mask_tokens(split=self.mode) != (None, None):
      # We are not adding the loss mask to the dummy input by default as it can
      # cause a slowdown during evaluation and perhaps inference.
      fake_input_dict["loss_mask"] = jnp.ones(
          [b, self.task_config.sequence_length], dtype=jnp.bool_)
    return fake_input_dict

  def metrics_summary_operations(self, aggregate_over: str) -> Dict[str, str]:
    """Summary operation to use for recorded metrics."""
    metric_ops = {
        "loss": "mean",
        "nats_per_token": "mean",
        "bits_per_token": "mean",
        "bits_per_char": "mean",
        "accuracy": "mean",
        "num_tokens": "mean",
        "num_chars_per_device": "mean",
        "num_chars_per_batch": "mean",
        "nonzero_tokens": "mean",
        "num_tokens_per_device": "mean",
        "num_tokens_per_batch": "mean",
        "epoch": "mean",
    }
    if aggregate_over == "steps":
      return metric_ops
    elif aggregate_over == "devices":
      # Ensure that statistics that refer to the total batch size stay constant
      # as TPU topologies change. For those we have to sum over devices, but
      # compute the mean over steps.
      metric_ops.update({
          "num_tokens_per_batch": "sum",
          "num_chars_per_batch": "sum",
          "loss": "sum"})
      return metric_ops
    else:
      raise ValueError("Don't know how to aggregate over: %s" % aggregate_over)

  def setup(self):
    self.decoder = self.decoder_factory(mode=self.mode,
                                        task_config=self.task_config)  # pytype: disable=wrong-keyword-args  # trace-all-classes

  def __call__(self, inputs: ...):
    task_config = self.task_config

    input_tokens = inputs["targets"]                  # [b, seq_len]
    start_of_sequence = inputs["start_of_sequence"]   # [b]
    epochs = inputs["epoch"]                          # [b]
    if "loss_mask" in inputs:
      loss_mask = inputs["loss_mask"]                 # [b, seq_len]
    else:
      loss_mask = jnp.ones((1, 1), dtype=jnp.bool_)

    input_tokens = jnp.asarray(input_tokens)
    assert input_tokens.ndim == 2
    assert input_tokens.shape[0] == task_config.batch_size
    assert input_tokens.shape[1] == task_config.sequence_length
    assert start_of_sequence.shape[0] == task_config.batch_size

    # Sanity check to avoid out-of-bounds on token lookup.
    input_tokens = input_tokens % task_config.vocab_size

    logging.info("langmodel: Compiling model for mode %s", self.mode)
    logging.info("langmodel: input_tokens = %r", input_tokens)
    logging.info("langmodel: start_of_sequece = %r", start_of_sequence)
    logging.info("langmodel: epochs = %r", epochs)

    # The target outputs are the next character in each sequence.
    # Shift tokens left and pad with a zero at the end.
    # TODO(delesley): We don't predict the first token of each sequence.
    target_tokens = jnp.pad(input_tokens[:, 1:], [(0, 0), (0, 1)])
    logging.info("langmodel: target_tokens = %r", target_tokens)

    # Invoke the decoder stack.
    # The decoder will return pre-softmax logits for the predicted targets.
    (logits, _, d_metrics) = self.decoder(input_tokens=input_tokens,
                                          target_tokens=target_tokens,
                                          start_of_sequence=start_of_sequence)

    # Softmax cross-entropy loss on target tokens.
    logits = nn.log_softmax(logits, axis=-1)   # (b, seq_len, vocab_size)
    logging.info("langmodel: logits = %r", logits)
    soft_targets = common_utils.onehot(target_tokens, task_config.vocab_size)
    logging.info("langmodel: soft_targets = %r", soft_targets)

    losses = -jnp.sum(soft_targets * logits, axis=-1)  # (b, seq_len)
    logging.info("langmodel: losses = %r", losses)

    # Don't predict null tokens which are past the end-of-sequence.
    # Also don't predict the 0 at the end of the sequence.
    # TODO(delesley): Predict the final end-of-sequence marker.
    loss_mask = jnp.logical_and(
        loss_mask,
        input_tokens > 0)
    loss_mask = jnp.logical_and(
        loss_mask,
        target_tokens > 0)
    logging.info("langmodel: loss_mask = %r", loss_mask)

    losses = jnp.where(loss_mask, losses, 0.0)  # (batch_size, seq_len)
    loss = jnp.sum(losses)  # total loss on device

    token_count = jnp.sum(loss_mask)  # tokens on device
    token_count_nz = token_count + 1.0e-6
    loss_per_token = loss / token_count_nz
    bits_per_token = loss_per_token * 1.442695  # log(e)/log(2)
    accuracy = metric_utils.compute_accuracy_sum(logits, target_tokens,
                                                 loss_mask)
    accuracy = accuracy / token_count_nz  # Percent correct.
    epoch = jnp.mean(epochs)

    if self.mode == "generate" and self.decoder.supports_generate():
      # Generate example text.
      logging.info("lang_model: text inference.")
      gen_tokens = self.generate(inputs, task_config.sequence_length)

      # Return generated text, along with vizualizations and histograms.
      metrics = {"gen_tokens": gen_tokens, **d_metrics}
      return (loss, metrics)

    # Just return metrics related to the loss.
    metrics = {
        "loss": loss,   # will be summed over devices
        "nats_per_token": (loss_per_token, token_count),
        "bits_per_token": (bits_per_token, token_count),
        "accuracy": (accuracy, token_count),
        "num_tokens_per_device": token_count,
        "num_tokens_per_batch": token_count,  # will be summed over devices
        "epoch": epoch,
    }

    # Compute bits per character if we have the number of characters.
    if "num_chars" in inputs:
      num_chars = jnp.sum(inputs["num_chars"])
      bits_per_char = loss / (num_chars + 1e-6) * 1.442695
      metrics["num_chars_per_device"] = num_chars
      metrics["num_chars_per_batch"] = num_chars  # will be summed over devices
      metrics["bits_per_char"] = (bits_per_char, num_chars)

    # Provided to make sure that the data pipeline and the the model agree
    # on the number of tokens with a loss.
    if "nonzero_tokens" in inputs:
      nonzero_tokens = jnp.sum(inputs["nonzero_tokens"])
      metrics["nonzero_tokens"] = nonzero_tokens

    if self.output_token_losses:
      metrics["token_losses"] = losses

    return (loss, metrics)

  def generate(self, inputs: ..., sequence_length: int) -> Array:
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
    dstate = self.decoder.init_decoder_state(sequence_length,
                                             start_of_sequence)

    # TODO(delesley): Handle start-of-sequence in a better way.
    # There is no special token for start of sequence, so we grab the first
    # one from the ground-truth input data.
    first_token = input_tokens[:, 0:1]
    no_start_of_seq = jnp.array([False] * batch_size, dtype=jnp.bool_)
    sample_method = self.sample_method
    sample_prng = self.make_rng("sample")

    # Greedy autoregressive decoder function.
    def loop_fn(scan_state: Any, i: Array) -> Tuple[Any, Array]:
      prng = jax.random.fold_in(sample_prng, i)
      (dstate, input_token) = scan_state
      del i
      (logits, dstate, _) = self.decoder(input_tokens=input_token,
                                         target_tokens=None,
                                         start_of_sequence=no_start_of_seq,
                                         decoder_state=dstate)
      if sample_method == "sample":
        logging.info("Using categorical sampling.")
        output_token = jax.random.categorical(prng, logits, axis=-1)
      elif sample_method == "greedy":
        logging.info("Using greedy sampling.")
        output_token = jnp.argmax(logits, axis=-1)
      else:
        raise ValueError(f"Invalid sampling method: {sample_method}")
      logging.info("generate_loop_fn: output_token = %r", output_token)
      return ((dstate, output_token), output_token)

    # Scan over the sequence length.
    iterations = jnp.arange(sequence_length)
    initial_scan_state = (dstate, first_token)
    (_, output_tokens) = jax.lax.scan(loop_fn, initial_scan_state, iterations)
    logging.info("generate: output_tokens = %r", output_tokens)

    # Output_tokens has shape (sequence_length, batch_size, 1)
    assert output_tokens.shape == (sequence_length, batch_size, 1)
    output_tokens = jnp.reshape(
        output_tokens, (sequence_length, self.task_config.batch_size))
    output_tokens = output_tokens.transpose([1, 0])
    return output_tokens
