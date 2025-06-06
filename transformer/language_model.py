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

"""Utility functions used by many language models."""

from typing import Any, Callable, Dict, Optional, Tuple

from absl import logging
from flax import linen as nn
from flax import struct
from flax.training import common_utils
import gin
import jax
import jax.numpy as jnp
import  metrics_summary
import  model_base
from transformer import nn_components
from transformer import text_dataset
import numpy as np
import seqio


Array = model_base.Array
MetricsSummary = metrics_summary.MetricsSummary
Metrics = model_base.Metrics


vshape = nn_components.vshape
average_metric = metrics_summary.average_metric
average_of_sum_metric = metrics_summary.average_of_sum_metric
scalar_metric = metrics_summary.scalar_metric
text_metric = metrics_summary.text_metric


BITS_PER_TOKEN_FACTOR = 1.442695  # log(e)/log(2)


# Basic gin-configurable task configuration, used in many places.
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


def targets_from_input_tokens(input_tokens: Array,
                              loss_mask: Array) -> Tuple[Array, Array]:
  """Shift tokens left by 1 to get target_tokens, for autoregressive decoding.

  Args:
    input_tokens: Array of int of shape (batch_size, sequence_length)
    loss_mask: Optional array of bool of shape (batch_size, sequence_length)

  Returns:
    target_tokens, loss_mask, with the same shape as input_tokens, loss_mask.
  """
  assert input_tokens.shape == loss_mask.shape

  # The target outputs are the next character in each sequence.
  # Shift tokens left and pad with a zero at the end.
  # TODO(delesley): We don't predict the first token of each sequence.
  target_tokens = jnp.pad(input_tokens[:, 1:], [(0, 0), (0, 1)])
  loss_mask = jnp.pad(loss_mask[:, 1:], [(0, 0), (0, 1)])
  return (target_tokens, loss_mask)


def add_padding_to_loss_mask(loss_mask: Array,
                             input_tokens: Array,
                             target_tokens: Array) -> Array:
  """Mask out zero tokens that are part of padding."""

  # Don't predict null tokens which are past the end-of-sequence.
  # Also don't predict the 0 at the end of the sequence.
  # TODO(delesley): Predict the final end-of-sequence marker.
  loss_mask = jnp.logical_and(loss_mask, input_tokens > 0)
  loss_mask = jnp.logical_and(loss_mask, target_tokens > 0)
  return loss_mask


def softmax_cross_entropy_loss(logits: Array,
                               target_tokens: Array) -> Tuple[Array, Array]:
  """Return the softmax cross-entropy loss.

  Args:
    logits: Array of shape(batch_size, sequence_len, vocab_size).
        The pre-softmax raw scores output by the NN.
    target_tokens: Array of ints of shape(batch_size, sequence_len)

  Returns:
    Per-token losses of shape (batch_size, sequence_len),
    Per-token entropy of shape (batch_size, sequence_len)
  """

  (batch_size, seq_len, vocab_size) = logits.shape
  assert target_tokens.shape == (batch_size, seq_len)

  # Softmax cross-entropy loss on target tokens.
  log_probs = nn.log_softmax(logits, axis=-1)   # (b, seq_len, vocab_size)
  logging.info("cross_entropy: log_probs = %s", vshape(log_probs))
  soft_targets = common_utils.onehot(target_tokens, vocab_size)
  logging.info("cross_entropy: soft_targets = %s", vshape(soft_targets))

  # The loss is the negative log probability of predicting the ground-truth.
  losses = -jnp.sum(soft_targets * log_probs, axis=-1)  # (b, seq_len)
  logging.info("cross_entropy: losses = %s", vshape(losses))

  # Compute the entropy (uncertainty) of the prediction for each character.
  # Entropy is the expected negative log-probability.
  probs = jnp.exp(log_probs)
  entropy = -jnp.sum(probs * log_probs, axis=-1)
  logging.info("cross_entropy: entropy = %s", vshape(entropy))

  return (losses, entropy)


def sigmoid_cross_entropy(logits: Array, labels: Array) -> Array:
  """Computes element-wise sigmoid cross entropy given logits and labels.

  Args:
    logits: Pre-sigmoid predictions, of shape s.
            sigmoid(logits) is the probability that the value is 1.0.
    labels: Expected value (either 1 or 0) of shape s.

  Returns:
    Per-element cross-entropy loss of shape s.
  """
  if logits.shape != labels.shape:
    raise ValueError(f"Shape mismatch {logits.shape} != {labels.shape}.")

  # log_p and log_not_p together are the same as a 2-element log_softmax.
  # labels and (1 - labels) together are the same as a one_hot.
  labels = labels.astype(logits.dtype)
  log_p = jax.nn.log_sigmoid(logits)
  logging.info("sigmoid_cross_entropy: log_probs = %s", vshape(log_p))
  logging.info("sigmoid_cross_entropy: labels = %s", vshape(labels))
  # log(1 - sigmoid(x)) = log_sigmoid(-x), the latter more numerically stable.
  log_not_p = jax.nn.log_sigmoid(-logits)
  losses = -(labels * log_p + (1. - labels) * log_not_p)
  logging.info("sigmoid_cross_entropy: losses = %s", vshape(losses))
  return losses


def per_token_accuracy(logits: Array, targets: Array) -> Array:
  """Compute per-token accuracy, given logits.

  The output of this function can be passed to metrics_summary.average_metric()
  (along with the loss mask) to report the actual accuracy.

  Args:
   logits: Array of shape [batch_size, seq_length, vocab_size].
   targets: Integer array of shape [batch_size, seq_length]

  Returns:
    Boolean array of size (batch, sequence_length), which is True for each
    token that is correct, and False for each token that is incorrect.
  """

  (batch_size, seq_len, _) = logits.shape
  if targets.shape != (batch_size, seq_len):
    raise ValueError(f"Incorrect shapes. Got logits: {logits.shape}"
                     f" and targets: {targets.shape}")
  return jnp.equal(jnp.argmax(logits, axis=-1), targets)


def compute_token_metrics(inputs: Dict[str, Any],
                          token_losses: Array,
                          loss_mask: Array,
                          logits: Array,
                          target_tokens: Array) -> Metrics:
  """Compute various metrics like accuracy and bits-per-token.

  Args:
    inputs: Dictionary of inputs to the language model.
        If present "epochs" is extracted as a metric.
    token_losses: Array of shape (batch_size, sequence_length), which provides
        the loss for each token.
    loss_mask: Float or bool array of shape (batch_size, sequence_length),
        which provides the weight for each token -- 1.0/True, or 0.0/False.
    logits: Array of shape (batch_size, sequence_length, vocab_size), which
        has the predicted token probabilities.
    target_tokens: Integer array of shape (batch_size, sequence_length)
        with the expected tokens.  Used with logits to compute accuracy.

  Returns:
    A dictionary of metrics.
  """

  if "epoch" in inputs:
    epochs = inputs["epoch"]   # shape (batch_size,)
  else:
    epochs = None

  (batch_size, seq_len, _) = logits.shape
  assert token_losses.shape == (batch_size, seq_len)
  assert loss_mask.shape == (batch_size, seq_len)

  # Compute relevant metrics.
  loss_mask = loss_mask.astype(jnp.float32)
  bits_per_token = token_losses * BITS_PER_TOKEN_FACTOR
  pt_accuracy = per_token_accuracy(logits, target_tokens)
  token_count = jnp.sum(loss_mask, axis=-1)  # tokens on device

  metrics = {
      "nats_per_token": average_metric(token_losses, loss_mask),
      "bits_per_token": average_metric(bits_per_token, loss_mask),
      "accuracy": average_metric(pt_accuracy, loss_mask),
      "num_tokens_per_example": average_metric(token_count),
      "num_tokens_per_batch": average_of_sum_metric(token_count),
  }
  if epochs is not None:
    metrics["epoch"] = average_metric(epochs)

  # Compute bits per character if we have the number of characters.
  # "num_chars" should hold the number of characters, before tokenization.
  if "num_chars" in inputs:
    num_chars = inputs["num_chars"]
    metrics["num_chars_per_example"] = average_metric(num_chars)
    metrics["num_chars_per_batch"] = average_of_sum_metric(num_chars)

    assert num_chars.shape == (batch_size,)
    bits_per_char = jnp.sum(bits_per_token, axis=-1) / (num_chars + 1e-6)
    metrics["bits_per_char"] = average_metric(bits_per_char, num_chars)

  # Provided to make sure that the data pipeline and the the model agree
  # on the number of tokens with a loss.
  if "nonzero_tokens" in inputs:
    nonzero_tokens = inputs["nonzero_tokens"]
    metrics["nonzero_tokens"] = average_metric(nonzero_tokens)

  # Compute the average word length if token boundaries have been provided.
  if "token_boundaries" in inputs:
    token_boundaries = inputs["token_boundaries"]
    assert token_boundaries.shape == (batch_size, seq_len)
    num_token_boundaries = jnp.sum(token_boundaries, axis=-1)
    avg_word_length = token_count / (num_token_boundaries + 1.0e-6)
    metrics["word_length_avg"] = average_metric(avg_word_length)

  return metrics


def compute_binary_metrics(predictions: Array,
                           labels: Array,
                           mask: Array) -> Metrics:
  """Computes accuracy, precision, recall, etc.

  Args:
    predictions: Bool array of True/False predictions.
    labels: Bool array of the accurate True/False predictions.
    mask: Array whose True values represent entries that *should* be considered
      when computing metrics like accuracy. (False entries are "masked out",
      ignored for both numerator and denominator.)

  Returns:
    A dictionary mapping metric names (accuracy, precision, recall,
    positive_prediction_fraction, positive_fraction) to tuples of (metric,
    weight).
  """
  assert predictions.dtype == jnp.bool_
  assert labels.dtype == jnp.bool_
  assert mask.dtype == jnp.bool_

  assert predictions.ndim == 2  # (batch_size, sequence_length)
  assert predictions.shape == labels.shape
  assert predictions.shape == mask.shape

  # All of these should sum along the sequence length dimension.
  true_positives = jnp.sum(predictions & labels & mask,
                           axis=1)
  false_positives = jnp.sum(predictions & jnp.logical_not(labels) & mask,
                            axis=1)
  true_negatives = jnp.sum(
      jnp.logical_not(predictions) & jnp.logical_not(labels) & mask,
      axis=1
  )
  false_negatives = jnp.sum(jnp.logical_not(predictions) & labels & mask,
                            axis=1)
  total = jnp.sum(mask, axis=1)
  epsilon = 1e-6
  accuracy = (true_positives + true_negatives) / (total + epsilon)
  precision = true_positives / (true_positives + false_positives + epsilon)
  recall = true_positives / (true_positives + false_negatives + epsilon)
  positive_prediction_fraction = (true_positives + false_positives) / (
      total + epsilon
  )
  positive_fraction = (true_positives + false_negatives) / (total + epsilon)

  return {
      "accuracy": average_metric(accuracy, total),
      "precision": average_metric(precision, total),
      "recall": average_metric(recall, total),
      "positive_prediction_fraction": average_metric(
          positive_prediction_fraction, total),
      "positive_fraction": average_metric(positive_fraction, total),
  }


def compute_bpt_metrics(token_losses: Array,
                        loss_mask: Array,
                        postfix: str = "") -> Metrics:
  """Compute bits_per_token metrics.

  Args:
    token_losses: per-token lossses of shape (batch_size, sequence_length)
    loss_mask: float or bool array of shape (batch_size, sequence_length)
    postfix: An optional postfix to the metric name.

  Returns:
    A dictionary with the metric { ("bits_per_token" + postfix): bpt }
  """
  if token_losses.shape != loss_mask.shape:
    raise ValueError(f"Shape mismatch {token_losses.shape} != "
                     f"{loss_mask.shape}.")

  bits_per_token = token_losses * BITS_PER_TOKEN_FACTOR
  return {
      ("bits_per_token" + postfix): average_metric(bits_per_token, loss_mask),
  }


# TODO(delesley): Move this functionality into the compute() method of
# custom metrics.
def process_summaries(vocab: seqio.Vocabulary,
                      met_summary: MetricsSummary,
                      mode: str) -> MetricsSummary:
  """Compute some additional summaries, and convert tokens to text.

  This function is called immedately before writing each MetricsSummary.

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
  # TODO(delesley): Define a dedicated "perplexity" metric to fix this.
  if "nats_per_token" in mdict:
    nats_per_token = mdict["nats_per_token"].compute().value
    met_summary.merge({
        "perplexity": scalar_metric(np.exp(nats_per_token))
    })

  if mode == "generate" and "gen_tokens" in mdict:
    # Convert output tokens to example output text.
    # Write text to both the summary, and pretty-print to the log file.
    gen_toks_m = mdict["gen_tokens"]
    if not isinstance(gen_toks_m, metrics_summary.OutputValue):
      raise ValueError("gen_tokens must be an instance of OutputValue.")
    gen_toks = gen_toks_m.value
    if np.ndim(gen_toks) != 2:
      raise ValueError("Unsupported shape for gen_tokens: %s" % gen_toks.shape)

    ntoks = gen_toks.shape[-1]
    gen_text = text_dataset.pretty_print_tokens(gen_toks, vocab,
                                                max_length=ntoks)
    logging.info("Generated text = %s", gen_text)
    met_summary.merge({
        "gen_text": text_metric(gen_text)
    })
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


def _construct_pad_widths(
    *, ndims: int, axis: int, num_before: int = 0, num_after: int = 0
) -> np.typing.ArrayLike:
  pad_widths = np.zeros(shape=(ndims, 2), dtype=np.int32)
  pad_widths[axis, 0] = num_before
  pad_widths[axis, 1] = num_after
  return pad_widths


def shift_right(a: Array,
                /,
                *,
                axis: int = -1,
                shift_by: int = 1,
                padding_constant: Any = 0,
                prepend_array: Optional[Array] = None) -> Tuple[Array, Array]:
  """Shifts values along axis.  Similar to np.roll, but pads with zero.

  The result for a 1D array will be ([0] * shift_by) + a[:-shift_by]

  Args:
    a: The array to shift, of shape [..., len, ...]
    axis: The axis to shift along.
    shift_by: The number of indices to shift.
      The values in 'a' will be shifted over by the given amount.
    padding_constant: The scalar constant to pad with.
    prepend_array: Array of shape [..., shift_by, ...]
      If specified, instead prepending zeros to padd the result,
      prepend_array will be prepended instead.
      E.g. concat(prepend_array, a[:-shift_by])

  Returns:
    (A shifted and padded version of a,
     An array of the values that were shifted off of the end of a.)
  """
  assert shift_by >= 0

  ashape = list(a.shape)
  andim = len(ashape)
  if axis < 0:
    axis = andim + axis
  a_length = ashape[axis]
  assert axis >= 0 and axis < andim

  # Slice for shifted values.
  shift_slices = tuple(
      [np.s_[:] for _ in range(axis)]               # slice for :
      + [np.s_[:-shift_by]]                         # slice for :-shift_by
      + [np.s_[:] for _ in range(axis + 1, andim)]  # slice for :
  )
  # Slice for values that are shifted off of the end.
  end_slices = tuple(
      [np.s_[:] for _ in range(axis)]               # slice for :
      + [np.s_[a_length - shift_by:]]
      + [np.s_[:] for _ in range(axis + 1, andim)]  # slice for :
  )
  sliced_a = a[shift_slices]
  end_slice = a[end_slices]

  if prepend_array is not None:
    # Ensures that the input and output arrays have the same shape.
    # (E.g. the shape of prepend_array should be the same as the shape of
    # the padding that we would otherwise use if prepend_array were None.)
    assert prepend_array.shape[axis] == shift_by
    shifted_result = jnp.concatenate([prepend_array, sliced_a], axis=axis)
  else:
    pad_widths = _construct_pad_widths(
        ndims=andim, axis=axis, num_before=shift_by
    )
    shifted_result = jnp.pad(sliced_a, pad_widths,
                             constant_values=padding_constant)
  assert shifted_result.shape == a.shape
  return (shifted_result, end_slice)


def shift_left(a: Array,
               /,
               *,
               axis: int = -1,
               shift_by: int = 1,
               padding_constant: Any = 0,
               append_array: Optional[Array] = None) -> Tuple[Array, Array]:
  """Shifts values along axis.  Like np.roll, but pads with zero.

  The result for a 1D array will be ([0] * shift_by) + a[:-shift_by]

  Args:
    a: The array to shift, of shape [..., len, ...]
    axis: The axis to shift along.
    shift_by: The number of indices to shift each value over by.
      The values in 'a' will be shifted over by the given amount.
    padding_constant: The scalar constant to pad with.
    append_array: Array of shape [..., shift_by, ...]
      If specified, instead appending zeros to padd the result,
      append_array will be appended instead.
      E.g. concat(a[:-shift_by], append_array)

  Returns:
    (A shifted and padded version of a,
     An array of the values that were shifted off of the end of a.)
  """
  assert shift_by >= 0

  ashape = list(a.shape)
  andim = len(ashape)
  if axis < 0:
    axis = andim + axis
  assert axis >= 0 and axis < andim

  # Slice for shifted values.
  shift_slices = tuple(
      [np.s_[:] for _ in range(axis)]               # slice for :
      + [np.s_[shift_by:]]                          # slice for shift_by:
      + [np.s_[:] for _ in range(axis + 1, andim)]  # slice for :
  )
  # Slice for values that are shifted off of the end.
  end_slices = tuple(
      [np.s_[:] for _ in range(axis)]               # slice for :
      + [np.s_[:shift_by]]
      + [np.s_[:] for _ in range(axis + 1, andim)]  # slice for :
  )
  sliced_a = a[shift_slices]
  end_slice = a[end_slices]

  if append_array is not None:
    # Ensures that the input and output arrays have the same shape.
    # (E.g. the shape of prepend_array should be the same as the shape of
    # the padding that we would otherwise use if prepend_array were None.)
    assert append_array.shape[axis] == shift_by
    shifted_result = jnp.concatenate([sliced_a, append_array], axis=axis)
  else:
    pad_widths = _construct_pad_widths(
        ndims=andim, axis=axis, num_after=shift_by
    )
    shifted_result = jnp.pad(sliced_a, pad_widths,
                             constant_values=padding_constant)
  assert shifted_result.shape == a.shape
  return (shifted_result, end_slice)

