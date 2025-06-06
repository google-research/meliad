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

"""Load text datasets for long-range transformer models."""

import math
import os
import re
import time
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional,
                    Sequence, Set, Tuple, Union)

from absl import flags
from absl import logging
from flax import struct
import gin
import jax
from jax.experimental import multihost_utils
from transformer import synthetic_text_data
import numpy as np
import seqio
import tensorflow.compat.v2 as tf



flags.DEFINE_string("default_data_dir", None,
                    "Default directory where data is stored.")
FLAGS = flags.FLAGS


_DEFAULT_DATA_DIRECTORY = None


@gin.configurable
def set_default_data_directory(directory_name=None):
  """Set the default directory where training data is located."""
  global _DEFAULT_DATA_DIRECTORY
  # If the data directory has been overridden with a command-line flag, use it.
  # If not, the see if directory_name has been configured by Gin.
  # Otherwise, use the default tfds directory.
  if FLAGS.default_data_dir:
    directory_name = FLAGS.default_data_dir
  if directory_name is not None:
    seqio.set_tfds_data_dir_override(directory_name)
    _DEFAULT_DATA_DIRECTORY = directory_name


def get_iterator_function(dataset: Optional[tf.data.Dataset]):
  """Returns a function which gets an iterator over the given dataset."""
  if dataset is None:
    return None
  else:
    return dataset.as_numpy_iterator


@gin.configurable
@struct.dataclass
class DataPipelineConfig:
  """Gin-configurable options for the data pipeline."""

  # Add a lost mask to the token stream.
  loss_mask_fn: Optional[Callable[[np.ndarray, str], np.ndarray]] = None

  # Tokenize and detokenize the input to get token boundaries.
  compute_token_boundaries: bool = False

  # Simple whitespace-based tokenization, suitable for English.
  compute_simple_token_boundaries_before_space: bool = False
  compute_simple_token_boundaries_after_space: bool = False

  # Remove whitespace from the input string. A character c is considered
  # whitespace if the Python builtin `c.isspace()` returns `True`.
  # TODO(delesley):  Currently works only with datasets that are ASCII text.
  remove_whitespace: bool = False

  # Tokenize and detokenize the input to normalize it.
  decode_tokens_to_string: bool = False

  # Size of character set and vocabulary,
  # when using compute_token_boundaries or remove_whitespace.
  character_set_size: int = 256


def data_pipeline_uses_token_boundaries():
  """Returns true if the pipeline is configured to produce token boundaries."""
  data_pipe_config = DataPipelineConfig()
  return (data_pipe_config.compute_token_boundaries or
          data_pipe_config.compute_simple_token_boundaries_before_space or
          data_pipe_config.compute_simple_token_boundaries_after_space)


def load_text_dataset_vocabulary(name: str) -> seqio.Vocabulary:
  """Load the vocabulary object for a given text dataset."""

  # compute_token_boundaries replaces the original task vocab.
  pipeline_config = DataPipelineConfig()
  charset_size = pipeline_config.character_set_size
  if (pipeline_config.compute_token_boundaries or
      pipeline_config.compute_simple_token_boundaries_before_space or
      pipeline_config.compute_simple_token_boundaries_after_space):
    return seqio.PassThroughVocabulary(charset_size, 0)
  elif pipeline_config.remove_whitespace:
    return seqio.PassThroughVocabulary(charset_size, 0)

  if name == "synthetic":
    return seqio.PassThroughVocabulary(256, 0)

  # Bypass the seqio "feature converter", and get the task directly.
  task = seqio.get_mixture_or_task(name)
  vocab = task.output_features["targets"].vocabulary
  return vocab


@gin.configurable
def load_text_dataset(name: str,
                      split: str,
                      sequence_length: int,
                      batch_size: int,
                      sequential: bool = True,
                      num_shards: int = 1,
                      shard_id: int = 0,
                      verbose: bool = False,
                      random_seed: Optional[int] = None,
                      shuffle_splits: Sequence[str] = ("train",)
                      ) -> Tuple[tf.data.Dataset, seqio.Vocabulary]:
  """Load a text dataset of long articles or books, and split_and_batch them.

  The input dataset must produce complete books or articles, where each article
  is a dictionary containing a "tokens" field.
  See split_and_batch for more information on the output dataset.

  Args:
    name:  The name of the seqio task which produces the dataset.
    split: The name of the split to use, e.g. "train" or "test".
    sequence_length: Split text into sequences of this length.
    batch_size: Draw from batch_size articles in each batch.
    sequential: If True, return the chunks of each article in sequence.
    num_shards: Number of shards to divide the dataset into; 1 for no sharding.
    shard_id: The shard# for this process; 0 <= shard_id < num_shards.
    verbose: Log (an excerpt) of every text example loaded from disk. If False,
      will only print 1 excerpt every 60 seconds.
    random_seed: The random seed to use for shuffling. Uses current system
      time as the random seed if None.
    shuffle_splits: List of splits; e.g. ("train", "test") to shuffle.

  Returns:
    (dataset, vocabulary)
    where vocabulary is the seqio.Vocabulary which is used to encode "targets".
  """

  logging.info("Loading text data set %s, split=%s, shape=(%d, %d)",
               name, split, batch_size, sequence_length)

  if random_seed is None:
    # All devices must use the same random seed.
    # With model parallelism, a single model may span multiple hosts.
    # All hosts with the same shard_id must return the same input sequence.
    random_seed = multihost_utils.broadcast_one_to_all(np.int32(time.time()))
  random_seed = int(random_seed)
  logging.info("Using %d as random seed for input pipeline.", random_seed)

  if name == "synthetic":
    # Synthetic data set for hermetic testing purposes.
    ds = synthetic_data(split)
    vocab = seqio.PassThroughVocabulary(256, 0)
  else:
    # Load the data set from a task defined in tasks.py.
    # Bypass the seqio "feature converter", and get the task directly.
    task = seqio.get_mixture_or_task(name)
    vocab = task.output_features["targets"].vocabulary

    # Create the task input pipeline.
    assert shard_id < num_shards
    if num_shards > 1:
      logging.info("Shards: %d of %d", shard_id, num_shards)
      shard_info = seqio.ShardInfo(index=shard_id, num_shards=num_shards)
    else:
      shard_info = None

    # Process long documents sequentially, or shuffle segments.
    if sequential:
      task_seqlen = None             # We do our own splitting.
      shuffle_buffer_size = 1000     # Number of full-length books.
    else:
      # TODO(delesley): Asking the task may simply result in truncation.
      task_seqlen = {"targets": sequence_length}  # Ask the task to split.
      shuffle_buffer_size = 10_000   # Number of segments.

    # Choose whether or not to shuffle; often the test set is not shuffled.
    if split in shuffle_splits:
      shuffle = True
    else:
      shuffle = False
      shuffle_buffer_size = None

    # Grab the dataset, as defined in tasks.py.
    ds = task.get_dataset(
        sequence_length=task_seqlen,
        split=split,
        use_cached=False,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=random_seed,
        shard_info=shard_info,
        num_epochs=1)

  if sequence_length == 0:
    return (ds, vocab)  # Don't chop into subsequences.

  # Process configuration options for the data pipeline.
  pipeline_config = DataPipelineConfig()
  splittable_keys = []
  output_keys = ["nonzero_tokens"]
  if vocab is not None:
    output_keys.append("num_chars")

  # Configure a loss mask.  See loss_mask_from_tokens() for an example.
  if pipeline_config.loss_mask_fn is not None:
    def loss_mask_fn(tokens: np.ndarray) -> np.ndarray:
      return pipeline_config.loss_mask_fn(tokens, split)

    splittable_keys.append("loss_mask")
    output_keys.append("loss_mask")
  else:
    loss_mask_fn = None

  # Configure token boundaries.
  orig_vocab = vocab  # Used for preprocessing
  charset_size = pipeline_config.character_set_size
  if (pipeline_config.compute_token_boundaries or
      pipeline_config.compute_simple_token_boundaries_before_space or
      pipeline_config.compute_simple_token_boundaries_after_space):
    logging.info("load_text_dataset: Compute token boundaries selected.")
    splittable_keys.append("token_boundaries")
    output_keys.append("token_boundaries")
    # Preprocessing will change the tokens to straight unicode values.
    vocab = seqio.PassThroughVocabulary(charset_size, 0)
  elif pipeline_config.remove_whitespace:
    # To remove whitespace, we must decode to string and use a byte vocab.
    vocab = seqio.PassThroughVocabulary(charset_size, 0)

  def preprocess_fn(article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return preprocess_article(
        article,
        loss_mask_fn=loss_mask_fn,
        vocab=orig_vocab,
        pipeline_config=pipeline_config,
        verbose=verbose,
    )

  def postprocess_fn(segment: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return postprocess_segment(segment, vocab=vocab)

  def split_article_fn(article: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    return split_article(article,
                         segment_length=sequence_length,
                         splittable_keys=splittable_keys,
                         preprocess_fn=preprocess_fn,
                         postprocess_fn=postprocess_fn)

  ds = split_and_batch(ds,
                       split_article_fn=split_article_fn,
                       segment_length=sequence_length,
                       batch_size=batch_size,
                       output_keys=output_keys,
                       auto_rewind=True)
  return (ds, vocab)


def preprocess_article(
    article: Dict[str, Any],
    loss_mask_fn: Optional[Callable[[Any], np.ndarray]] = None,
    vocab: Optional[seqio.Vocabulary] = None,
    pipeline_config: DataPipelineConfig = DataPipelineConfig(),
    verbose: bool = False
) -> Optional[Dict[str, Any]]:
  """Do additional preprocessing on each article, before splitting.

  Current preprocessing options include adding a loss mask, and computing
  token boundaries.

  Args:
    article: An article, pulled from the seqio data pipeline.
        It must have a "targets" field.
    loss_mask_fn: Can be used to mask out the loss on certain tokens.
    vocab: A seqio Vocabulary to use for decoding.
    pipeline_config: Specifies various preprocessing options.
    verbose: If true, then do verbose logging.

  Returns:
    A processed article.
  """

  compute_token_boundaries = pipeline_config.compute_token_boundaries
  stokens_before = pipeline_config.compute_simple_token_boundaries_before_space
  stokens_after = pipeline_config.compute_simple_token_boundaries_after_space
  decode_tokens_to_string = pipeline_config.decode_tokens_to_string
  remove_whitespace = pipeline_config.remove_whitespace

  if verbose:
    logging.info("Reading article: %s", pretty_print_article(article, {}))
  else:
    logging.log_every_n_seconds(logging.INFO, "Reading article: %s", 60,
                                pretty_print_article(article, {}))

  tokens = article["targets"]

  # Make sure that targets is an array of integers.
  if isinstance(tokens, str) or isinstance(tokens, bytes):
    tokens = _string_to_tokens(tokens)
  elif isinstance(tokens, np.ndarray):
    tokens = tokens.astype(np.int32)
  else:
    raise ValueError(f"Unsupported value for tokens: {tokens}")

  if remove_whitespace:
    # We have to decode the tokens to a string to remove whitespace.
    # We re-encode using a byte-level pass-through vocabulary.
    assert vocab is not None
    dstr = decode_tokens_1d(tokens, vocab)
    tokens = np.array([ord(c) for c in dstr if not c.isspace()],
                      dtype=np.int32)
  elif stokens_before or stokens_after:
    assert vocab is not None
    dstr = decode_tokens_1d(tokens, vocab)
    tokens = np.array([ord(c) for c in dstr], dtype=np.int32)
    if stokens_before:
      tbounds = _open_vocab_segment_before_space(dstr)
    else:
      tbounds = _open_vocab_segment_after_space(dstr)
    article["targets_decoded"] = dstr
    article["token_boundaries"] = np.array(tbounds, dtype=np.float32)
    article["targets_original"] = article["targets"]

  assert np.ndim(tokens) == 1
  article["targets"] = tokens

  # Possibly use the vocab to decode tokens back to a string, and use the
  # string to train a character-level model.
  # TODO(delesley): As the number of options increases, it may make sense to
  # split up this function.
  if compute_token_boundaries or decode_tokens_to_string:
    assert vocab is not None
    if compute_token_boundaries:
      # Convert tokens to a list of integers to make vocab.decode happy.
      dtoks = [int(i) for i in tokens]
      tokens = dtoks
      (dstr, tbounds) = find_token_boundaries_1d(tokens, vocab)
      article["targets_decoded"] = dstr
      article["token_boundaries"] = np.array(tbounds, dtype=np.float32)
    elif decode_tokens_to_string:
      dstr = decode_tokens_1d(tokens, vocab)
      article["targets_decoded"] = dstr

    # Convert the decoded string back to a sequence of characters, which
    # can be used to train a character-level model.
    article["targets_original"] = article["targets"]
    article["targets"] = _string_to_tokens(article["targets_decoded"])

  # Generate a loss mask for tasks which require it.
  if loss_mask_fn is not None:
    try:
      loss_mask = loss_mask_fn(tokens)
      article["loss_mask"] = loss_mask
    except ValueError:
      logging.exception("Error while computing loss mask; dropping sequence.")
      return None

  return article


def _string_to_tokens(s: Union[str, bytes]):
  """Convert a byte string to an array of integers."""
  if isinstance(s, str):
    return np.array([ord(c) for c in s], dtype=np.int32)
  elif isinstance(s, bytes):
    return np.array([i for i in s], dtype=np.int32)
  else:
    raise TypeError(f"Invalid type {type(s)}")


def postprocess_segment(segment: Dict[str, np.ndarray],
                        vocab: Optional[seqio.Vocabulary] = None
                       ) -> Dict[str, np.ndarray]:
  """Do some additional postprocessing on each segment.

  This function will count the number of characters and nonzero tokens, for
  the purpose of reporting bits-per-character numbers.

  Args:
    segment: The segment, as returned from split_article.
    vocab: A seqio Vocabulary, to decode the tokens.

  Returns:
    A segment with additional information, which has stackable arrays.
  """

  if "loss_mask" in segment:
    lmask = segment["loss_mask"]
  else:
    lmask = None
  toks = _nonzero_tokens(segment["targets"], lmask)
  segment["nonzero_tokens"] = np.array(len(toks), dtype=np.int32)

  if vocab is not None:
    bchars = decode_tokens_1d(toks, vocab)
    segment["num_chars"] = np.array(len(bchars), dtype=np.int32)

  return segment


def _nonzero_tokens(tokens: np.ndarray,
                    loss_mask: Optional[np.ndarray]) -> list[int]:
  """Removes tokens that are not predicted by the model."""
  # TODO(delesley): Fix the model so that it predicts the first token.
  # The language model doesn't predict the first token.
  toks = [int(tokens[i]) for i in range(1, len(tokens))
          if (tokens[i] != 0 and (loss_mask is None or loss_mask[i]))]
  return toks


def split_article(
    article: Dict[str, Any],
    segment_length: int,
    splittable_keys: List[str],
    preprocess_fn: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
    postprocess_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Iterable[Dict[str, np.ndarray]]:
  """Split an article into segments of length sequence_length.

  Args:
    article: An article, as returned from the dataset iterator.
    segment_length: The length of segments to split the article into.
    splittable_keys: Keys within the article that should be split.
        All articles must have "targets", but may include other keys,
        e.g. "loss_mask", which are declared in splittable_keys.
    preprocess_fn: A function to preprocess each article.
    postprocess_fn: A function to postprocess each segment after splitting.

  Yields:
    A segment, which is a dictionary of { key: value }
  """

  # Preprocess the article.
  # After preprocessing, the article should have at least a "targets" field.
  article = preprocess_fn(article)
  if article is None:
    logging.info("Empty article.")
    return  # Empty sequence

  tokens = article["targets"]
  assert np.ndim(tokens) == 1

  # This function will split the "targets" field into multiple segments.
  # Additional fields (e.g. "loss_mask") may also be split.
  skeys = ["targets"]
  for k in splittable_keys:
    skeys.append(k)

  # Check that splittable_keys have the same length as "targets".
  # We warn instead of failing to prevent training from blownin up on a single
  # bad example.
  for k in splittable_keys:
    if len(article[k]) != len(tokens):
      logging.warning("length of %s (%d) does not match length of 'targets'"
                      " (%d) in article", k, len(article[k]), len(tokens))
      return  # Skip article

  # Split all keys into segments.
  for i in range(0, len(tokens), segment_length):
    segment = {}
    for k in skeys:
      segment[k] = _pad_segment(article[k][i:i + segment_length],
                                segment_length)
    segment = postprocess_fn(segment)
    yield segment


def _pad_segment(s: Optional[np.ndarray],
                 segment_length: int,
                 dtype=np.int32) -> np.ndarray:
  """Pad an array s out to the given sequence_length."""
  if s is None:
    return np.zeros(segment_length, dtype=dtype)
  assert np.ndim(s) == 1
  seg_len = len(s)
  assert seg_len <= segment_length
  if seg_len == segment_length:
    return s
  else:
    return np.pad(s, (0, segment_length - seg_len),
                  mode="constant", constant_values=0)


def rekey_articles(ds: tf.data.Dataset,
                   rekey: Mapping[str, str],
                   keep: Optional[Set[str]] = None) -> tf.data.Dataset:
  """Rekey the articles in ds.  Used in tasks.py.

  Fields in rekey will be renamed, field in keep will be kept, others will
  be discarded.  E.g., For PG19:

    rekey_article(ds,
                  rekey={"book_text": "targets"},
                  keep={"book_title", "book_id"})
  Args:
    ds: The dataset to rekey.
    rekey: Dictionary which contains fields to rename.
    keep: Set of fields to keep.

  Returns:
    A rekeyed dataset.
  """

  def rekey_fn(article):
    result_dict = {}
    for (k, v) in article.items():
      if k in rekey:
        result_dict[rekey[k]] = v
      elif k in keep:
        result_dict[k] = v
    return result_dict

  return ds.map(rekey_fn)


def pretty_print_article(article,
                         vocab_map: Mapping[str, Optional[seqio.Vocabulary]],
                         max_length: int = 60) -> str:
  """Convert the contents of a long article to a short string."""
  if not hasattr(article, "items"):
    return pretty_print_value(article, max_length)  # Not a dictionary.
  dstr = "{"
  for (k, v) in article.items():
    if vocab_map and k in vocab_map:
      vstr = pretty_print_tokens(v, vocab_map[k], max_length)
    else:
      vstr = pretty_print_value(v, max_length)
    dstr += "\n  " + k + ": " + vstr
  return dstr + "\n}"


def pretty_print_value(value, max_length: int) -> str:
  """Convert a possibly large value to a short string."""
  if isinstance(value, bytes):
    if len(value) <= max_length:
      return str(value)
    else:
      return f"bytes[{len(value)}] " + str(value[:max_length]) + "..."
  elif isinstance(value, str):
    if len(value) <= max_length:
      return value
    else:
      return f"str[{len(value)}] " + value[:max_length] + "..."
  elif isinstance(value, np.ndarray):
    vstr = f"ndarray({value.shape}, {value.dtype.str})"
    if value.size <= (max_length / 4):
      vstr += " = " + str(value)
    return vstr
  elif np.ndim(value) == 0:
    return str(value)   # Scalar data.
  else:
    return str(type(value))


def pretty_print_tokens(tokens: Any, vocab: seqio.Vocabulary,
                        max_length: int) -> str:
  """Convert tokens to a human-readable string."""
  if isinstance(tokens, np.ndarray):
    tstr = f"ndarray({tokens.shape}, {tokens.dtype.str}) = "
  else:
    tstr = f"{str(type(tokens))} = "

  if np.ndim(tokens) == 1:
    tstr += pretty_print_tokens_1d(tokens, vocab, max_length)
  elif np.ndim(tokens) == 2:
    jtstr = ",\n    ".join([pretty_print_tokens_1d(s, vocab, max_length)
                            for s in tokens])
    tstr += f"[\n    {jtstr}\n  ]"
  else:
    tstr = pretty_print_value(tokens, max_length)
  return tstr


def pretty_print_tokens_1d(tokens: Any, vocab: seqio.Vocabulary,
                           max_length: int) -> str:
  """Decode tokens to a human-readable string for pretty printing."""

  assert np.ndim(tokens) == 1
  tstr = decode_tokens_1d(tokens[:max_length], vocab)

  # TODO(delesley): Make pretty-printing better for unicode.
  # Converting it to bytes and back will convert, e.g., newlines as "\n".
  tstr = str(tstr.encode("utf-8"))
  if len(tokens) > max_length:
    tstr += "..."
  return tstr


def decode_tokens_1d(tokens: Any, vocab: Any) -> str:
  r"""Convert a 1D array of tokens to a human-readable string.

  Args:
    tokens:     1-dimensional array of integers.
    vocab:      The vocabulary to detokenize the array.

  Returns:
    The detokenized string.
  """

  assert np.ndim(tokens) == 1
  # The type of tokens is np.ndarray((sequence_length,), "int32")
  # We have to convert this to an actual list of python integers, NOT numpy
  # integers, or decode will blow up, and fail to marshall the data to C++.
  dtoks = [int(i) for i in tokens]
  return _decode_tokens_to_string(dtoks, vocab)


def _decode_tokens_to_string(tokens: Any, vocab: seqio.Vocabulary) -> str:
  tstr = vocab.decode(tokens)
  if isinstance(tstr, str):
    return tstr
  # PassThroughVocabulary returns a list, not a string.
  assert isinstance(tstr, list)
  # Convert to a unicode string, opposite of _string_to_tokens().
  tstr = "".join([chr(t) for t in tokens])
  return tstr


def find_token_boundaries_1d(tokens: Any,
                             vocab: seqio.Vocabulary,
                             start_dummy_char: int = 100,
                            ) -> Tuple[str, List[float]]:
  """Convert a 1D array of tokens to a string, with token boundaries.

  This function will decode the tokens back to a string, and return an array,
  with the same length as the string, which is 1.0 for those characters which
  mark end-of-token, and 0.0 everywhere else.

  This function will decode each token individually to find the length of the
  token, and is thus not particularly fast.

  Due to whitespace handling, this function may return a slightly different
  string than simply calling vocab.decode(tokens).

  Args:
    tokens: A sequence of integer token values.
    vocab: The vocabulary for decoding.
    start_dummy_char: a "dummy" character, can be any non-special token number.

  Returns:
    A tuple of (decoded string, token boundaries)
  """

  prev_tok: int = start_dummy_char   # random previous token
  decoded_str = ""
  token_boundaries = []
  for t in tokens:
    tstr = decode_single_token(prev_tok, t, vocab)
    tlen = len(tstr)
    if tlen == 0:
      break                         # if decode_single_token has failed.
    for _ in range(0, tlen - 1):
      token_boundaries.append(0.0)  # 0 for chars in the middle of the token
    token_boundaries.append(1.0)    # 1 for the end-of-token char
    decoded_str += tstr
  return (decoded_str, token_boundaries)


def decode_single_token(prev_tok: int,
                        tok: int,
                        vocab: seqio.Vocabulary) -> str:
  """Decodes a single token to a string."""

  # The normal decode operation strips leading whitespace.  This is a hack
  # to put it back.  We add a the previous token to the front, which forces any
  # whitespace to be added, and then remove it from the output string.

  tstr = _decode_tokens_to_string([prev_tok, tok], vocab)
  pstr = _decode_tokens_to_string([prev_tok], vocab)
  if tstr.startswith(pstr):
    tstr = tstr[len(pstr):]
    return tstr
  else:
    return ""


def _batched_interleave_generator(
    ds: tf.data.Dataset,
    flat_map_func: Callable[[Dict[str, Any]], Iterable[Dict[str, np.ndarray]]],
    batch_size: int,
    auto_rewind: bool = False) -> Iterable[Dict[str, np.ndarray]]:
  """Generator which combines the interleave and batch dataset operations.

  Given a set of articles from ds, flat_map_func is applied to each article
  to break it into a sequence of segments.  The generator will return the
  segments from each article in sequential order, for transformer-XL style
  models that process long articles over multiple training steps.

  Articles are combined into batches of size batch_size, where each example in
  the batch is pulled from a different article. When one article ends, the
  generator will start pulling examples from the next article.  The overall
  result is similar to tf.Data.Dataset.interleave, except that interleave does
  not always maintain the same order of articles.  If this generator starts
  pulling from article "foo" as the 3rd item in the batch, then consecutive
  examples from "foo" will remain as the 3rd item until the article ends.  This
  guarantee is necessary to pass state from one training step to the next.

  If auto_rewind, then the generator will automatically grab a new iterator
  from ds at the end of the epoch, and increment the epoch counter. Otherwise,
  it will yield empty datasets until all articles in the batch have been
  completed.

  Args:
    ds:            A dataset of articles.
    flat_map_func: A function which returns an iterable over segments.
        Each segment is a dictionary of values, e.g. { "targets": ... }
    batch_size:    The number of articles in a batch.
    auto_rewind:   Automatically rewind ds at end of epoch.

  Yields:
    Batches of consecutive examples from articles.
    Each example has type: {
      "targets": int32[batch_size, sequence_length],
      ...  any additional keys for each segment, e.g. "loss_mask"
      "start_of_sequence": bool[batch_size],
      "epoch": int32[batch_size],
    }
  """

  ds_iter = ds.as_numpy_iterator()

  readers = [None] * batch_size         # Iterator for each article
  still_reading = [True] * batch_size   # End of current article?
  document_start = [True] * batch_size  # At start of each article.
  item_epochs = [0] * batch_size        # Epoch of the given item.
  epoch = 0

  # Main generator loop
  while any(still_reading):
    targets = [None] * batch_size
    for i in range(0, batch_size):
      targets_i = None
      while targets_i is None and still_reading[i]:
        if readers[i] is not None:
          try:
            # Grab the next item from the article.
            targets_i = next(readers[i])
          except StopIteration:
            # Article has ended; continue the while loop to grab a new one.
            readers[i] = None
        else:
          # Grab the next article from ds if the current one has ended.
          dsi = None
          try:
            article_i = next(ds_iter)             # Grab a new article.
            dsi = iter(flat_map_func(article_i))  # Break article into chunks.
          except StopIteration:
            logging.info("End of epoch %d.", epoch)
            if auto_rewind:
              epoch = epoch + 1
              logging.info("Starting epoch %d.", epoch)
              ds_iter = ds.as_numpy_iterator()      # Rewind ds_iter.
              article_i = next(ds_iter)             # Grab a new article.
              dsi = iter(flat_map_func(article_i))  # Break article into chunks.
            else:
              still_reading[i] = False  # No more articles on i
          if dsi is not None:
            # Start reading the new article.
            # Continue while loop to grab the first chunk.
            readers[i] = dsi
            document_start[i] = True
            item_epochs[i] = epoch
        # end while targets_i...
      targets[i] = targets_i
      # end for i in range(0, batch_size)...

    doc_start_orig = document_start.copy()  # We will return doc_start_orig.
    for i in range(0, batch_size):
      # Now that we've read an item, set /start/ to false for each reader.
      document_start[i] = False

    if not any(still_reading):
      logging.info("End of dataset.")
      break   # If there are no valid items, then stop immediately.

    item = _stack_dictionaries(targets)
    if item is None:
      logging.info("No valid items.")
      continue

    item["start_of_sequence"] = np.array(doc_start_orig)
    item["epoch"] = np.array(item_epochs)
    yield item


def _stack_dictionaries(items: List[Optional[Dict[str, np.ndarray]]]
                       ) -> Optional[Dict[str, Any]]:
  """Turn a list of dictionaries into a dictionary of stacked arrays."""

  # Find the first item in the stack that is not None.
  # This item will be used as a template for the result.
  first = None
  for d in items:
    if d is not None:
      first = d
      continue
  if first is None:
    return None  #  All items are None

  # Replace any None values with zeros, so that we can stack the arrays.
  def get_value(item, k: str):
    if item is None:
      return np.zeros_like(first[k])
    else:
      return item[k]

  # Return a single dictionary, which stacks the values in the other dicts.
  item_keys = first.keys()
  return {
      k: np.stack([get_value(items[i], k) for i in range(0, len(items))])
      for k in item_keys
  }


def split_and_batch(ds: tf.data.Dataset,
                    split_article_fn: Callable[[Dict[str, Any]],
                                               Iterable[Dict[str, np.ndarray]]],
                    segment_length: int,
                    batch_size: int,
                    output_keys: List[str],
                    auto_rewind: bool) -> tf.data.Dataset:
  """Converts articles to tokens and chops and batches them.

  See batched_interleave_generator for more details.

  Args:
    ds:                A dataset of articles.
    split_article_fn:  Split article into a sequence of segments.
    segment_length:    The length of each segment.
    batch_size:        The number of examples in each batch.
    output_keys:       A set of addtional keys (e.g. "loss_mask") that will
                       appear in the output examples.  These keys should match
                       whatever is produced by split_article_fn.
    auto_rewind:       If True, will automatically rewind at end of epoch.

  Returns:
    A dataset which yields examples of shape {
        "targets": int32[batch_size, sequence_length],
        "start_of_sequence": bool[batch_size],
        "epoch": int32[batch_size],
        --- additional keys may include ---:
        "loss_mask": bool[batch_size, sequence_length],
        "token_boundaries": float32[batch_size, sequence_length],
        "num_chars": A count of the number of detokenized characters,
        "nonzero_tokens": A count of the number of nonzero predicted tokens
    }
  """

  def wrap_batched_interleave_generator():
    return _batched_interleave_generator(ds,
                                         flat_map_func=split_article_fn,
                                         batch_size=batch_size,
                                         auto_rewind=auto_rewind)

  out_sig = {
      "targets": tf.TensorSpec(shape=(batch_size, segment_length),
                               dtype=tf.int32),
      "start_of_sequence": tf.TensorSpec(shape=(batch_size,), dtype=tf.bool),
      "epoch": tf.TensorSpec(shape=(batch_size,), dtype=tf.int32),
  }

  # TODO(delesley):  Rethink type signature plumbing.
  # The output signature here is a bit of a hack, because the type signature
  # for each of these keys is hard-coded here, even though the keys themselves
  # are defined elsewhere in the file.  However, it would be also ugly to
  # scatter the types across the file.
  if "num_chars" in output_keys:
    out_sig["num_chars"] = tf.TensorSpec(shape=(batch_size,),
                                         dtype=tf.int32)
  if "nonzero_tokens" in output_keys:
    out_sig["nonzero_tokens"] = tf.TensorSpec(shape=(batch_size,),
                                              dtype=tf.int32)
  if "loss_mask" in output_keys:
    out_sig["loss_mask"] = tf.TensorSpec(shape=(batch_size, segment_length),
                                         dtype=tf.bool)
  if "token_boundaries" in output_keys:
    out_sig["token_boundaries"] = tf.TensorSpec(
        shape=(batch_size, segment_length), dtype=tf.float32)

  cds = tf.data.Dataset.from_generator(wrap_batched_interleave_generator,
                                       output_signature=out_sig)
  return cds


@gin.configurable
def get_loss_mask_tokens(
    split: str,
    loss_mask_start_tokens: Sequence[int] = (),
    loss_mask_end_tokens: Sequence[int] = (),
    splits: Sequence[str] = ("all",)
) -> Tuple[Sequence[int], Sequence[int]]:
  """Returns two token sequences to indicate start and end of the loss.

  Please configure loss_mask_start_tokens, loss_mask_end_tokens, and
  split_filter via gin. Example gin config to only apply loss between tokens 2
  and 1 for the test set (and everywhere for any other data split):

  ```
  text_dataset.get_loss_mask_tokens:
    loss_mask_start_tokens=(2,)
    loss_mask_end_tokens=(1,)
    splits=("test",)
  ```

  Args:
    split: The mode ("test", "train", ...)
    loss_mask_start_tokens: token sequence to start the loss
    loss_mask_end_tokens: token sequence to stop the loss
    splits: Only compute the loss mask for splits in this list. By default it is
      'all', which is a reserved split string that applies to all splits.
  """
  loss_mask_start_tokens = tuple(loss_mask_start_tokens)
  loss_mask_end_tokens = tuple(loss_mask_end_tokens)
  if loss_mask_start_tokens and loss_mask_start_tokens == loss_mask_end_tokens:
    raise ValueError("loss_mask_start_tokens can't equal loss_mask_end_tokens.")
  if "all" in splits or split in splits:
    return loss_mask_start_tokens, loss_mask_end_tokens
  return (), ()


def loss_mask_from_tokens(tokens: np.ndarray, split: str) -> np.ndarray:
  """Compute a mask for language modelling loss using start and end tokens.

  This function can be configured with gin using get_loss_mask_tokens.

  Args:
    tokens: An array of integer tokens.
    split: The dataset split, e.g. "train" or "test".

  Returns:
    An loss mask of the same size and shape as tokens.
  """

  assert np.ndim(tokens) == 1
  tokens = tokens.astype(np.int32)

  start_tokens, end_tokens = get_loss_mask_tokens(split=split)
  if not start_tokens and not end_tokens:
    # default to not masking out any loss
    logging.log_first_n(
        logging.ERROR,
        "Computed loss mask but no start and end tokens were defined!", 1)
    return np.ones_like(tokens, dtype=bool)

  # If there are no start_indices, then all positions are masked out.
  mask = np.zeros_like(tokens)

  start_indices = _find_subsequence_idxs(tokens, start_tokens)
  if start_indices:
    end_indices = _find_subsequence_idxs(tokens, end_tokens)

    start_indices, end_indices = _matching_index_pairs(start_indices,
                                                       end_indices)

    # We include the start_tokens and the end_tokens, which represents that the
    # model must predict the location, the content, and the end of the
    # subsequence.
    start_indices = np.array(start_indices, dtype=np.int32)
    end_indices = np.array(end_indices, dtype=np.int32)
    end_indices += len(end_tokens)
    end_indices = np.fromiter((x for x in end_indices if x < len(tokens)),
                              dtype=np.int32)

    mask[start_indices] = 1
    mask[end_indices] = -1
    mask = np.cumsum(mask)
    # for safety, clamp to interval [0, 1]
    if np.any(mask < 0) or np.any(mask > 1):
      raise ValueError(
          "Computation of the loss mask failed! Please report to mrabe@. "
          "Split: %s, tokens: %s" % (split, tokens[:100]))
    logging.log_first_n(
        logging.INFO, "Computed loss mask from input: %s", 10, mask)
  return mask


def _find_subsequence_idxs(sequence: np.ndarray,
                           subsequence: Sequence[int]) -> List[int]:
  """Returns the indices where `subsequence` occurs in `sequence`."""
  if len(subsequence) == 0:  # pylint: disable=g-explicit-length-test
    return []
  subsequence = np.asarray(subsequence, dtype=np.int32)
  # use np.where as an efficient way to iterate over the whole array; but we can
  # only test for a single token, unfortunately.
  potential_matches = np.where(sequence == subsequence[0])[0]
  match_indices = []
  for start_index in potential_matches:
    if np.array_equal(sequence[start_index:start_index + len(subsequence)],
                      subsequence):
      match_indices.append(start_index)
  return match_indices


def _matching_index_pairs(
    start_indices: List[int],
    end_indices: List[int]) -> Tuple[List[int], List[int]]:
  """Matches each start index to the next larger end index (or sequence end).

  Args:
    start_indices: Indices where the spans should start.
    end_indices: Indices where the spans should end.

  Returns:
    Produces start and end indices that can be turned efficiently into a mask
    using a cumsum while limiting the values produced to 0s and 1s. Hence, the
    length of the returned start_indices and stop_indices is the same unless the
    last start index is matched by the end of the sequence.
  """
  start_indices = sorted(start_indices)
  end_indices = sorted(end_indices)
  # We reverse the index lists here, as we are reversing them again through the
  # interaction of pop and append in the while loop below.
  start_indices.reverse()
  end_indices.reverse()
  filtered_start_indices = []
  filtered_end_indices = []
  while start_indices or end_indices:
    start = math.inf
    if start_indices:
      start = start_indices[-1]
    end = math.inf
    if end_indices:
      end = end_indices[-1]

    if end == start and end != math.inf:
      raise ValueError("Start and end tokens should not be the same.")
    if end < start:
      end_indices.pop()
      continue
    if len(start_indices) >= 2 and start_indices[-2] < end:
      # We have two start indices before the next end index. So the second start
      # index will not have any effect. We want to remove it to later use the
      # start and end indices in an efficient cumsum and still limit the mask to
      # the range [0, 1].
      start_indices.pop(-2)
      continue
    filtered_start_indices.append(start_indices.pop())
    if end_indices:
      filtered_end_indices.append(end_indices.pop())

  return filtered_start_indices, filtered_end_indices




def synthetic_data(split: str) -> tf.data.Dataset:
  """Returns a synthetic data set with several long articles."""
  del split

  articles = [
      synthetic_text_data.text1_illiad_book1,
      synthetic_text_data.text2_huckleberry_finn,
      synthetic_text_data.text3_call_of_the_wild,
      synthetic_text_data.text4_the_prince
  ]
  logging.info("Building synthetic dataset (long).")
  ds = tf.data.Dataset.from_tensor_slices(articles)
  ds = ds.map(_wrap_text_in_dict)
  ds = ds.shuffle(4, reshuffle_each_iteration=True, seed=42)
  return ds


def _wrap_text_in_dict(text) -> Dict[str, Any]:
  return {"targets": text}


def _open_vocab_segment_before_space(atext: str) -> List[int]:
  """Add segmentation boundaries at end of word, just before each space."""
  max_word_length = 8
  next_atext = atext[1:] + "a"
  seg = []
  wlen = 0
  seg_i: bool
  for (c, nc) in zip(atext, next_atext):
    if c.isalpha():
      if nc.isalpha():
        seg_i = False
      else:
        seg_i = True       # EOW on last character
    elif not c.isspace():  # c is a "symbol" if it is not alpha, and not space.
      seg_i = True         # EOW after every symbol
    else:
      seg_i = False

    if seg_i or wlen >= max_word_length:
      seg.append(1)
      wlen = 0             # reset word length
    else:
      seg.append(0)
      wlen = wlen + 1      # increment word length
  return seg


def _open_vocab_segment_after_space(atext: str) -> List[int]:
  """Add segmentation boundaries on each space, just before the next word."""
  max_word_length = 8
  next_atext = atext[1:] + "a"
  seg = []
  seg_i: bool
  wlen = 0
  for (c, nc) in zip(atext, next_atext):
    if c.isalpha():
      if not nc.isalpha() and not nc.isspace():
        seg_i = True       # EOW on last char, if next char is not space
      else:
        seg_i = False
    else:
      if not nc.isspace():
        # For symbols: EOW on every symbol, unless followed by a space.
        # For spaces:  EOW right before the next non-space character.
        seg_i = True
      else:
        seg_i = False

    if seg_i or wlen >= max_word_length:
      seg.append(1)
      wlen = 0             # reset word length
    else:
      seg.append(0)
      wlen = wlen + 1      # increment word length
  return seg

