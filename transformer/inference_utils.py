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

r"""Various utility functions for doing inference on data.

This file provides a simple procedural API for loading a model, loading data,
and running the model over data.  It is intended for use in, e.g., colabs.
"""

from typing import Any, Dict, Optional, Sequence, Tuple

from absl import logging

import gin
import jax
import numpy as np
import seqio

import  training_loop
from transformer import decoder_stack
from transformer import models
from transformer import text_dataset


Trainer = training_loop.Trainer
TrainState = training_loop.TrainState
TrainingTask = training_loop.TrainingTask
PRNGKeys = training_loop.PRNGKeys

ModelInput = Dict[str, Any]     # Input to model.
MetricsOutput = Dict[str, Any]  # Metrics output by model.
ArticleData = Tuple[Sequence[ModelInput], seqio.Vocabulary]
TaskState = Tuple[TrainState, int]


DEFAULT_GIN_PATHS = [
    "transformer/configs"
]


def parse_gin_configuration(gin_files: Optional[Sequence[str]],
                            gin_params: Optional[Sequence[str]],
                            gin_paths: Optional[Sequence[str]] = None):
  """Load gin configuration options.

  Args:
    gin_files: A list of gin file names with the configuration to load.
    gin_params: A list of additional parameter overrides.
    gin_paths: A list of paths to search for gin_files.
  """

  # We allow None values to more easily handle command-line flags.
  if gin_files is None:
    gin_files = []
  if gin_params is None:
    gin_params = []
  if gin_paths is None:
    gin_paths = DEFAULT_GIN_PATHS

  logging.info("Parsing gin configuration.")
  for path in gin_paths:
    logging.info("Added Gin search path %s", path)
    gin.add_config_file_search_path(path)
  for file_name in gin_files:
    logging.info("Loading Gin config file %s", file_name)
  for param in gin_params:
    logging.info("Overriding Gin param %s", param)
  gin.parse_config_files_and_bindings(gin_files, gin_params)


def read_article(split: Optional[str] = None,
                 verbose: bool = False) -> ArticleData:
  """Read a single article from the dataset and save it as a list of blocks.

  This routine will return blocks for a single article; so the tokens will
  have a batch size of 1. The blocks can be fed to the model directly as input.

  Args:
    split: The dataset split to load from.  Defaults to the test split.
    verbose: If True, will dump the contents of the article to the log.

  Returns:
    A pair of (list_of_blocks, vocabulary)
  """

  logging.info("Reading article.")

  text_dataset.set_default_data_directory()
  task_config = decoder_stack.TransformerTaskConfig()
  batch_size = 1

  if split is None:
    split = task_config.test_split

  (test_ds, vocab) = text_dataset.load_text_dataset(
      name=task_config.dataset_name,
      split=split,
      sequence_length=task_config.sequence_length,
      batch_size=batch_size,
      sequential=task_config.sequential_chunks,
      shard_dataset=False)

  logging.info("Configured vocab_size = %d", task_config.vocab_size)
  logging.info("Task vocabulary size = %d", vocab.vocab_size)
  if task_config.vocab_size < vocab.vocab_size:
    raise ValueError(
        "Task vocabulary size does not match configured vocab_size: " +
        f"{task_config.vocab_size} < {vocab.vocab_size}")

  article_segments = []
  ds_iter = test_ds.as_numpy_iterator()
  vocab_map = {"targets": vocab}

  segment_num = 0
  while True:
    try:
      x = next(ds_iter)
    except StopIteration:
      logging.info("End of epoch? Something went wrong.")
      break

    # Make sure we've started reading, otherwise it immediately quits...
    if article_segments:
      if x["start_of_sequence"][0]:
        break

    if verbose:
      logging.info("Segment %d = %s", segment_num,
                   text_dataset.pretty_print_article(x, vocab_map,
                                                     max_length=10_000))
    article_segments.append(x)
    segment_num += 1

  logging.info("Done reading article: %d segments.", segment_num)
  logging.info("Num tokens = %d", segment_num * task_config.sequence_length)
  return (article_segments, vocab)


def create_model_and_task(vocab: seqio.Vocabulary,
                          load_dir: Optional[str] = None) -> (
                              Tuple[TrainingTask, TaskState, Trainer]):
  """Initialize the model and get a task for inference.

  The task will be configured to take test (inference) steps with the model.
  The task will also be configured to run on a single replica, at batch size 1.

  Args:
    vocab: The vocabulary for the training data, used for logging and decoding.
    load_dir: A directory which contains a pre-trained model.

  Returns:
    (task -- has a run_step method to take individual steps with the model,
     state -- contains trainable parameters and other state,
     trainer -- a Trainer object (see training_loop.py))
  """

  logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
  logging.info("JAX local devices: %r", jax.local_devices())

  # This task won't be pulling from a dataset.
  def null_iter_fn():
    return None

  trainer = training_loop.Trainer(
      get_training_dataset_iterator=null_iter_fn,
      get_test_dataset_iterator=None,
      pretty_print_input_function=None,
      process_summaries_function=models.process_summaries_function(vocab),
      load_dir=load_dir,
      workdir="",            # Don't log or save checkpoints.
      replicate_mode=False)  # Run on a single device at batch size 1.

  # Create and initialize the model.
  (tstate, start_step, imodel, prngs) = trainer.initialize_model()

  # Create an inference task.
  writers = {}
  task = trainer.create_training_task("test", imodel, prngs, writers)

  # Register any additional actions.
  # Actions are cleared first for use with colab.
  training_loop.clear_interstep_callbacks()
  training_loop.register_interstep_callbacks()

  task_state = (tstate, start_step)
  return (task, task_state, trainer)


def run_model(task: TrainingTask, task_state: TaskState,
              article_data: ArticleData, verbose: bool = False) -> (
                  Sequence[MetricsOutput]):
  """Run the model on an article, and return the outputs for each segment.

  Args:
    task: The task to run, from create_model_and_task.
    task_state: The state of the model, from create_model_and_task.
    article_data: The article and vocabulary, from read_article.
    verbose: If True, will send input and output to the log.

  Returns:
    A sequence of model outputs for each block.
  """

  logging.info("Running the model.")

  (article_segments, vocab) = article_data
  (tstate, start_step) = task_state
  vocab_map = {"targets": vocab}

  # Ignore the iterator for the test task, and loop over the article.
  step = start_step
  segment_num = 0

  # Loop over the article, and run the model on each segment.
  segment_outputs = []
  for x in article_segments:
    if verbose:
      logging.info("Segment [%d] = %s", segment_num,
                   text_dataset.pretty_print_article(x, vocab_map,
                                                     max_length=10_000))
    else:
      logging.info("Segment %d, step %d.", segment_num, step)

    (tstate, metrics_np) = task.run_step(tstate, x, step)
    training_loop.run_interstep_callbacks("test", step)
    segment_outputs.append(metrics_np)

    if verbose:
      logging.info("Output [%d] = %s", segment_num, metrics_np)

    del x
    segment_num += 1
    step += 1

  logging.info("Done running the model: %d segments.", segment_num)
  return segment_outputs


def get_token_losses(segment_outputs: Sequence[Any]) -> np.ndarray:
  """Return the loss for each token in a sequence.

  Given a list of model outputs, extract the token losses from each output
  and concatenate them together.

  Args:
    segment_outputs: the outputs from run_model().

  Returns:
    An array of shape (batch_size, sequence_length), of float.
  """

  block_token_losses = []
  for seg in segment_outputs:
    if "token_losses" in seg:
      block_token_losses.append(seg["token_losses"])
    else:
      raise ValueError("Token losses were not recorded.")

  logging.info("Got token losses for %d segments", len(block_token_losses))
  token_losses = np.concatenate(block_token_losses, axis=-1)
  logging.info("token_losses.shape = %r", token_losses.shape)
  return token_losses
