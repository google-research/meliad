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

r"""Various utility functions for doing inference on data.

This file provides a simple procedural API for loading a model, loading data,
and running the model over data.  It is intended for use in, e.g., colabs.
"""

from typing import Any, Dict, Optional, Sequence, Tuple

from absl import logging
import gin
import jax
import  model_info
import  training_loop
from transformer import language_model
from transformer import text_dataset
import numpy as np
import seqio


ModelInfo = model_info.ModelInfo
TrainState = model_info.TrainState
Metrics = model_info.Metrics
Trainer = training_loop.Trainer
TrainingTask = training_loop.TrainingTask

ModelInput = Dict[str, Any]     # Input to model.
ArticleData = Sequence[ModelInput]
ArticleList = Sequence[ArticleData]


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


def read_articles(split: Optional[str] = None,
                  verbose: bool = False,
                  random_seed: int = 42,
                  num_articles: int = 1,
                 ) -> Tuple[ArticleList, seqio.Vocabulary]:
  """Read one or more articles from the dataset.

  This routine will read one or more articles from the data set.
  Each article consists of a list of segments.  Each segment can be fed to the
  model directly as input.

  Args:
    split: The dataset split to load from.  Defaults to the test split.
    verbose: If True, will dump the contents of the article to the log.
    random_seed: The random seed to use when shuffling the dataset.
                 Specifying the same seed should yield the same data.
    num_articles: The number of articles to return.

  Returns:
    A pair of (list_of_blocks, vocabulary)
  """

  logging.info("Reading article.")

  text_dataset.set_default_data_directory()
  task_config = language_model.TransformerTaskConfig()
  batch_size = 1

  if split is None:
    split = task_config.test_split

  (test_ds, vocab) = text_dataset.load_text_dataset(
      name=task_config.dataset_name,
      split=split,
      sequence_length=task_config.sequence_length,
      batch_size=batch_size,
      sequential=task_config.sequential_chunks,
      num_shards=1,
      shard_id=0,
      # In a colab, specify random_seed to avoid hanging on multihost broadcast.
      random_seed=random_seed,
  )

  logging.info("Configured vocab_size = %d", task_config.vocab_size)
  logging.info("Task vocabulary size = %d", vocab.vocab_size)
  if task_config.vocab_size < vocab.vocab_size:
    raise ValueError(
        "Task vocabulary size does not match configured vocab_size: " +
        f"{task_config.vocab_size} < {vocab.vocab_size}")

  articles = []
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

    if x["start_of_sequence"][0]:
      # Wait until we've at least started reading the first article.
      if article_segments:
        # Add old article to the list, and start a new one.
        articles.append(article_segments)
        if len(articles) >= num_articles:
          # The primary exit route from the `while True` loop:
          break
        article_segments = []

    if verbose:
      logging.info("Segment %d = %s", segment_num,
                   text_dataset.pretty_print_article(x, vocab_map,
                                                     max_length=10_000))
    article_segments.append(x)
    segment_num += 1

  logging.info("Done reading article: %d segments.", segment_num)
  logging.info("Num tokens = %d", segment_num * task_config.sequence_length)
  return (articles, vocab)


def create_model_and_task(
    vocab: seqio.Vocabulary,
    load_dir: Optional[str] = None,
) -> Tuple[TrainingTask, TrainState, ModelInfo, Trainer]:
  """Initialize the model and get a task for inference.

  The task will be configured to take test (inference) steps with the model.
  The task will also be configured to run on a single replica, at batch size 1.

  Args:
    vocab: The vocabulary for the training data, used for logging and decoding.
    load_dir: A directory which contains a pre-trained model.

  Returns:
    (task -- Has a run_step method to take individual steps with the model,
     tstate -- The TrainState object with trainable parameters,
     mdl_info -- A ModelInfo object (see model_info.py),
     trainer -- a Trainer object (see training_loop.py),
    )
  """

  logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
  logging.info("JAX local devices: %r", jax.local_devices())

  # This task won't be pulling from a dataset.
  def get_dataset_iterator(mode: str, num_shards: int, shard_id: int,
                           batch_size_per_shard: int):
    del mode, num_shards, shard_id, batch_size_per_shard
    return None

  # Create a trainer object.
  trainer = training_loop.Trainer(
      batch_size_per_replica=1,
      get_dataset_iterator_function=get_dataset_iterator,
      pretty_print_input_function=None,
      process_summaries_function=(
          language_model.process_summaries_function(vocab)),
      load_dir=load_dir,
      workdir="")             # Don't log or save checkpoints.
  # What follows is a reimplementation of parts of trainer.train()

  # Create and initialize the model.
  mdl_info = trainer.create_model_info()
  tstate = mdl_info.initialize_model()

  # Create a training task.
  writers = {}
  task = trainer.create_training_task("train", mdl_info, writers)

  # Register any additional actions.
  # Actions are cleared first for use with colab.
  training_loop.clear_interstep_callbacks()
  training_loop.register_interstep_callbacks()

  return (task, tstate, mdl_info, trainer)


def run_model(task: TrainingTask,
              tstate: TrainState,
              article_segments: ArticleData,
              vocab: seqio.Vocabulary,
              start_step: int = 0,
              verbose: bool = False,
             ) -> Tuple[Sequence[Metrics], TrainState]:
  """Run the model on an article, and return the outputs for each segment.

  Args:
    task: The task to run, from create_model_and_task.
    tstate: A TrainState for the model, from create_model_and_task.
    article_segments: An article returned from read_article.
    vocab: The vocabulary used for the article.
    start_step: The starting step number to use.  (E.g. mdl_info.step)
    verbose: If True, will send input and output to the log.

  Returns:
    A sequence of model outputs for each block, and a new TrainState.
  """

  logging.info("Running the model.")

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
  return (segment_outputs, tstate)


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
