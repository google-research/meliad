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

"""Setup the data pipeline and launch the main training loop."""

from typing import Optional

from absl import flags
from absl import logging

import gin
import jax

import  training_loop
from transformer import language_model
from transformer import tasks  # pylint: disable=unused-import
from transformer import text_dataset


flags.DEFINE_string("workdir", "", "Directory to save model checkpoints.")
flags.DEFINE_string("load_dir", "", "Directory to load pre-trained model.")
flags.DEFINE_boolean("test_model", False,
                     "Run the model briefly for num_steps, for testing.")
flags.DEFINE_integer("num_steps", 110,
                     "Number of steps, when using test_model.")
flags.DEFINE_boolean("test_rerun_training_loop", False,
                     "Run the training loop a second time to test loading " +
                     "of checkpoints.")

flags.DEFINE_list(
    "gin_search_paths",
    ["transformer/configs"],
    "List of paths where the Gin config files are located.")
flags.DEFINE_multi_string(
    "gin_file", ["base_htrans.gin"], "List of Gin config files.")
flags.DEFINE_multi_string(
    "gin_param", None, "Newline separated list of Gin parameter bindings.")

FLAGS = flags.FLAGS


def parse_gin_configuration():
  """Load and parse Gin configuration from command-line flags."""
  for gin_file_path in FLAGS.gin_search_paths:
    logging.info("Added Gin search path %s", gin_file_path)
    gin.add_config_file_search_path(gin_file_path)
  for gin_file in FLAGS.gin_file:
    logging.info("Loading Gin config file %s", gin_file)
  if FLAGS.gin_param:
    for gin_param in FLAGS.gin_param:
      logging.info("Overriding Gin param %s", gin_param)
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param,
                                      print_includes_and_imports=True)
  config_str = gin.config_str()
  logging.info("==== Gin config. ====")
  config_lines = config_str.splitlines()
  for line in config_lines:
    logging.info("gin_config: %s", line)
  logging.info("Done parsing gin config.")


def run_training_loop(testing: bool = False, workdir: Optional[str] = None):
  """Setup data pipeline and launch the main training loop."""

  testing = testing or FLAGS.test_model
  if testing:
    logging.info("Testing model...")

  if not workdir:
    workdir = FLAGS.workdir
  logging.info("Working directory = %s", workdir)
  logging.info("Pretrained model load_dir = %s (len=%d)",
               FLAGS.load_dir, len(FLAGS.load_dir))

  logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
  logging.info("JAX local devices: %r", jax.local_devices())

  text_dataset.set_default_data_directory()
  task_config = language_model.TransformerTaskConfig()

  # Grab the vocab object for pretty-printing purposes.
  logging.info("Loading vocabulary.")
  task_vocab = text_dataset.load_text_dataset_vocabulary(
      task_config.dataset_name)

  logging.info("Configured vocab_size = %d", task_config.vocab_size)
  logging.info("Task vocabulary size = %d", task_vocab.vocab_size)
  if task_config.vocab_size < task_vocab.vocab_size:
    raise ValueError(
        "Task vocabulary size does not match configured vocab_size: " +
        f"{task_config.vocab_size} < {task_vocab.vocab_size}")

  # This function will get an iterator for the given mode.
  # Each python process runs in a separate shard, so the dataset must be
  # sharded.
  def get_dataset_iterator(mode: str, num_shards: int, shard_id: int,
                           batch_size_per_shard: int):
    nonlocal task_config
    nonlocal task_vocab

    if mode == "train":
      split = task_config.train_split
    elif mode == "test":
      # We don't shard the test set, because test set sharding can be
      # nondeterministic for, e.g. PG19.
      split = task_config.test_split
      num_shards = 1
      shard_id = 0
    else:
      raise ValueError(f"Invalid mode {mode}")

    (ds, ds_vocab) = text_dataset.load_text_dataset(
        name=task_config.dataset_name,
        split=split,   # test
        sequence_length=task_config.sequence_length,
        batch_size=batch_size_per_shard,
        sequential=task_config.sequential_chunks,
        num_shards=num_shards,
        shard_id=shard_id)

    assert ds_vocab.vocab_size == task_vocab.vocab_size
    return text_dataset.get_iterator_function(ds)

  # Pretty printing depends on the vocabulary object.
  def pretty_print_article(article) -> str:
    nonlocal task_vocab
    return text_dataset.pretty_print_article(article, {"targets": task_vocab},
                                             32768)

  # Logging pretty-printed summaries depends on the vocabulary object.
  process_summaries_fn = language_model.process_summaries_function(task_vocab)

  if not testing:
    # Run the training loop normally; num_steps is gin-configured by the task.
    trainer = training_loop.Trainer(
        batch_size_per_replica=task_config.batch_size,
        get_dataset_iterator_function=get_dataset_iterator,
        pretty_print_input_function=pretty_print_article,
        process_summaries_function=process_summaries_fn,
        load_dir=FLAGS.load_dir,
        workdir=workdir)
    trainer.train()
    return

  # Test the model by running the training loop for num_steps.
  logging.info("==== Test: running training loop the first time. ====")
  trainer = training_loop.Trainer(
      batch_size_per_replica=task_config.batch_size,
      get_dataset_iterator_function=get_dataset_iterator,
      pretty_print_input_function=pretty_print_article,
      process_summaries_function=process_summaries_fn,
      num_steps=FLAGS.num_steps,  # Ignore Gin config for these options.
      load_dir=FLAGS.load_dir,
      workdir=workdir)
  trainer.train()
  del trainer

  if not FLAGS.test_rerun_training_loop:
    return

  # Test checkpoint saving and loading by running the model a second time.
  logging.info("==== Test: running training loop the second time. ====")
  training_loop.clear_interstep_callbacks()
  trainer2 = training_loop.Trainer(
      batch_size_per_replica=task_config.batch_size,
      get_dataset_iterator_function=get_dataset_iterator,
      pretty_print_input_function=pretty_print_article,
      process_summaries_function=process_summaries_fn,
      num_steps=FLAGS.num_steps + 100,  # Ignore Gin config for these options.
      load_dir=FLAGS.load_dir,
      workdir=workdir)
  trainer2.train()
