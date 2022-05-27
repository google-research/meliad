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

"""Setup the data pipeline and launch the main training loop."""

from absl import flags
from absl import logging

import gin
import jax

import  training_loop
from transformer import decoder_stack
from transformer import models
from transformer import tasks  # pylint: disable=unused-import
from transformer import text_dataset


flags.DEFINE_string("workdir", "", "Directory to save model checkpoints.")
flags.DEFINE_string("load_dir", "", "Directory to load pre-trained model.")
flags.DEFINE_integer("num_steps", 110, "Number of steps.")

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
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)


def run_training_loop(testing: bool = False):
  """Setup data pipeline and launch the main training loop."""

  logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
  logging.info("JAX local devices: %r", jax.local_devices())

  text_dataset.set_default_data_directory()
  task_config = decoder_stack.TransformerTaskConfig()
  batch_size = task_config.batch_size * jax.local_device_count()

  (train_ds, vocab) = text_dataset.load_text_dataset(
      name=task_config.dataset_name,
      split=task_config.train_split,  # train
      sequence_length=task_config.sequence_length,
      batch_size=batch_size,
      sequential=task_config.sequential_chunks,
      shard_dataset=True)

  (test_ds, test_vocab) = text_dataset.load_text_dataset(
      name=task_config.dataset_name,
      split=task_config.test_split,   # test
      sequence_length=task_config.sequence_length,
      batch_size=batch_size,
      sequential=task_config.sequential_chunks,
      shard_dataset=False)

  logging.info("Configured vocab_size = %d", task_config.vocab_size)
  logging.info("Task vocabulary size = %d", vocab.vocab_size)
  assert vocab.vocab_size == test_vocab.vocab_size  # Sanity check.
  if task_config.vocab_size < vocab.vocab_size:
    raise ValueError(
        "Task vocabulary size does not match configured vocab_size: " +
        f"{task_config.vocab_size} < {vocab.vocab_size}")

  # Pretty printing depends on the vocabulary object.
  def pretty_print_article_fn(article) -> str:
    return text_dataset.pretty_print_article(article, {"targets": vocab}, 32768)

  train_ds_iter_fn = text_dataset.get_iterator_function(train_ds)
  test_ds_iter_fn = text_dataset.get_iterator_function(test_ds)

  if testing:
    # Build trainer, which is configurable by Gin, and run training loop.
    trainer = training_loop.Trainer(
        get_training_dataset_iterator=train_ds_iter_fn,
        get_test_dataset_iterator=test_ds_iter_fn,
        pretty_print_input_function=pretty_print_article_fn,
        process_summaries_function=models.process_summaries_function(vocab),
        num_steps=FLAGS.num_steps,  # Ignore Gin config for these options.
        load_dir=FLAGS.load_dir,
        workdir=FLAGS.workdir)
  else:
    trainer = training_loop.Trainer(
        get_training_dataset_iterator=train_ds_iter_fn,
        get_test_dataset_iterator=test_ds_iter_fn,
        pretty_print_input_function=pretty_print_article_fn,
        process_summaries_function=models.process_summaries_function(vocab),
        load_dir=FLAGS.load_dir,
        workdir=FLAGS.workdir)

  trainer.train()
