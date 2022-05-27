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

r"""Program to run a transformer model over a single article.

"""

# This program is currently a template, which can be expanded to do more
# sophisticated analysis.

from typing import Sequence

from absl import app
from absl import flags
from clu import platform
import jax
import tensorflow.compat.v2 as tf

from transformer import inference_utils
from transformer import tasks


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


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], "GPU")

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(f"process_index: {jax.process_index()}, "
                                       f"process_count: {jax.process_count()}")
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.workdir, "workdir")

  inference_utils.parse_gin_configuration(FLAGS.gin_file, FLAGS.gin_param,
                                          gin_paths=FLAGS.gin_search_paths)

  article_data = inference_utils.read_article(True)
  (_, vocab) = article_data
  (task, task_state, _) = inference_utils.create_model_and_task(
      vocab, load_dir=FLAGS.load_dir)
  outs = inference_utils.run_model(task, task_state, article_data,
                                   verbose=True)
  inference_utils.get_token_losses(outs)

if __name__ == "__main__":
  app.run(main)
