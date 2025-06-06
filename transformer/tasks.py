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

"""Add Tasks to registry."""

import functools
from typing import Dict, Optional

from transformer import text_dataset
import seqio
import t5.data
from t5.data import preprocessors
import tensorflow as tf


def _features_from_vocab(vocab: seqio.Vocabulary) -> Dict[str, seqio.Feature]:
  return {
      "targets": seqio.Feature(vocabulary=vocab, add_eos=False, dtype=tf.int32)
  }


def define_pg19_task(name: str, vocab: seqio.Vocabulary):
  seqio.TaskRegistry.add(
      name,
      seqio.TfdsDataSource(
          tfds_name="pg19:0.1.1"
      ),
      preprocessors=[
          functools.partial(text_dataset.rekey_articles,
                            rekey={"book_text": "targets"},
                            keep={"book_title", "book_id", "publication_date"}),
          seqio.preprocessors.tokenize,
      ],
      output_features=_features_from_vocab(vocab)
  )

T5_DEFAULT_VOCABULARY = t5.data.get_default_vocabulary()
define_pg19_task("pg19_bytes", seqio.ByteVocabulary())
define_pg19_task("pg19_tokens_t5", T5_DEFAULT_VOCABULARY)


