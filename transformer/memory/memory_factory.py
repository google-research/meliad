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

"""Flax modules and functions for using external memory."""

from typing import Any, Optional, Sequence, Tuple

from absl import logging
from flax import linen
import gin
import jax
from transformer.memory import memory_layer



PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any
MemoryResource = Any


class MemoryManager:
  """Manages any external resources that may be required by external memory.

  MemoryManager also functions as a factory, to create Flax modules that will
  read and write to whatever external memory has been configured.
  """

  def __init__(self,
               batch_size: int,
               mode: str,
               num_heads: int,
               key_size: int,
               value_size: int,
               database_size: Optional[int] = None,
               dtype: Dtype = "float32",
               off_device_memory: Optional[MemoryResource] = None,
               cross_batch_memory: bool = True):
    """Create a MemoryManager object.

    A MemoryManager configures external memory, and is used as a factory to
    construct flax modules that read or write to the memory.

    Args:
      batch_size: The number of separate documents in a batch.
      mode:       e.g. ("train", or "test")
      num_heads:  The number of transformer heads.
      key_size:   The length of the key vectors.
      value_size: The length of the value vectors.
      database_size:  The total number of tokens in the database.
      dtype:      The datatype used for keys and values.
      off_device_memory: An object which manages underlying SCAM memory.
          If None, then the model will use on-device memory.
      cross_batch_memory: Whether memory should be shared between sequences in
          the same batch.
    """
    if not cross_batch_memory:
      assert off_device_memory is None
    self.cross_batch_memory = cross_batch_memory
    self.batch_size = batch_size
    self.mode = mode
    self.num_heads = num_heads
    self.key_size = key_size
    self.value_size = value_size
    self.database_size = database_size
    self.dtype = dtype
    self.off_device_memory = off_device_memory

  def create_memory_layer(self) -> linen.Module:
    """Create a flax Module that implements external memory."""
    num_datasets = (
        self.num_heads if self.cross_batch_memory  #
        else self.batch_size * self.num_heads)
    if self.off_device_memory is not None:
      mem_layer = None
      if mem_layer is None:
        raise ValueError("Off-device memory is not supported at this time.")
      return memory_layer.BatchedMemory(
          mem_layer,
          split_dimensions=(-2,),
      )
    else:
      assert self.database_size is not None
      split_dimensions: Sequence[int] = (0, -2)
      disallow_reset_because: str = ""
      database_size = self.database_size  # database_size modified below
      if self.cross_batch_memory:
        # We want self.database_size to represent the number of stored
        # tokens *per batch element*.
        database_size *= self.batch_size
        split_dimensions = (-2,)
        disallow_reset_because = (
            "Memory is shared between documents. Therefore you should not "
            "reset all memory whenever a single document ends. Relevant gin "
            "config settings: memory_on_tpu_factory.cross_batch_memory, "
            "TransformerLayer.memory_reset_on_new_doc."
        )
      mem_layer = memory_layer.MemoryOnTpu(
          num_datasets=num_datasets,
          key_features=self.key_size,
          value_features=self.value_size,
          database_size=self.database_size,
          dtype=self.dtype,
          disallow_reset_because=disallow_reset_because,
      )
    # Handle queries of shape [batch_size, seq_len, num_heads, kv_features]
    return memory_layer.BatchedMemory(mem_layer,
                                      split_dimensions=split_dimensions)


@gin.configurable
def memory_on_tpu_factory(batch_size: int,
                          mode: str,
                          num_heads: int = gin.REQUIRED,
                          key_size: int = gin.REQUIRED,
                          value_size: int = gin.REQUIRED,
                          database_size: int = gin.REQUIRED,
                          dtype: Dtype = gin.REQUIRED,
                          cross_batch_memory: bool = False) -> MemoryManager:
  """Implement SCAM memory on device."""
  return MemoryManager(batch_size=batch_size,
                       mode=mode,
                       num_heads=num_heads,
                       key_size=key_size,
                       value_size=value_size,
                       database_size=database_size,
                       dtype=dtype,
                       off_device_memory=None,
                       cross_batch_memory=cross_batch_memory)


