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

"""Implements memorizing-transformer style memory using dense attention."""

from typing import Any, Optional, Sequence, Tuple

from absl import logging
from flax import linen as nn
import gin

import jax
import jax.numpy as jnp

from transformer import attention
from transformer import nn_components
from transformer import position


Array = jax.Array
KVITuple = attention.KVITuple
ArraySeq = Sequence[Array]
OptArray = Optional[Array]

vshape = nn_components.vshape


# (mem_keys, mem_values, keys, values, query_index)
DenseKVMemoryState = Tuple[Array, Array, Array, Array, int]


# Implements recurrent_memory_layer.RecurrentStateModule
@gin.configurable
class DenseKVMemory(nn.Module):
  """Long-range non-differentiable memory."""

  # Configured by the enclosing TransformerLayer.
  # The mode corresponds to different steps as defined by the training loop.
  # E.g. a "train" step draws an example from the training corpus, and applies
  # a gradient, while a "test" step just calculates a loss, on an example drawn
  # from the test set.  Other possibilities are "validate", if the data set
  # has a validation split, or "generate".  A "generate" step runs the model
  # in inference mode to periodically produce examples of generated outputs,
  # which are logged along with other metrics for manual inspection of
  # generation quality.
  #
  # Because the memory/cache stores the context (and thus documents) from prior
  # steps, separate memories must be allocated for separate data set splits.
  # By default, the training loop has "test" and "train" splits.
  mode: str
  batch_size: int
  embedding_size: int
  num_heads: int
  head_size: int

  # gin configurable parameters
  memory_size: int = 32768    # Number of key,value entries in the memory.
  clear_memory: bool = True   # Clear memory at start of new document.

  supported_modes: Sequence[str] = ("train", "test")
  dtype: Any = jnp.float32

  def _get_memory_mode(self) -> Tuple[str, bool]:
    """Get the name of the memory, and whether to update it."""
    # This is a hack to ensure that "generate" steps generate text as a
    # continuation of the text that is stored in the "test" memory,
    # but it does not update the "test" memory.
    if self.mode == "generate":
      assert "test" in self.supported_modes
      return ("test", False)    # Use the test cache, but don't update it.
    elif self.mode == "init":
      return ("train", False)   # Use training cache for initialization.
    else:
      return (self.mode, True)

  def _allocate_memory(self, mem_mode: str):
    """Allocate memory for the given mode.  Called from setup()."""

    kv_shape = tuple([self.batch_size, self.memory_size,
                      self.num_heads, self.head_size])
    mem_keys = self.variable("state", "memory_keys_" + mem_mode,
                             jnp.zeros, kv_shape, dtype=self.dtype)
    mem_vals = self.variable("state", "memory_values_" + mem_mode,
                             jnp.zeros, kv_shape, dtype=self.dtype)
    write_index = self.variable("state", "write_index_" + mem_mode,
                                jnp.zeros, (), dtype=jnp.int32)
    return (mem_keys, mem_vals, write_index)

  def _get_mutable_memory(self) -> Tuple[Any, Any, Any]:
    """Return mutable (keys, values, write_index) for the current mode."""
    logging.info("dense_kv: Accessing memory for mode %s", self.mode)
    (mem_mode, _) = self._get_memory_mode()
    assert mem_mode in self.memories
    return self.memories[mem_mode]

  def _get_memory(self) -> Tuple[Array, Array, Array]:
    """Return (keys, vals, write_index) as jax arrays for the current mode."""
    (k, v, wi) = self._get_mutable_memory()
    return (k.value, v.value, wi.value)

  def setup(self):
    """Initialize the model."""
    # Allocate memories for all supported modes.
    memories: dict[str, Tuple[Any, Any, Any]] = {}
    for mkey in self.supported_modes:
      memories[mkey] = self._allocate_memory(mkey)
    self.memories = memories

  def also_use_self_attention(self) -> bool:
    """Returns True if this module can be used with self-attention."""
    # The purpose of this module is to learn to produce keys and values for
    # long-range attention.  Thus, we train with semi-long-range attention over
    # the current segment, rather than local sliding-window attention, which
    # is shorter range, and only attends over the current block.  Returning
    # False here disables sliding-window attention in RecurrentMemoryLayer.
    return False

  def start_segment(self,
                    keys: Array,
                    values: Array,
                    start_of_sequence: Array) -> DenseKVMemoryState:
    """Start processing a new segment."""
    assert keys.shape == values.shape
    (batch_size, segment_length, _, _) = keys.shape

    if (self.memory_size % segment_length) != 0:
      raise ValueError(f"Memory size {self.memory_size} must be a multiple of "
                       f" the segment length {segment_length}")

    (mem_keys, mem_vals, _) = self._get_memory()
    logging.info("dense_kv: Loading memory keys = %s", vshape(mem_keys))
    logging.info("dense_kv: Loading memory values = %s", vshape(mem_vals))

    # Clear memory by multiplying the appropriate batch rows by 0.
    if self.clear_memory:
      start_doc = jnp.asarray(start_of_sequence)
      logging.info("dense_kv: Clearing memory; start_of_doc = %s",
                   vshape(start_of_sequence))
      start_doc = start_doc.astype(jnp.bool)
      start_doc = start_doc.reshape(batch_size, 1, 1, 1)
      start_doc = jnp.logical_not(start_doc).astype(self.dtype)
      mem_keys = mem_keys * start_doc
      mem_vals = mem_vals * start_doc

    return (mem_keys, mem_vals, keys, values, 0)

  def end_segment(self, state: DenseKVMemoryState):
    """Finish processing the segment."""
    # Only update memory if the mode supports it.
    (_, update_mem) = self._get_memory_mode()
    if not update_mem:
      logging.info("Skipping memory update for mode %s", self.mode)
      return

    (mem_keys, mem_vals, keys, values, _) = state
    (mem_keys_var, mem_vals_var, write_index_var) = self._get_mutable_memory()
    write_index = write_index_var.value

    # Special case if the memory and segment are the same size;
    # just replace the whole memory with the current segment.
    (_, segment_length, _, _) = keys.shape
    if self.memory_size == segment_length:
      # dynamic_update_slice would otherwise assert that dtypes are the same.
      assert mem_keys_var.value.dtype == keys.dtype
      assert mem_vals_var.value.dtype == values.dtype
      mem_keys_var.value = keys
      mem_vals_var.value = values
      write_index_var.value = jnp.zeros((), dtype=jnp.int32)
      return

    # Write keys and values from the current segment to memory.
    logging.info("dense_kv: Writing memory keys = %s", vshape(keys))
    logging.info("dense_kv: Writing memory values = %s", vshape(values))
    mem_keys = jax.lax.dynamic_update_slice_in_dim(
        mem_keys, keys, write_index, axis=1
    )
    mem_vals = jax.lax.dynamic_update_slice_in_dim(
        mem_vals, values, write_index, axis=1
    )
    mem_keys_var.value = mem_keys
    mem_vals_var.value = mem_vals

    # Increment the write index
    segment_length = keys.shape[1]
    write_index = (write_index + segment_length) % self.memory_size
    write_index_var.value = write_index

  def read(self,
           state: DenseKVMemoryState,
           queries: Array,
           *,
           attention_scale_factor: Optional[Array]) -> Array:
    """Query memory."""
    (mem_keys, mem_vals, keys, values, queries_offset) = state
    logging.info("dense_kv: mem_read() -- queries_offset = %d", queries_offset)

    # keys are keys from the current segment.
    # queries are queries from the current block.
    # kq_pos is the distance from each query to a given key.
    (_, num_queries, _, _) = queries.shape  # (b, q, h, d)
    (_, num_keys, _, _) = keys.shape        # (b, k, h, d)
    kq_pos = position.relative_positions(num_queries, num_keys,
                                         offset=queries_offset)
    # A query can only attend to keys that come before it.
    # The distance from query to key is negative for previous keys.
    causal_mask = (kq_pos < 0)

    # Attend jointly over KV in memory, and KV from the current segment.
    # The current segment is differentiable, memory is not.
    # Ideally, the gradients for intra-segment lookups will be enough to
    # train decent keys and values for the non-differentiable memory.
    y = attention.joint_attention(keys_list=[mem_keys, keys],
                                  values_list=[mem_vals, values],
                                  queries=queries,
                                  causal_mask_list=[None, causal_mask],
                                  scale_factor=attention_scale_factor,
                                  dtype=self.dtype)
    return y

  def write(self,
            state: DenseKVMemoryState,
            keys: Array,
            values: Array,
            *,
            attention_scale_factor: Optional[Array]) -> DenseKVMemoryState:
    """Pretend to write keys and values from the current block to memory."""
    # We don't actually update the memory at the end of each block.
    # The memory is updated in end_segment instead.
    # This function merely increments queries_offset to track which block.
    assert keys.shape == values.shape
    assert keys.ndim == 4  # (batch_size, block_size, heads, embedding_dim)
    block_size = keys.shape[1]
    del keys
    del values
    del attention_scale_factor   # unused for write operations

    # Update queries_offset as we loop over blocks.
    # Note -- queries_offset is a python integer, not a jax Array, so this
    # function cannot be used with lax.scan.
    (mem_keys, mem_vals, keys, values, queries_offset) = state
    queries_offset += block_size
    return (mem_keys, mem_vals, keys, values, queries_offset)

