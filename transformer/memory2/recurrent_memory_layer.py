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

"""Implements a transformer layer with block-wise recurrence."""

import typing
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

from absl import logging
import gin

import jax

from transformer import attention
from transformer import nn_components
from transformer import transformer_layer


Array = jax.Array
OptArray = Optional[Array]
KVITuple = attention.KVITuple
KVQITuple = attention.KVQITuple
KVITupleFlaxVars = Tuple[Any, Any, Any]   # Tuple of flax Variables.
RecurrentState = Any

vshape = nn_components.vshape


class RecurrentStateModule(Protocol):
  """Defines the interface expected for recurrent_state_factory.

  RecurrentStateModule is the interface for a flax module that defines an
  external "memory" of some kind, for long-context experiments.
  It supports read and write operations.
  """

  def also_use_self_attention(self) -> bool:
    """Returns True if this module can be used with self-attention."""
    return True

  def start_segment(self,
                    keys: Array,
                    values: Array,
                    start_of_sequence: Array) -> RecurrentState:
    """Start processing a new segment.

    A segment is the total sequence for a single training step.

    Args:
      keys: Keys for the current segment, of shape
          (batch_size, segment_length, num_heads, head_dim)
      values: Values for the current segment, of the same shape.
      start_of_sequence: Array of booleans of shape (batch_size,)
          which is True when starting a new document.

    Returns:
      A state object that can be used to query and update the memory.
    """
    return None

  def end_segment(self, state: RecurrentState):
    """Finish processing the segment.

    Args:
      state: The state object from the initial call to start_segment.
    """
    pass

  def read(self,
           state: RecurrentState,
           queries: Array,
           *,
           attention_scale_factor: Optional[Array]) -> Array:
    """Attend over memory using the given queries.

    Args:
      state: The state object from the initial call to start_segment.
      queries: Queries for the current block of shape
          (batch_size, block_length, num_heads, head_dim)
      attention_scale_factor: Array of shape (num_heads,).
          Used with key/query normalization.

    Returns:
      An array of shape (batch_size, block_length, num_heads, head_dim)
    """
    return queries   # Satisfy the type checker.

  def write(self,
            state: RecurrentState,
            keys: Array,
            values: Array,
            *,
            attention_scale_factor: Optional[Array]) -> RecurrentState:
    """Update the memory with the given keys and values from the current block.

    Blocks are processed sequentially, and the recurrent state is updated at
    the end of each block with the keys and values for that block.

    Args:
      state: Recurrent state, on entry to the current block.
      keys: Keys from the current block of shape
          (batch_size, block_length, num_heads, head_dim)
      values: Values from the current block of shape
          (batch_size, block_length, num_heads, head_dim)
      attention_scale_factor: Array of shape (num_heads,).
          Used with key/query normalization.

    Returns:
      An updated state, on exit from the current block.
    """
    return state


@gin.configurable
class RecurrentMemoryLayer(transformer_layer.TransformerLayer):
  """Transformer layer equipped with recurrent memory."""

  # Factory that returns a RecurrentStateModule.
  recurrent_state_factory: Any = None

  def setup(self):
    super().setup()
    self.recurrent_memory = self.recurrent_state_factory(
        mode=self.mode,
        batch_size=self.batch_size,
        embedding_size=self.embedding_size,
        num_heads=self.num_heads,
        head_size=self.head_size,
        dtype=self.dtype)

  def windowed_attention(self,
                         single_window_attn_fn: Callable[[KVQITuple], Array],
                         kvqi: KVQITuple,
                         prev_kvi: Optional[KVITuple],
                         *,
                         start_of_sequence: Array,
                         attention_scale_factors: Dict[str, OptArray],
                         num_windows: int) -> Tuple[Array, Optional[KVITuple]]:
    """Implement sliding window attention, with recurrent memory."""

    logging.info("Recurrent memory layer.")
    # Define a function that loops over all windows in a segment, and keeps
    def single_window_attention_with_memory(carry, inputs):
      nonlocal self
      nonlocal single_window_attn_fn
      nonlocal attention_scale_factors
      (prev_kvi, rstate) = carry
      kvqi_w = inputs
      (keys_w, values_w, queries_w, _, _) = kvqi_w

      # Read from (attend over) memory.
      ys = self.recurrent_memory.read(
          rstate,
          queries_w,
          attention_scale_factor=attention_scale_factors["self"])
      # Write keys,values from current window to memory.
      next_rstate = self.recurrent_memory.write(
          rstate,
          keys_w,
          values_w,
          attention_scale_factor=attention_scale_factors["self"])

      # Self-attention.
      # Some implementations of memory may subsume self-attention.
      # Note that only local self-attention uses relative positions.
      # Memory does not use relative positions.
      if self.recurrent_memory.also_use_self_attention():
        (kvqi_w, next_kvi) = attention.concat_kvqi(kvqi_w, prev_kvi)
        ys += single_window_attn_fn(kvqi_w)
      else:
        assert prev_kvi is None
        next_kvi = None

      next_carry = (next_kvi, next_rstate)
      return (next_carry, ys)

    # Load recurrent memory state from previous training step cache.
    (keys, values, _, _, _) = kvqi
    rec_state = self.recurrent_memory.start_segment(
        keys, values, start_of_sequence=start_of_sequence)

    # Loop over all the blocks, while updating recurrent state.
    (next_carry, attn_ys) = attention.split_and_scan(
        single_window_attention_with_memory,
        carry=(prev_kvi, rec_state),
        input_arrays=kvqi,
        sections=num_windows,
        axis=1,
        max_unrolled_windows=-1)  # Always unroll fully.

    # Store recurrent memory state in cache for use on the next training step.
    (next_kvi, next_rstate) = next_carry
    self.recurrent_memory.end_segment(next_rstate)

    # Cast b/c split_and_scan does not have precise return type information.
    attn_ys = typing.cast(Array, attn_ys)
    next_kvi = typing.cast(Optional[KVITuple], next_kvi)
    return (attn_ys, next_kvi)

