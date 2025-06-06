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

"""Jax functions for parallel recurrence in overlapping windows.
"""

from typing import Any, Callable, Optional, Tuple, Union

from absl import logging

import einops
import jax
import jax.numpy as jnp
from transformer import nn_components


Array = jax.Array
vshape = nn_components.vshape


def windowed_recurrence(recurrent_fun: Callable[[Any, Any], Tuple[Any, Any]],
                        initial_state: Union[Array, Tuple[Array, ...]],
                        inputs: Tuple[Array, ...],
                        block_length: int,
                        initial_blocks: Optional[Tuple[Array, ...]] = None,
                        left_to_right: bool = True) -> Tuple[Array, ...]:
  """Run a recurrent function in parallel over overlapping windows.

  This function will break an input sequence into overlapping windows, and
  run a recurrent function in parallel within each window.  This limits the
  context length of the recurrent function to block_length, instead of the
  entire sequence.  However, it is much, much faster on parallel hardware, for
  cases where the recurrent function only needs local context.

  Each overlapping window has a prefix of the previous block_length items.
  The recurrent_fun is run over the prefix to compute a carry value that
  incorporates prior context.  It then runs over the current block (also of
  length block_length), to get output values for the block.

  The initial_block argument provides a prefix for the first window (i.e. the
  first block of the sequence), and should be used to initialize the carry.

  Recurrence will be done in parallel over batch_size * num_blocks.

  Args:
    recurrent_fun: A function from (carry, input) -> (carry, output)
        Both input and output must be a tuple.
        The carry has the same number of items as initial_state.
        input is a tuple of arrays of shape (batch_size, num_blocks, 1, ...)
        output is a tuple of arrays of shape (batch_size, num_blocks, 1, ...)
        carry is array(s) of shape (batch_size, num_blocks, ...),
    initial_state: The initial value of the recurrent state (or carry), which
        is an array, or tuple of arrays, of shape (batch_size, ...).  It will
        be broadcast to array(s) of shape: (batch_size, num_blocks, ...).
        In most cases, this is just zeros of the appropriate shape.
    inputs: Tuple of input arrays.  Each array must have shape:
        (batch_size, sequence_length, ...).  The inputs will be sliced into
        slices, and passed as the arguments to recurrent_fun.
    block_length: The length of each block.
    initial_blocks: Tuple of arrays of shape (batch_size, block_length, ...)
        The initial_block is prepended onto the beginning of the sequence
        in order to obtain a window covering the first block.  The initial
        block can thus be used to set up the recurrent state (the carry)
        properly at the start of the sequence.  (The initial_state cannot be
        used for this, because it is broadcast in parallel over all windows,
        not just the first one.)  Defaults to zeros if not specified.
    left_to_right: If True, recurrence will be applied from left-to-right,
        if False, it will run in the opposite direction, from right-to-left.

  Returns:
    A tuple of arrays of shape (batch_size, sequence_length, ...)
  """
  assert isinstance(inputs, tuple)
  for (j, x_j) in enumerate(inputs):
    logging.info("windowed_recurrence: input %d = %s", j, vshape(x_j))

  # Check that the shape of the input arguments all match.
  batch_size = 0
  sequence_length = 0
  for (i, arg) in enumerate(inputs):
    if arg.ndim < 2:
      raise ValueError(f"Invalid shape for inputs[{i}]: {arg.shape}; "
                       f"should be shape (batch_size, seq_len, ...)")
    if i == 0:
      # Grab batch_size and sequence_length from the first argument.
      batch_size = arg.shape[0]
      sequence_length = arg.shape[1]
      continue
    if arg.shape[0] != batch_size:
      raise ValueError(
          f"Shape mismatch: inputs[{i}] of shape {arg.shape} differs from "
          f"inputs[0] of shape {inputs[0].shape} along batch dimension.")
    if arg.shape[1] != sequence_length:
      raise ValueError(
          f"Shape mismatch: inputs[{i}] of shape {arg.shape} differs from "
          f"inputs[0] of shape {inputs[0].shape} along sequence dimension.")

  if initial_blocks is not None:
    assert len(initial_blocks) == len(inputs)
    for (i, b) in enumerate(initial_blocks):
      if b.ndim < 2:
        raise ValueError(f"Invalid shape for initial_blocks[{i}]: {b.shape}; "
                         f"should be shape (batch_size, seq_len, ...).")
      if b.shape[0] != batch_size:
        raise ValueError(
            f"Shape mismatch: initial_blocks[{i}] of shape {b.shape} differs "
            f"from inputs[0] of shape {inputs[0].shape} along batch dimension.")
      if b.shape[1] != block_length:
        raise ValueError(
            f"Shape mismatch: initial_blocks[{i}] of shape {b.shape} "
            f"should have block_length {block_length} as dimension 1.")
  else:
    # Use None for each initial_block; split_into_windows will create zeros.
    initial_blocks = [None] * len(inputs)

  if sequence_length % block_length != 0:
    raise ValueError(f"Sequence length {sequence_length} must be divisible by"
                     f" the block length {block_length}.")
  num_blocks = sequence_length // block_length

  # Broadcast initial_state over the num_blocks dimension.
  def reshape_istate(istate: Array) -> Array:
    logging.info("windowed_recurrence: carry = %s", vshape(istate))
    # Add num_blocks dimension to istate, and broadcast over all blocks.
    ist_shape = istate.shape[1:]
    istate = einops.rearrange(istate, "b ... -> b 1 ...", b=batch_size)
    istate = jnp.broadcast_to(istate, (batch_size, num_blocks, *ist_shape))
    return istate

  carry = jax.tree.map(reshape_istate, initial_state)

  windowed_inputs = (  # tuple of arrays.
      # Step 1: Divide a long sequence into overlapping windows.
      split_into_windows(xs, block_length, iblock, left_to_right=left_to_right)
      for (xs, iblock) in zip(inputs, initial_blocks)
  )
  tuple_of_split_inputs = (  # tuple of lists.
      # Step 2: Split each window into a list of separate elements.
      jnp.split(xs, block_length * 2, axis=2) for xs in windowed_inputs
  )
  # Step 3: Convert tuple of lists into list of input-tuples.
  list_of_inputs = list(zip(*tuple_of_split_inputs))
  assert len(list_of_inputs) == 2 * block_length  # Guaranteed by split, above.

  if not left_to_right:
    # To reverse direction, reverse the list before and after the loop.
    list_of_inputs.reverse()

  # lax.scan is ludicrously slow, so we use a python for-loop.
  # This is much faster, assuming block_length isn't too large.
  list_of_outputs = []
  logging.info("windowed_recurrence: %d iterations", len(list_of_inputs))
  for (i, xsi) in enumerate(list_of_inputs):
    if i == 0:
      # Log input shapes for debugging purposes.
      for (j, xj) in enumerate(xsi):
        logging.info("windowed_recurrence: arg %d = %s", j, vshape(xj))
    # Call recurrent function.
    (carry, ysi) = recurrent_fun(carry, xsi)
    assert isinstance(ysi, tuple)
    if i == 0:
      # Log output shapes for debugging purposes.
      for (j, yj) in enumerate(ysi):
        logging.info("windowed_recurrence: out %d = %s", j, vshape(yj))
    list_of_outputs.append(ysi)

  if not left_to_right:
    # To reverse direction, reverse the list before and after the loop.
    list_of_outputs.reverse()

  # Undo Step 1a.  Drop the prefix or postfix of each window from the outputs.
  assert len(list_of_outputs) == 2 * block_length
  if left_to_right:
    # Drop prefix from outputs.
    list_of_outputs = list_of_outputs[block_length:]
  else:
    # Drop postfix from outputs.
    list_of_outputs = list_of_outputs[:block_length]

  tuple_of_split_outputs = (  # tuple of lists
      # Undo Step 3. Convert list of output-tuples to a tuple of output-lists.
      list(xs) for xs in zip(*list_of_outputs)
  )
  concat_outputs = (  # tuple of arrays
      # Undo Step 2. Concatenate the (previously split) values in each block.
      jnp.concatenate(ys, axis=2) for ys in tuple_of_split_outputs
  )

  def ys_reshape(ys: Array) -> Array:
    nonlocal batch_size
    nonlocal sequence_length
    reshaped = einops.rearrange(
        ys, "batch nblocks blen ... -> batch (nblocks blen) ...")
    assert reshaped.shape[0] == batch_size
    assert reshaped.shape[1] == sequence_length
    return reshaped

  outputs = (
      # Undo Step 1b. Reshape blocks back into a sequence again.
      ys_reshape(ys) for ys in concat_outputs
  )
  outputs = tuple(outputs)
  for (j, y_j) in enumerate(outputs):
    logging.info("windowed_recurrence: output %d = %s", j, vshape(y_j))
  return outputs


def split_into_windows(xs: Array,
                       block_length: int,
                       initial_block: Optional[Array] = None,
                       left_to_right: bool = True) -> Array:
  """Split xs into overlapping windows of block_length * 2.

  Args:
    xs: Array of shape (batch_size, sequence_length, ...)
    block_length: int -- length of the blocks.
    initial_block: Array of shape (batch_size, block_length, ...)
        Initial block to prepend onto xs to create the first window.
        Defaults to zero if not specified.
        If left_to_right is False, then initial_block is appended to xs.
    left_to_right: If True, each window will have the prefix on the left,
        suitable for a left-to-right RNN.  If false, the prefix will be added
        to the right instead.

  Returns:
    Array of shape (batch_size, num_blocks, block_length * 2, embed_dim)
    or (batch_size, num_blocks, block_length * 2)
  """
  logging.info("split_into_windows: xs = %s", vshape(xs))

  # Reshape xs into blocks of length block_size.
  # Each block is then concatenated with the previous block to yield
  # overlapping windows of size block_size*2.
  # Zeros or initial_blocks are prepended onto the first block.
  (batch_size, sequence_length) = xs.shape[:2]
  embed_dims = list(xs.shape[2:])
  assert sequence_length % block_length == 0
  num_blocks = sequence_length // block_length

  # Get the initial (or last) block for the overlapping windows.
  xpad_shape = tuple([batch_size, 1, block_length] + embed_dims)
  if initial_block is not None:
    logging.info("split_into_windows: initial_block = %s",
                 vshape(initial_block))
    if initial_block.shape[:2] != (batch_size, block_length):
      raise ValueError(f"initial_block of shape {initial_block.shape} should "
                       f"have shape ({batch_size}, {block_length}, ...).")
    if initial_block.shape[2:] != xs.shape[2:]:
      raise ValueError(f"initial_block of shape {initial_block.shape} should "
                       f"have dimensions that match xs of shape {xs.shape}, "
                       f"except for dimension 1 (block_length).")
    xpad = jnp.reshape(initial_block, xpad_shape).astype(xs.dtype)
  else:
    xpad = jnp.zeros(xpad_shape, dtype=xs.dtype)

  # Reshape xs into blocks.
  xs = einops.rearrange(
      xs,
      "batch (nblocks blen) ... -> batch nblocks blen ...",
      batch=batch_size,
      nblocks=num_blocks,
      blen=block_length,
  )

  # Create overlapping windows.
  if left_to_right:
    xs_prefix = jnp.concatenate([xpad, xs[:, :-1, ...]], axis=1)
    xs2 = jnp.concatenate([xs_prefix, xs], axis=2)   # concat prev & curr block.
  else:
    xs_postfix = jnp.concatenate([xs[:, 1:, ...], xpad], axis=1)
    xs2 = jnp.concatenate([xs, xs_postfix], axis=2)  # concat curr & next block.

  logging.info("split_into_windows: result = %s", vshape(xs2))
  return xs2
