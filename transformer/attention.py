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

"""Transformer attention functions."""

import typing
from typing import Any, Callable, Mapping, NewType, Optional, Sequence, Tuple, Union

from absl import logging
from flax import linen as nn
import jax
import jax.numpy as jnp

from transformer import language_model
from transformer import nn_components
from transformer import position


Array = jax.Array
OptArray = Optional[Array]
ArrayTree = Union[Array, Tuple["ArrayTree", ...]]
DecoderState = NewType("DecoderState", Mapping[str, Array])

# Tuple of keys, values, importance.
KVITuple = Tuple[Array, Array, Optional[Array]]

# Tuple of keys, values, queries, queries2, importance.
KVQITuple = Tuple[Array, Array, Array, Optional[Array], Optional[Array]]


vshape = nn_components.vshape


def initial_kvi(shape: Sequence[int], use_importance: bool, dtype: Any):
  """Returns initial (zero) keys/values/i that can be passed to prev_kvi."""
  z = jnp.zeros(shape, dtype=dtype)
  if use_importance:
    i = jnp.zeros((shape[0], shape[1]), dtype=dtype)  # (bsize, window_length)
  else:
    i = None
  return (z, z, i)


def concat_kvqi(kvqi: KVQITuple, prev_kvi: Optional[KVITuple]) -> (
    Tuple[KVQITuple, Optional[KVITuple]]):
  """Concatenate previous keys,values with current keys,values.

  Args:
    kvqi: Current keys, values, queries, quieres2, importance.
    prev_kvi: Previous keys, values, importance.

  Returns:
    (kvqi: Concatenated (keys, values, queries, importance),
     next_kvi:  Next (keys, values, importance))  (from kvqi)
  """

  (keys, values, queries, queries2, importance) = kvqi
  # The current keys,values,importance will be passed to the next window.
  next_kvi = (keys, values, importance)
  (batch_size, _, num_heads, head_dim) = keys.shape    # (b, _, h, d)

  if prev_kvi is None:
    return (kvqi, None)   # If prev_kvi is None, next_kvi should be None.

  # Unpack prev_kvi and check shapes.
  (pkeys, pvalues, pimportance) = prev_kvi
  num_pkeys = pkeys.shape[1]
  if pkeys.shape[0] != batch_size:
    raise ValueError(
        f"Batch size in XL cache {pkeys.shape[0]} does not match {batch_size}."
        f" Perhaps set ModelInfo.restore_state_variables=False?")
  assert pkeys.shape == (batch_size, num_pkeys, num_heads, head_dim)
  assert pkeys.shape == pvalues.shape
  if pimportance is not None:
    assert pimportance.shape == (batch_size, num_pkeys)

  # Concatenate keys and values.
  keys = jnp.concatenate([pkeys, keys], axis=1)        # (b, k, h, d)
  values = jnp.concatenate([pvalues, values], axis=1)  # (b, k, h, d)
  if importance is not None:
    assert pimportance is not None
    importance = jnp.concatenate([pimportance, importance], axis=1)  # (b, k)
    logging.info("concat_kvqi: importance = %s", vshape(importance))

  return ((keys, values, queries, queries2, importance), next_kvi)


def simple_attention(keys: Array,
                     values: Array,
                     queries: Array,
                     importance: Optional[Array],
                     *,
                     relative_position_bias: Optional[Array] = None,
                     scale_factor: Optional[Array] = None,
                     causal_mask: Optional[Array] = None,
                     dropout_multiplier: Optional[Array] = None,
                     dtype: Any = jnp.float32) -> Array:
  """Simple attention from a set of queries to a set of keys,values.

  Args:
    keys: of shape [batch_size, num_keys, num_heads, head_dim].
    values: of shape [batch_size, num_keys, num_heads, head_dim].
    queries: of shape [batch_size, num_queries, num_heads, head_dim].
    importance: of shape [batch_size, num_keys].

    *: ---- the following arguments are passed by keyword only ----
    relative_position_bias:  A positional attention matrix of shape
          [num_heads, num_queries, num_keys]
    scale_factor:  Learned scale factor for use with normalized keys,queries
          of shape [num_heads]
    causal_mask: A boolean array of shape [num_heads, num_queries, num_keys]
    dropout_multiplier: A random mask of either 0.0 or 1.0/keep_prob,
          of shape [num_heads, num_queries, num_keys]
    dtype: data type to perform attention at.

  Returns:
    Attention outputs of shape [batch_size, num_queries, num_heads, head_size]
  """

  # (batch_size, num_keys, num_heads, head_dim)
  (batch_size, num_keys, num_heads, head_dim) = keys.shape  # (b, k, h, d)
  num_queries = queries.shape[1]
  assert keys.shape == values.shape
  assert queries.shape == (batch_size, num_queries, num_heads, head_dim)
  if importance is not None:
    assert importance.shape == (batch_size, num_keys)

  logging.info("attn: keys = %s", vshape(keys))
  logging.info("attn: queries = %s", vshape(queries))

  # Compute attention matrix.
  attn = jnp.einsum("...qhd,...khd->...hqk", queries, keys)  # (b, h, q, k)

  logging.info("attn: content attn = %s", vshape(attn))

  # Apply relative position bias.
  if relative_position_bias is not None:
    logging.info("attn: pbias = %s", vshape(relative_position_bias))
    relative_position_bias = jnp.asarray(relative_position_bias, dtype=dtype)
    pbias = position.broadcast_mask(relative_position_bias, attn)
    attn = attn + pbias

  # Apply importance bias.
  if importance is not None:
    logging.info("attn: importance = %s", vshape(importance))
    importance = jnp.asarray(importance, dtype=dtype)
    attn = attn + importance

  # Apply learned attention scale.
  if scale_factor is not None:
    logging.info("attn: learned attention scale: %s", vshape(scale_factor))
    # Broadcast scale over batch/keys/queries.
    scale_factor = jnp.asarray(scale_factor, dtype=dtype)
    scale_factor = scale_factor.reshape((1, num_heads, 1, 1))
    attn = attn * scale_factor

  # Apply causal mask.
  if causal_mask is not None:
    logging.info("attn: causal_mask = %s", vshape(causal_mask))
    causal_mask = position.broadcast_mask(causal_mask, attn)
    attn = jnp.where(causal_mask, attn, jnp.asarray(-1_000_000.0, dtype=dtype))

  logging.info("attn: pre-softmax attn = %s", vshape(attn))

  # Normalize attention matrix with softmax.
  # min_x should be much smaller than minimum expected values in attn, but
  # much larger than the masked_out values created by the causal mask. That
  # way, if all tokens are masked out, then softmax will attend to nothing,
  # rather than attend to everything equally.
  min_x = jnp.asarray(-1000.0, dtype=dtype)
  attn = nn_components.safe_softmax(attn, axis=-1, min_x=min_x)  # (b, h, q, k)

  # Apply dropout to attention matrix.
  if dropout_multiplier is not None:
    logging.debug("attn: drop = %r", dropout_multiplier)
    dropout_multiplier = jnp.asarray(dropout_multiplier, dtype=dtype)
    attn = attn * dropout_multiplier

  logging.info("attn: final attn = %s", vshape(attn))

  # Compute output -- values weighted by attention matrix.
  y = jnp.einsum("...hqk,...khd->...qhd", attn, values)  # (b, q, h, d)

  logging.info("attn: y = %s", vshape(y))
  return y


def joint_attention(keys_list: Sequence[Array],
                    values_list: Sequence[Array],
                    queries: Array,
                    *,
                    importance_list: Optional[Sequence[OptArray]] = None,
                    scale_factor: Optional[Array] = None,
                    causal_mask_list: Optional[Sequence[OptArray]] = None,
                    dtype: Any = jnp.float32) -> Array:
  """Attention from a set of queries to multiple sets of keys,values.

  The result of this function should be identical to what would happen if the
  keys and values in each list were concatenated together.  However, it avoids
  the memory overhead of doing an actual concatenate operation.  Joint
  attention involves doing a joint softmax over multiple attention matrices.

  Args:
    keys_list: list of arrays [batch_size, num_keys, num_heads, head_dim].
    values_list: list of arrays [batch_size, num_keys, num_heads, head_dim].
    queries: of shape [batch_size, num_queries, num_heads, head_dim].

    *: ---- the following arguments are passed by keyword only ----
    importance_list: list of arrays of shape [batch_size, num_keys].
    scale_factor:  Learned scale factor for use with normalized keys,queries
          of shape [num_heads]
    causal_mask_list: list of arrays of shape [num_heads, num_queries, num_keys]
    dtype: data type to perform attention at.

  Returns:
    Attention outputs of shape [batch_size, num_queries, num_heads, head_size]
  """

  if importance_list is None:
    importance_list = [None] * len(keys_list)
  if causal_mask_list is None:
    causal_mask_list = [None] * len(keys_list)

  assert len(values_list) == len(keys_list)
  assert len(importance_list) == len(keys_list)
  assert len(causal_mask_list) == len(keys_list)

  logging.info("joint_attn: queries = %s", vshape(queries))
  (batch_size, _, num_heads, head_dim) = queries.shape  # (b, q, h, d)

  if scale_factor is not None:
    # Broadcast scale over batch/keys/queries.
    logging.info("joint_attn: learned attention scale: %s",
                 vshape(scale_factor))
    scale_factor = jnp.asarray(scale_factor, dtype=dtype)
    scale_factor = scale_factor.reshape((1, num_heads, 1, 1))

  # Loop variables.
  attn_list = []
  input_lists = zip(keys_list, values_list, importance_list, causal_mask_list)

  for (keys, values, importance, causal_mask) in input_lists:
    num_keys = keys.shape[1]
    assert keys.shape == (batch_size, num_keys, num_heads, head_dim)
    assert keys.shape == values.shape
    if importance is not None:
      assert importance.shape == (batch_size, num_keys)

    logging.info("joint_attn: keys = %s", vshape(keys))

    # Compute attention matrix.
    attn = jnp.einsum("...qhd,...khd->...hqk", queries, keys)  # (b, h, q, k)
    logging.info("joint_attn: content attn = %s", vshape(attn))

    # Apply importance bias.
    if importance is not None:
      logging.info("joint_attn: importance = %s", vshape(importance))
      importance = jnp.asarray(importance, dtype=dtype)
      attn = attn + importance

    # Apply learned attention scale.
    if scale_factor is not None:
      attn = attn * scale_factor

    # Apply causal mask.
    if causal_mask is not None:
      logging.info("joint_attn: causal_mask = %s", vshape(causal_mask))
      causal_mask = position.broadcast_mask(causal_mask, attn)
      attn = jnp.where(causal_mask, attn,
                       jnp.asarray(-1_000_000.0, dtype=dtype))

    logging.info("joint_attn: pre-softmax attn = %s", vshape(attn))
    attn_list.append(attn)

  # Joint softmax over all attention matrices.
  # min_x should be smaller than the expected values in the attention matrix,
  # but much larger than the masked-out values from the causal mask.
  min_x = jnp.asarray(-1000.0, dtype=dtype)
  attn_list = nn_components.joint_softmax(attn_list, axis=-1, min_x=min_x)

  # Weighted sum of values using all attention matrices.
  y_total = None
  for (attn, values) in zip(attn_list, values_list):
    y = jnp.einsum("...hqk,...khd->...qhd", attn, values)  # (b, q, h, d)
    logging.info("joint_attn: y = %s", vshape(y))
    if y_total is None:
      y_total = y
    else:
      y_total += y

  logging.info("joint_attn: y_total = %s", vshape(y_total))
  assert y_total is not None   # Otherwise type checker complains.
  return y_total


def external_attention(external_keys: Array,
                       external_values: Array,
                       queries: Array,
                       *,
                       scale_factor: Optional[Array] = None,
                       dtype: Any = jnp.float32) -> Array:
  """Attention over (keys, values) retrieved from external memory.

  Args:
    external_keys: per-query keys from external memory, of shape
        [batch_size, num_queries, num_heads, num_neighbors, head_size]
    external_values: per-query values from external memory, of shape
        [batch_size, num_queries, num_heads, num_neighbors, head_size]
    queries: current queries, of shape:
        [batch_size, num_queries, num_heads, head_size]

    *: ---- the following arguments are passed by keyword only. ---
    scale_factor:  Learned scale factor for use with normalized keys,queries
          of shape [num_heads]
    dtype: data type to perform attention at.

  Returns:
    Attention outputs of shape [batch_size, num_queries, num_heads, head_size]
  """

  (batch_size, num_queries, num_heads, _, head_dim) = external_keys.shape
  assert queries.shape == (batch_size, num_queries, num_heads, head_dim)
  assert external_values.shape == external_keys.shape

  # Build attention matrix.
  logging.info("ext_attn: external keys = %s", vshape(external_keys))
  ext_attn = jnp.einsum("...qhd,...qhid->...hqi", queries, external_keys)

  logging.info("ext_attn: external_mem_attn: %s", vshape(ext_attn))
  if scale_factor is not None:
    scale_factor = jnp.asarray(scale_factor, dtype=dtype)
    scale_factor = scale_factor.reshape((1, num_heads, 1, 1))
    logging.info("ext_attn: scaling external_mem_attn by %s",
                 vshape(scale_factor))
    ext_attn = ext_attn * scale_factor

  ext_attn = nn.softmax(ext_attn, axis=-1)

  # Compute weighted sum of values.
  ext_y = jnp.einsum("...hqi,...qhid->...qhd", ext_attn, external_values)
  logging.info("ext_attn: ext_y = %s", vshape(ext_y))
  return ext_y


def sliding_attention_window_shape(kvi: KVITuple,
                                   prev_kvi: Optional[KVITuple],
                                   queries: Array,
                                   window_length: int) -> Tuple[int, int]:
  """Return (num_queries, num_keys) for the sliding attention window.

  Used internally by meliad attention models.  Uses the shapes of given arrays
  to determine whether sliding window is being used, and if so, what the size
  of the window should be.  Also does shape checking on all arrays.

  Returns (window_length, window_length * 2) for sliding window,
  or (queries.shape[1], keys.shape[1]) for a conventional vanilla transformer.

  Args:
    kvi: tuple of (keys, values, importance)
    prev_kvi: tuple of previous block of (keys, values, importance)
    queries: queries
    window_length: selected window length for sliding window.

  Returns:
    (num_queries, num_keys) for a single window.
  """

  # Do error checking here.
  (keys, values, importance) = kvi
  assert keys.shape == queries.shape
  assert values.shape == queries.shape

  # Get sizes...
  (batch_size, sequence_length, _, _) = queries.shape
  num_prev_keys = prev_kvi[0].shape[1] if prev_kvi is not None else 0

  if importance is not None:
    assert importance.shape == (batch_size, sequence_length)

  assert window_length > 0
  if window_length >= sequence_length:
    # No sliding window.
    num_queries = sequence_length
    num_keys = sequence_length + num_prev_keys
  else:
    # Sliding window.
    if prev_kvi is not None:
      assert num_prev_keys == window_length
    num_queries = window_length
    num_keys = window_length * 2

  return (num_queries, num_keys)


def split_tree(tree: ArrayTree, sections: int, axis: int = 0) -> (
    Sequence[ArrayTree]):
  """Recursively splits a possibly nested tuple of arrays along the given axis.

  Args:
    tree: A nested tree of tuples and arrays.
    sections: The number of sections to split the tree into.
    axis: The axis to do the split on arrays.

  Returns:
    A list of trees, of length sections, where each has the same shape as the
    original, but with arrays of size 1/sections.
  """

  if tree is None:
    return [None] * sections
  elif isinstance(tree, jnp.ndarray):
    return jnp.split(tree, sections, axis=axis)
  elif isinstance(tree, tuple):
    # Recursively split each element of the tuple into a list.
    branch_lists = [split_tree(tree_i, sections, axis=axis) for tree_i in tree]
    # Rearrange the tuple of lists into a list of tuples.
    return [tuple([brs[i] for brs in branch_lists]) for i in range(sections)]
  else:
    raise ValueError("Argument %r must be an ndarray or tuple." % tree)


def concat_trees(tree_list: Sequence[ArrayTree], axis: int = 0) -> ArrayTree:
  """Merges a list of trees into a single tree by concatenating their elements.

  Args:
    tree_list: A list of trees, all of the same shape.
    axis: The axis to concatenate arrays on.

  Returns:
    A single tree, with the same shape as the trees in tree_list.
  """

  # All trees in the list are required to have the same shape.
  # We return a tree with the same shape as each of the trees in the list,
  first_tree = tree_list[0]
  if first_tree is None:
    # Merge a list of None into a single None.
    for tree_i in tree_list:
      assert tree_i is None
    return None
  elif isinstance(first_tree, jnp.ndarray):
    # Concatenate a list of arrays.
    for tree_i in tree_list:
      assert isinstance(tree_i, jnp.ndarray)
    return jnp.concatenate(tree_list, axis=axis)
  elif isinstance(first_tree, tuple):
    # Reshape a list of tuples into a tuple of concatenated lists.
    for tree_i in tree_list:
      assert isinstance(tree_i, tuple) and len(tree_i) == len(first_tree)
    num_branches = len(first_tree)
    return tuple([concat_trees([tree[b] for tree in tree_list], axis=axis)
                  for b in range(num_branches)])
  else:
    raise ValueError("Argument %r must be an ndarray or tuple." % first_tree)


def _split_sequence_into_blocks(tree: ArrayTree,
                                sections: int,
                                axis: int = 0,
                                transpose: bool = True) -> ArrayTree:
  """Reshape the sequence_length dimension into block_length, sections.

  This function breaks a sequence of items into equal sized blocks by
  reshaping the sequence_length dimension into (num_blocks, block_length).

  Args:
    tree: A pytree of Arrays.
    sections: The number of sections or blocks in the sequence.
    axis: the axis that holds the sequence length dimension.
    transpose: If True, arrays will be further transposed so that the
      section dimension is dimension 0, which is required by lax.scan.

  Returns:
    A pytree of reshaped and transposed arrays.
  """

  # We could use jax tree utils for this, but we do it the hard way so the
  # implementaiton can be compared with split_tree.
  if tree is None:
    return None
  elif isinstance(tree, jnp.ndarray):
    tree = typing.cast(Array, tree)  # Tell type-checker about isinstance
    ndim = tree.ndim
    wlen = tree.shape[axis] // sections
    assert sections * wlen == tree.shape[axis]  # Must be evenly divisible.

    # Break the axis dimension into sections * block_length
    arr = tree
    sh = list(arr.shape)
    nshape = sh[0:axis] + [sections, wlen] + sh[axis + 1:]
    arr = jnp.reshape(arr, nshape)

    # Transpose sections to be dimension 0, for use with lax.scan.
    if transpose:
      tdims = [axis] + list(range(0, axis)) + list(range(axis + 1, ndim + 1))
      arr = jnp.transpose(arr, tdims)
    return arr
  elif isinstance(tree, tuple):
    return tuple([
        _split_sequence_into_blocks(b, sections, axis,
                                    transpose=transpose) for b in tree
    ])
  else:
    raise ValueError("Argument %r must be an ndarray or tuple." % tree)


def _join_blocks_into_sequence(tree: ArrayTree,
                               sections: int,
                               axis: int = 0,
                               transpose: bool = True) -> ArrayTree:
  """Inverts _split_sequence_into_blocks()."""

  # We could use jax tree utils for this, but we do it the hard way so the
  # implementation can be compared with split_tree.
  if tree is None:
    return None
  elif isinstance(tree, jnp.ndarray):
    tree = typing.cast(Array, tree)  # Tell type-checker about isinstance
    ndim = tree.ndim - 1   # Input tree has 1 extra dimension on front.
    assert axis < ndim

    # Transpose dimension 0 back to its proper place, for use with lax.scan.
    arr = tree
    if transpose:
      tdims = list(range(1, axis + 1)) + [0] + list(range(axis + 1, ndim + 1))
      arr = jnp.transpose(arr, tdims)

    # Combine the sections and block_length dimensions.
    sh = list(arr.shape)
    wlen = tree.shape[axis + 1]  # Block length.
    nshape = sh[0:axis] + [sections * wlen] + sh[axis + 2:]
    arr = jnp.reshape(arr, nshape)
    return arr
  elif isinstance(tree, tuple):
    return tuple([
        _join_blocks_into_sequence(b, sections, axis,
                                   transpose=transpose) for b in tree
    ])
  else:
    raise ValueError("Argument %r must be an ndarray or tuple." % tree)


def split_and_scan(fn: Callable[[ArrayTree, ArrayTree],
                                Tuple[ArrayTree, ArrayTree]],
                   carry: ArrayTree,
                   input_arrays: ArrayTree,
                   sections: int,
                   axis: int = 0,
                   max_unrolled_windows: int = -1) -> (
                       Tuple[ArrayTree, ArrayTree]):
  """Scan over a set of input arrays in chunks.

  Splits each array in 'input_arrays' into the number of chunks given by
  'sections', and then loops over the chunks using a scan operation.
  Returns a concatenation of the results.

  Args:
    fn: A function from (carry, input_i) -> (carry, output_i).
    carry: The initial state for the scan, that will be passed from one
           iteration to the next.
    input_arrays: A nested tree of tuples of arrays.
    sections: The number of sections or chunks for the split.
    axis: The axis to split each array along.
    max_unrolled_windows: If 0 <= max_unrolled_windows < sections,
        use jax.lax.scan rather than unrolling the windows with a python loop.

  Returns:
    (carry, output)
  """

  if sections == 1:
    logging.info("Single window, no scan.")
    return fn(carry, input_arrays)

  if axis < 0:
    raise ValueError(f"Axis must be positive. Got {axis}")

  logging.info("Scanning over %d windows", sections)

  if 0 <= max_unrolled_windows and max_unrolled_windows < sections:
    logging.info("Using jax.lax.scan.")
    in_arrs = _split_sequence_into_blocks(input_arrays, sections, axis=axis,
                                          transpose=True)
    (carry, out_arrs) = jax.lax.scan(fn, carry, in_arrs)
    output_arrays = _join_blocks_into_sequence(out_arrs, sections, axis=axis,
                                               transpose=True)
    return (carry, output_arrays)

  logging.info("Using unrolled for-loop.")
  in_list = split_tree(input_arrays, sections, axis=axis)
  out_list = []

  for (k, in_chunk) in enumerate(in_list):
    logging.info("Processing window %d", k)
    (carry, out_chunk) = fn(carry, in_chunk)
    out_list.append(out_chunk)

  output_arrays = concat_trees(out_list, axis=axis)
  return (carry, output_arrays)


def sliding_window(prev_block: Optional[Array],
                   blocks: Array) -> Tuple[Array, Array]:
  """Combine blocks into windows for sliding-window attention.

  Each block in the sequence is concatenated with the previous block in the
  sequence to yield overlapping blocks that are each twice as long.  The
  first block is concatenated with prev_block, which should be the last block
  from the previous training step.  In addition to the windows, this function
  returns the last block in the sequence, which can be passed to the next
  training step as prev_block.

  Args:
    prev_block: Array of shape (batch_size, block_length, ...)
        If None, zeros will be used instead.
    blocks: Array of shape (batch_size, num_blocks, block_length, ...)

  Returns:
    (windows: Array of shape (batch_size, num_blocks, block_length * 2, ...)
     next_block: Array of shape (batch_size, block_length, ...)
  """
  (batch_size, _, block_length) = blocks.shape[:3]   # _ = num_blocks

  if prev_block is None:
    bdims = blocks.shape[3:]  # The remaining ... dimensions
    pblock_shape = tuple([batch_size, 1, block_length] + list(bdims))
    prev_block = jnp.zeros(pblock_shape, dtype=blocks.dtype)
    logging.info("sliding_window: using zeros for previous block: %s",
                 vshape(prev_block))
  else:
    # Insert dim for num_blocks, e.g.  (b, k, ...) --> (b, 1, k, ...)
    assert prev_block.shape[:2] == (batch_size, block_length)
    if prev_block.shape[2:] != blocks.shape[3:]:
      raise ValueError(f"sliding_window: prev_block shape {prev_block.shape} "
                       f" is not compatible with blocks shape {blocks.shape}")
    prev_block = jnp.expand_dims(prev_block, axis=1)

  # Shift all of the blocks over by 1 to get the "previous block".
  # next_block.shape == prev_block.shape
  # p_blocks.shape == blocks.shape
  (p_blocks, next_block) = language_model.shift_right(
      blocks, axis=1, shift_by=1, prepend_array=prev_block
  )
  # Remove dim for num_blocks, e.g. (b, 1, k, ...) -> (b, k, ...)
  next_block = jnp.squeeze(next_block, axis=1)

  # Concatenate each block with the previous block, along block_length axis.
  windows = jnp.concatenate([p_blocks, blocks], axis=2)
  return (windows, next_block)


def parallel_sliding_window_attention(
    single_window_attention_fun: Callable[[KVQITuple], Array],
    kvqi: KVQITuple,
    prev_kvi: Optional[KVITuple],
    num_windows: int) -> Tuple[Array, Optional[KVITuple]]:
  """Peform parallel sliding window attention.

  If num_windows == 1, and prev_kvi is None, this will default to normal
  (vanilla) attention.  If num_windows == 1, and prev_kvi is not None, it will
  implement transformer-XL, using prev_kvi as the XL cache.

  If num_windows > 1, this will implement a differentiable sliding window.

  Args:
    single_window_attention_fun: A function to do attention within a
      single window.
    kvqi: Keys, values, queries, and importance.
      k,v,q have shape (batch_size, sequence_length, num_heads, head_dim)
      i has shape (batch_size, sequence_length)
    prev_kvi: Cached keys and values from the previous training step.
    num_windows: The number of windows.

  Returns:
    (ys: Array of shape (batch_size, sequence_length, embedding_dim),
     next_kvi: Keys and values to cache for the next training step.
    )
  """

  if prev_kvi is not None:
    logging.info("windowed_attention: prev_keys = %s", vshape(prev_kvi[0]))
  else:
    logging.info("windowed_attention: prev_keys = None")

  if num_windows == 1:
    logging.info("windowed_attention: No sliding window.")
    # For transformer-XL, concatenate with kvi from previous training step.
    # For vanilla transformers, this is a no-op.
    (kvqi, next_kvi) = concat_kvqi(kvqi, prev_kvi)
    ys = single_window_attention_fun(kvqi)
    return (ys, next_kvi)

  logging.info("windowed_attention: Parallel sliding window attention: "
               "num_windows = %d", num_windows)
  # Reshape the sequence length dimension into (num_blocks, block_length)
  # All arrays must have shape (batch_size, sequence_length, ...)
  kvqi = _split_sequence_into_blocks(kvqi,
                                     sections=num_windows,
                                     axis=1,
                                     transpose=False)

  # Construct the sliding windows.
  # (queries2 is None in models that don't use cross-attention)
  (keys, values, queries, queries2, importance) = kvqi
  assert queries2 is None
  if prev_kvi is not None:
    (prev_keys, prev_vals, prev_imp) = prev_kvi
  else:
    prev_keys = None
    prev_vals = None
    prev_imp = None

  (keys, next_keys) = sliding_window(prev_keys, keys)
  (values, next_vals) = sliding_window(prev_vals, values)
  if importance is not None:
    (importance, next_imp) = sliding_window(prev_imp, importance)
  else:
    assert prev_imp is None
    next_imp = None
  next_kvi = (next_keys, next_vals, next_imp)

  # Parallel map over all windows.
  # axis 1 is the extra num_blocks dimension that the reshape gave us.
  vmapped_wfun = jax.vmap(single_window_attention_fun, in_axes=1, out_axes=1)

  # Call attention function.
  kvqi = (keys, values, queries, None, importance)
  ys = vmapped_wfun(kvqi)
  logging.info("windowed_attention: ys = %s", vshape(ys))

  # Reshape from (..., num_blocks, block_length, ...) to (..., seq_len, ...)
  ys = _join_blocks_into_sequence(ys,
                                  sections=num_windows,
                                  axis=1,
                                  transpose=False)
  ys = typing.cast(Array, ys)
  logging.info("windowed_attention: ys (reshaped) = %s", vshape(ys))

  return (ys, next_kvi)


def sequential_sliding_window_attention(
    single_window_attention_fun: Callable[[KVQITuple], Array],
    kvqi: KVQITuple,
    prev_kvi: Optional[KVITuple],
    num_windows: int,
    max_unrolled_windows: int = -1) -> Tuple[Array, Optional[KVITuple]]:
  """Peform sequential sliding window attention.

  This version is identical to parallel_sliding_window_attention(), but instead
  of using jax.vmap, it uses either a python for-loop or lax.scan.  Each
  block in the sequence will take the previous block as the "carry" argument,
  and then pass itself to the next block.

  This version avoids the memory overhead of concatenating large arrays, but
  will take longer to compile, especially if num_windows is large.

  Args:
    single_window_attention_fun: A function to do attention within a
      single window.
    kvqi: Keys, values, queries, and importance.
      k,v,q have shape (batch_size, sequence_length, num_heads, head_dim)
      i has shape (batch_size, sequence_length)
    prev_kvi: Cached keys and values from the previous training step.
    num_windows: The number of windows.
    max_unrolled_windows: Uses lax.scan if the num_windows is greater than this
      number.  The default is -1, which means never use lax.scan.

  Returns:
    (ys: Array of shape (batch_size, sequence_length, embedding_dim),
     next_kvi: Keys and values to cache for the next training step.
    )
  """

  logging.info("sequential windowed attention: num_windows = %d",
               num_windows)

  def sequential_wfun(carry, inputs):
    prev_kvi = carry
    kvqi = inputs
    (kvqi, next_kvi) = concat_kvqi(kvqi, prev_kvi)
    ys = single_window_attention_fun(kvqi)
    return (next_kvi, ys)

  (next_kvi, ys) = split_and_scan(sequential_wfun,
                                  carry=prev_kvi,
                                  input_arrays=kvqi,
                                  sections=num_windows,
                                  axis=1,
                                  max_unrolled_windows=max_unrolled_windows)
  ys = typing.cast(Array, ys)
  next_kvi = typing.cast(Optional[KVITuple], next_kvi)
  return (ys, next_kvi)
