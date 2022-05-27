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

"""Transformer attention functions."""

import typing
from typing import Any, Callable, Mapping, NewType, Optional, Sequence, Tuple, Union

from absl import logging
from flax import linen as nn
import jax
import jax.numpy as jnp

from transformer import nn_components
from transformer import position


Array = jnp.ndarray
ArrayTree = Union[Array, Tuple["ArrayTree", ...]]
DecoderState = NewType("DecoderState", Mapping[str, Array])

# Tuple of keys, values, importance.
KVITuple = Tuple[Array, Array, Optional[Array]]

# Tuple of keys, values, queries, queries2, importance.
KVQITuple = Tuple[Array, Array, Array, Optional[Array], Optional[Array]]

# Tuple of scale factors.  See TransformerBase.attention_scale_factors().
AttnScaleTuple = Tuple[Optional[Array], Optional[Array]]


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
    logging.info("attn: importance = %r", importance)

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

  logging.info("attn: keys = %r", keys)
  logging.info("attn: queries = %r", queries)

  # Compute attention matrix.
  attn = jnp.einsum("...qhd,...khd->...hqk", queries, keys)  # (b, h, q, k)

  logging.info("attn: content attn = %r", attn)

  # Apply relative position bias.
  if relative_position_bias is not None:
    logging.info("attn: pbias = %r", relative_position_bias)
    relative_position_bias = jnp.asarray(relative_position_bias, dtype=dtype)
    pbias = position.broadcast_mask(relative_position_bias, attn)
    attn = attn + pbias

  # Apply learned attention scale.
  if scale_factor is not None:
    logging.info("attn: learned attention scale: %s", scale_factor)
    # Broadcast scale over batch/keys/queries.
    scale_factor = jnp.asarray(scale_factor, dtype=dtype)
    scale_factor = scale_factor.reshape((1, num_heads, 1, 1))
    attn = attn * scale_factor

  # Apply causal mask.
  if causal_mask is not None:
    causal_mask = position.broadcast_mask(causal_mask, attn)
    attn = jnp.where(causal_mask, attn, jnp.asarray(-1_000_000.0, dtype=dtype))

  logging.info("attn: pre-softmax attn = %r", attn)

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

  logging.info("attn: final attn = %r", attn)

  # Compute output -- values weighted by attention matrix.
  y = jnp.einsum("...hqk,...khd->...qhd", attn, values)  # (b, q, h, d)

  logging.info("attn: y = %r", y)
  return y


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
  logging.info("ext_attn: external keys = %r", external_keys)
  ext_attn = jnp.einsum("...qhd,...qhid->...hqi", queries, external_keys)

  logging.info("ext_attn: external_mem_attn: %s", ext_attn)
  if scale_factor is not None:
    scale_factor = jnp.asarray(scale_factor, dtype=dtype)
    scale_factor = scale_factor.reshape((1, num_heads, 1, 1))
    logging.info("ext_attn: scaling external_mem_attn by %s", scale_factor)
    ext_attn = ext_attn * scale_factor

  ext_attn = nn.softmax(ext_attn, axis=-1)

  # Compute weighted sum of values.
  ext_y = jnp.einsum("...hqi,...qhid->...qhd", ext_attn, external_values)
  logging.info("ext_attn: ext_y = %r", ext_y)
  return ext_y


def sliding_attention_window_shape(kvi: KVITuple,
                                   prev_kvi: Optional[KVITuple],
                                   queries: Array,
                                   window_length: int) -> Tuple[int, int]:
  """Return (num_queries, num_keys) for the sliding attention window."""

  # Do error checking here.
  (keys, values, importance) = kvi
  assert keys.shape == queries.shape
  assert values.shape == queries.shape

  # Get sizes...
  (batch_size, sequence_length, _, _) = queries.shape

  if importance is not None:
    assert importance.ndim == 2
    assert importance.shape == (batch_size, sequence_length)

  assert window_length > 0
  if window_length >= sequence_length:
    # No sliding window.
    num_queries = sequence_length
    num_keys = sequence_length
    if prev_kvi is not None:
      num_keys += prev_kvi[0].shape[1]
  else:
    # Sliding window.
    if prev_kvi is not None:
      assert prev_kvi[0].shape[1] == window_length
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


def reshape_transpose_tree(tree: ArrayTree, sections: int, axis: int = 0) -> (
    ArrayTree):
  """Reshape and transpose arrays so that the window is dimension 0."""

  # We could use jax tree utils for this, but we do it the hard way so the
  # implementaiton can be compared with split_tree.
  if tree is None:
    return None
  elif isinstance(tree, jnp.ndarray):
    tree = typing.cast(Array, tree)  # Tell type-checker about isinstance
    ndim = tree.ndim
    wlen = tree.shape[axis] // sections
    assert sections * wlen == tree.shape[axis]  # Must be evenly divisible.

    # Break the axis dimension into sections * window_size
    arr = tree
    sh = list(arr.shape)
    nshape = sh[0:axis] + [sections, wlen] + sh[axis + 1:]
    arr = jnp.reshape(arr, nshape)

    # Transpose sections to be dimension 0.
    tdims = [axis] + list(range(0, axis)) + list(range(axis + 1, ndim + 1))
    arr = jnp.transpose(arr, tdims)
    return arr
  elif isinstance(tree, tuple):
    return tuple([reshape_transpose_tree(b, sections, axis) for b in tree])
  else:
    raise ValueError("Argument %r must be an ndarray or tuple." % tree)


def transpose_reshape_tree(tree: ArrayTree, sections: int, axis: int = 0) -> (
    ArrayTree):
  """Reshape and transpose arrays so that the window is dimension 0."""

  # We could use jax tree utils for this, but we do it the hard way so the
  # implementaiton can be compared with split_tree.
  if tree is None:
    return None
  elif isinstance(tree, jnp.ndarray):
    tree = typing.cast(Array, tree)  # Tell type-checker about isinstance
    ndim = tree.ndim - 1   # Input tree has 1 extra dimension on front.
    assert axis < ndim
    wlen = tree.shape[axis + 1]  # Window length.

    # Transpose dimension 0 back to its proper place.
    arr = tree
    tdims = list(range(1, axis + 1)) + [0] + list(range(axis + 1, ndim + 1))
    arr = jnp.transpose(arr, tdims)

    # Combine the sections and window_size dimensions.
    sh = list(arr.shape)
    nshape = sh[0:axis] + [sections * wlen] + sh[axis + 2:]
    arr = jnp.reshape(arr, nshape)
    return arr
  elif isinstance(tree, tuple):
    return tuple([transpose_reshape_tree(b, sections, axis) for b in tree])
  else:
    raise ValueError("Argument %r must be an ndarray or tuple." % tree)


def split_and_scan(fn: Callable[[ArrayTree, ArrayTree],
                                Tuple[ArrayTree, ArrayTree]],
                   carry: ArrayTree, input_arrays: ArrayTree,
                   sections: int, axis: int = 0,
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
    in_arrs = reshape_transpose_tree(input_arrays, sections, axis)
    (carry, out_arrs) = jax.lax.scan(fn, carry, in_arrs)
    output_arrays = transpose_reshape_tree(out_arrs, sections, axis)
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
