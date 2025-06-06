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

"""FLAX layers for on-TPU memory."""

import abc
import functools
from typing import Callable, Sequence, Tuple, TypeVar, Union

from flax import linen
import gin
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np  # use with care!

Shape = Sequence[int]
Dtype = jnp.dtype
Array = jnp.ndarray

Axes = Union[int, Tuple[int, ...]]
F = TypeVar('F', bound=Callable)


class MemoryLayer(linen.Module, metaclass=abc.ABCMeta):
  """Internal interface for memory layers without batch dim.

  See BatchedMemory for a layer that can be used in Flax models.
  """
  num_datasets: int

  @abc.abstractmethod
  def update(self, key: Array, value: Array) -> int:
    """Adds key/value pairs to memory.

    Args:
      key: of shape (num_kv, num_datasets, k_features)
      value: of shape (num_kv, num_datasets, v_features)

    Returns:
      Dummy value so that TPU operations can wait for the update to finish if
      desired.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def topk_retrieval(self, query: Array,
                     num_neighbors: int) -> Tuple[Array, Array]:
    """Retrieves the nearest neighbors for each query.

    Args:
      query: of shape (num_queries, num_datasets, k_features)
      num_neighbors: int indicating the number of neighbors to retrieve

    Returns:
      Tuple of selected keys and selected values of shapes
      (num_queries, num_datasets, num_neighbors, k_features), and
      (num_queries, num_datasets, num_neighbors, v_features)
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def reset(self, datasets: Array) -> int:
    """Reset some or all of the datasets in the memory.

    Args:
      datasets: A vector of shape (num_datasets) of type bool. Each position
        indicates whether the dataset with the same index should be reset.

    Returns:
      Dummy value so that TPU operations can wait for the update to finish if
      desired.
    """
    raise NotImplementedError()

  def __call__(self, query, num_neighbors):
    self.topk_retrieval(query, num_neighbors)


def _target_dimensions(shape: Shape,
                       source_dimensions: Sequence[int]) -> Sequence[int]:
  target_dimensions = range(-2, -2 - len(source_dimensions), -1)
  assert len(source_dimensions) == len(target_dimensions)
  return sorted(d % len(shape) for d in target_dimensions)


def _rearrange_dimensions_shapes(
    shape: Shape, split_dimensions: Sequence[int]) -> Tuple[Shape, Shape]:
  split_shape = tuple(shape[d] for d in split_dimensions)
  remaining_shape = tuple(
      shape[d] for d in range(len(shape)) if d not in split_dimensions)
  batch_shape = remaining_shape[:-1]
  return split_shape, batch_shape


def _rearrange_dimensions(x: Array, split_dimensions: Sequence[int]) -> Array:
  """Rearrange array so that we can split by a single dimension.

  Turns an array of shape [d1, ..., dn, features] and a list of dimensions to
  split by into [prod(remaining_dimensions), prod(split_dimensions),
  features]

  Args:
    x: array of shape [d1, ..., dn, features]
    split_dimensions: list of dimensions that should end up in dimension -2.

  Returns:
    Rearranged array as described above.
  """
  split_dimensions = [d % len(x.shape) for d in split_dimensions]
  split_dimensions = sorted(split_dimensions)
  split_shape, batch_shape = _rearrange_dimensions_shapes(
      x.shape, split_dimensions)

  target_dimensions = _target_dimensions(x.shape, split_dimensions)
  x = jnp.moveaxis(x, split_dimensions, target_dimensions)
  assert len(x.shape) > len(split_dimensions)
  assert all(isinstance(d, int) and d >= 0 for d in batch_shape)
  assert all(isinstance(d, int) and d >= 0 for d in split_shape)
  new_shape = [
      # The use of numpy is okay here, since shapes are concrete at jit time.
      np.prod(batch_shape),
      np.prod(split_shape),
      x.shape[-1]  # features dimension
  ]
  res = x.reshape(new_shape)
  assert res.ndim == 3
  return res


def _restore_dimensions(x: Array, original_shape: Shape,
                        split_dimensions: Sequence[int]) -> Array:
  """Restores arrays encoded with _rearrange_dimensions.

  Args:
    x: Array of shape [prod(batch_shape), prod(split_shape), feature...]
    original_shape: Shape of the array to restore to.
    split_dimensions: Dimensions that were multiplied into dimension 2.

  Returns:
    Array of the original shape and axis order for all dimensions in batch_shape
    and split_shape. Feature dimensions may have changed (can include additional
    dimensions for neighbors, for example).
  """
  split_dimensions = [d % len(original_shape) for d in split_dimensions]
  split_dimensions = sorted(split_dimensions)
  split_shape, batch_shape = _rearrange_dimensions_shapes(
      original_shape, split_dimensions)

  features_shape = x.shape[2:]
  x = x.reshape((*batch_shape, *split_shape, *features_shape))

  # rearrange
  target_dimensions = _target_dimensions(original_shape, split_dimensions)
  x = jnp.moveaxis(x, target_dimensions, split_dimensions)
  return x


@gin.configurable
class BatchedMemory(linen.Module):
  """Equips a memory module with a batch dimension."""

  # We wrap this linen.Module:
  wrapped: MemoryLayer

  # `split_dimensions` indicates the dimensions of the query and update tensors
  # that will go to separate databases. By default, we use a separate database
  # for each head.
  # Note that some implementations of the memory share memory across all hosts
  # and devices (memory_on_borg, unless configured otherwise) or just across
  # devices of each host (memory_on_host).
  # Default is (-2,) to split by head only; use (0, -2) to also slit by batch
  # dimensions.
  split_dimensions: Tuple[int, ...] = (-2,)

  query_stride: int = 1
  update_stride: int = 1

  def update(self, key: Array, value: Array):
    """Adds key/value pairs to memory.

    Args:
      key: typically of shape (batch, kv_len, num_heads, k_features). This
        tensor is split up into datasets according to `split_dimensions`.
      value: typically of shape (batch, kv_len, num_heads, v_features). This
        tensor is split up into datasets according to `split_dimensions`.

    Returns:
      A dummy value 0, once the operation has completed.
    """
    if key.ndim != 4 or value.ndim != 4:
      raise ValueError('Expected batched inputs; got shapes: %s and %s.' %
                       (key.shape, value.shape))
    key = _rearrange_dimensions(key, self.split_dimensions)
    value = _rearrange_dimensions(value, self.split_dimensions)
    update_stride = self.update_stride
    if update_stride == 1:
      return self.wrapped.update(key, value)
    return self.wrapped.update(key[update_stride - 1::update_stride, ...],
                               value[update_stride - 1::update_stride, ...])

  def topk_retrieval(self, query: Array, num_neighbors: int):
    """Retrieves the nearest neighbors for each query.

    Args:
      query: typically of shape (batch, q_len, num_heads, k_features). This
        tensor is split up into datasets according to `split_dimensions`.
      num_neighbors: number of neighbors to retrieve

    Returns:
      Tuple of tensors with the retrieved keys and value of the same shape as
      query, but with an extra dimension of length num_neighbors - typically:
      (batch, q_len, num_heads, num_neighbors, k_features)
    """
    if query.ndim != 4:
      raise ValueError('Expected batched inputs; got shape: %s.' % query.shape)
    query_stride = self.query_stride
    original_shape = query.shape
    query = _rearrange_dimensions(query, self.split_dimensions)
    if query_stride == 1:
      key, value = self.wrapped.topk_retrieval(query, num_neighbors)
    else:
      num_queries, num_heads, k_features = query.shape
      throttled_query = query[0::query_stride, ...]
      key = jnp.zeros(
          shape=(num_queries, num_heads, num_neighbors, k_features),
          dtype=query.dtype)
      throttled_key, throttled_value = (
          self.wrapped.topk_retrieval(throttled_query, num_neighbors))
      _, _, _, v_features = throttled_value.shape
      value = jnp.zeros(
          shape=(num_queries, num_heads, num_neighbors, v_features),
          dtype=query.dtype)
      key = key.at[0::query_stride, ...].set(throttled_key)
      value = value.at[0::query_stride, ...].set(throttled_value)
    key = _restore_dimensions(key, original_shape, self.split_dimensions)
    # Note that `original_shape` here may have the wrong feature dimension (if
    # k_features != v_features. But `_restore_dimensions` does not depend on
    # that dimension and the tests cover this case.
    value = _restore_dimensions(value, original_shape, self.split_dimensions)
    assert key.ndim == len(original_shape) + 1
    return key, value

  def reset(self, datasets: Array) -> int:
    """Resets the memory.

    Args:
      datasets: of shape (num_datasets,), typically the same as (num_heads,).

    Returns:
      A dummy value 0, once the operation has completed.
    """
    return self.wrapped.reset(datasets)


@functools.partial(jax.jit, static_argnames=('num_buckets', 'bucket_size'))
def _chunking_sparsify(query: Array, key: Array, num_buckets: int,
                       bucket_size: int) -> Tuple[Array, Array, Array]:
  """Approximate top k operation for a single head."""
  # q = q_length, f = qk features, d = database_size
  scores = jnp.einsum('qf,df->qd', query, key)
  mask = (key.sum(-1) == 0).astype(jnp.bfloat16) * -1e6
  scores += mask

  num_queries, _ = scores.shape
  reshaped_scores = jnp.reshape(scores, (num_queries, bucket_size, num_buckets))

  sparse_scores = linen.softmax(reshaped_scores * 1e6, axis=1)

  # topk_scores and topk_indices will only be computed if we depend on their
  # results.
  topk_scores = jnp.max(reshaped_scores, axis=1)
  local_indices = jnp.argmax(reshaped_scores, axis=1)
  topk_indices = (
      local_indices * num_buckets + jnp.arange(num_buckets).reshape(
          (1, num_buckets)))
  return sparse_scores, topk_scores, topk_indices


def _retrieve_topk_gatherless(
    query: Array, key: Array, value: Array,
    num_neighbors: int) -> Tuple[Array, Array, Array, Array]:
  """Retrieves for a single head - used to simplify array accesses."""
  num_kv, query_features = query.shape
  database_size, key_features = key.shape
  _, value_features = value.shape
  assert query_features == key_features
  num_buckets = num_neighbors
  if num_buckets > database_size:
    raise ValueError('More buckets than items in database. %s > %s' %
                     (num_buckets, database_size))
  if database_size % num_buckets:
    raise ValueError('Buckets must divide database: %s %% %s.' %
                     (database_size, num_buckets))
  bucket_size = database_size // num_buckets

  sparse_scores, topk_scores, topk_indices = _chunking_sparsify(
      query, key, num_buckets, bucket_size)
  key = key.reshape(bucket_size, num_buckets, key_features)
  value = value.reshape(bucket_size, num_buckets, value_features)
  selected_keys = jnp.einsum('qbn,bnd->qnd', sparse_scores, key)
  selected_values = jnp.einsum('qbn,bnd->qnd', sparse_scores, value)

  assert selected_keys.shape == (num_kv, num_neighbors, key_features)
  assert selected_values.shape == (num_kv, num_neighbors, value_features)
  return selected_keys, selected_values, topk_scores, topk_indices


class MemoryOnTpu(MemoryLayer):
  """Approximate top K search on TPU."""
  # database_size must be integer multiple of prod(batch_dims) * num_neighbors.
  database_size: int
  dtype: Dtype = jnp.float32  # pylint: disable=g-bare-generic
  key_features: int = 64
  value_features: int = 64
  report_scores_and_indices: bool = False
  disallow_reset_because: str = ''

  def setup(self):
    self.db_index = self.variable('database', 'database_index',
                                  functools.partial(jnp.zeros, dtype=jnp.int32),
                                  (self.num_datasets,))
    self.key_db = self.variable(
        'database', 'key_db', functools.partial(jnp.zeros, dtype=self.dtype),
        (self.num_datasets, self.database_size, self.key_features))
    self.value_db = self.variable(
        'database', 'value_db', functools.partial(jnp.zeros, dtype=self.dtype),
        (self.num_datasets, self.database_size, self.value_features))

    self.retrieved_indices = self.variable(
        'database', 'retrieved_indices',
        functools.partial(jnp.zeros, dtype=jnp.int32), (0, 0, 0))
    self.retrieved_indices_scores = self.variable(
        'database', 'retrieved_indices_scores',
        functools.partial(jnp.zeros, dtype=jnp.float32), (0, 0, 0))

  def _update_kv_database(self, database, new_values, start_index):
    num_datasets, database_size, _ = database.shape
    assert database_size == self.database_size, f'{database_size} vs {self.database_size}'
    assert num_datasets == self.num_datasets
    assert new_values.ndim == 3
    assert start_index.shape == (self.num_datasets,)

    def _update(database, new_values, start_index):
      return lax.dynamic_update_slice(
          database, new_values, start_indices=(start_index, 0))

    return jax.vmap(
        _update, in_axes=(0, 0, 0), out_axes=0)(database, new_values,
                                                start_index)

  def update(self, key: Array, value: Array) -> int:
    """Add keys and values to the memory; overwrite oldest if memory is full."""
    key = lax.stop_gradient(key)
    value = lax.stop_gradient(value)
    assert len(key.shape) == len(value.shape)
    assert key.shape[:-1] == value.shape[:-1]
    num_kv, num_datasets, key_features = key.shape
    assert num_datasets == self.num_datasets
    assert key_features == self.key_features
    assert value.shape[-1] == self.value_features
    assert self.database_size % num_kv == 0, (
        'Database size must be integer multiple of num_kv.')
    key = jnp.moveaxis(key, source=1, destination=0)  # split by dataset
    value = jnp.moveaxis(value, source=1, destination=0)  # split by dataset

    # start_index can be larger than DB - we use that to detect which entries
    # are not written to yet
    start_index = self.db_index.value % self.database_size
    self.key_db.value = self._update_kv_database(self.key_db.value, key,
                                                 start_index)
    self.value_db.value = self._update_kv_database(self.value_db.value, value,
                                                   start_index)
    self.db_index.value = self.db_index.value + num_kv
    return 0

  def topk_retrieval(self, query: Array,
                     num_neighbors: int) -> Tuple[Array, Array]:
    """Nearest neighbors by full multiplication and approximate top k on TPU."""
    query = lax.stop_gradient(query)
    unused_num_kv, num_datasets, query_features = query.shape
    assert num_datasets == self.num_datasets, (
        f'{num_datasets=} vs {self.num_datasets=}')
    assert query_features == self.key_features, (
        f'{query_features=} vs {self.key_features=}')
    query = jnp.moveaxis(query, source=1, destination=0)

    # Process different heads sequentially
    selected_keys, selected_values, topk_scores, topk_indices = lax.map(
        lambda x: _retrieve_topk_gatherless(*x, num_neighbors),
        (query, self.key_db.value, self.value_db.value))

    if self.report_scores_and_indices:
      # TODO(mrabe): These variable updates may not work perfectly yet. Find out
      # why Flax does not like them.
      self.retrieved_indices.value = topk_indices
      self.retrieved_indices_scores.value = topk_scores

    assert selected_keys.ndim == selected_values.ndim == 4
    selected_keys = jnp.moveaxis(selected_keys, source=0, destination=1)
    selected_values = jnp.moveaxis(selected_values, source=0, destination=1)
    return selected_keys, selected_values

  def reset(self, datasets: Array) -> int:
    """Resets specified datasets."""
    if self.disallow_reset_because:
      raise ValueError(
          f'Error on reset. Explanation: {self.disallow_reset_because}')
    datasets = lax.stop_gradient(datasets)
    assert datasets.shape == (self.num_datasets,)
    assert datasets.dtype == jnp.bool_

    def _reset_single_dataset(input_tuple):
      """Resets a single head; reset is a single bool."""
      database, reset = input_tuple
      assert reset.shape == tuple(), reset.shape
      assert reset.dtype == jnp.bool_
      return database * (1 - reset)

    self.db_index.value = self.db_index.value * (1 - datasets)
    self.key_db.value = lax.map(
        _reset_single_dataset, xs=(self.key_db.value, datasets))
    self.value_db.value = lax.map(
        _reset_single_dataset, xs=(self.value_db.value, datasets))
    return 0
