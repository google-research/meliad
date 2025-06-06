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

"""KV cache. This wraps the memory components in transformer_layer.py.
"""
import typing
from typing import Any, Optional, Sequence, Tuple, TypeVar, Generic
from absl import logging

from flax import linen as nn
import gin
import jax
import jax.numpy as jnp

from transformer import attention
from transformer import nn_components

KVITupleFlaxVars = Tuple[Any, Any, Any]
vshape = nn_components.vshape

T = TypeVar("T")


class BaseCache(nn.Module, Generic[T]):
  """Base class for caches."""

  mode: str
  supported_modes_for_cache: Sequence[str] = (
      "train",
      "test",
  )  # Modes which support caching of previous keys and values.
  enable_cache: bool = True

  def setup(self):

    cached_mem: dict[str, KVITupleFlaxVars] = {}
    for mkey in self.supported_modes_for_cache:
      cached_mem[mkey] = self._allocate_cache(mkey)
    self.cached_mem = cached_mem

  def _allocate_cache(self, mode: str) -> KVITupleFlaxVars:
    raise NotImplementedError

  def _get_cache_name_from_mode(self, mode: str) -> Tuple[str, bool]:
    """Get the name of the cache, and whether to update the cache, from mode."""
    # This is a hack to ensure that "generate" steps generate text as a
    # continuation of the text that is stored in the "test" cache,
    # but it does not update the "test" cache.
    if mode == "generate":
      assert "test" in self.supported_modes_for_cache
      return ("test", False)  # Use test cache, but don't update it.
    elif mode == "init":
      return ("train", False)  # Use training cache for initialization.
    else:
      return (mode, True)

  def _get_cache(
      self, start_of_sequence: jax.Array, mode: str
  ) -> Optional[T]:
    raise NotImplementedError

  def _set_cache(self, next_kvi: T, mode: str):
    raise NotImplementedError

  def load_prev_cache(
      self, start_of_sequence: jax.Array
  ) -> Optional[T]:
    """Load cached state that is passed from one window to the next."""

    if not self.enable_cache:
      logging.info("base-cache: Cache is disabled.")
      return None

    (mode, _) = self._get_cache_name_from_mode(self.mode)
    prev_mem = self._get_cache(start_of_sequence, mode)
    if prev_mem is not None:
      logging.info(
          "base-cache: Loaded memories for mode %s from cache %s",
          self.mode, mode
      )
      return prev_mem
    else:
      logging.info("base-cache: Skipping memory cache for mode %s.", self.mode)
      return None

  def store_next_cache(self, next_mem: Optional[T]):
    """Write window state to the cache."""

    if not self.enable_cache:
      logging.info("base-cache: Cache is disabled.")
      return

    (mode, update_cache) = self._get_cache_name_from_mode(self.mode)
    if update_cache and next_mem is not None:
      logging.info("base-cache: Storing memories for mode %s in cache %s.",
                   self.mode, mode)
      self._set_cache(next_mem, mode)
    else:
      logging.info("base-cache: Skipping memory cache update for mode %s.",
                   self.mode)


@gin.configurable
class KVICache(BaseCache[attention.KVITuple]):
  """KVI cache class. The cache is a tuple of (keys, values, importance).

  All arguments below should be passed by caller.
  """
  # Set by caller.
  batch_size: int = gin.REQUIRED
  window_length: int = gin.REQUIRED
  num_heads: int = gin.REQUIRED
  head_size: int = gin.REQUIRED
  use_importance: bool = False
  dtype: Any = jnp.float32

  def _allocate_cache(self, mode: str) -> KVITupleFlaxVars:
    """Allocate (keys, values, importance) which can be cached between steps."""
    # This function is called from setup().

    kv_shape = [self.batch_size, self.window_length,
                self.num_heads, self.head_size]
    imp_shape = [self.batch_size, self.window_length]

    def kv_initializer(shape):
      return jnp.zeros(shape, dtype=self.dtype)

    def imp_initializer(shape):
      return jnp.zeros(shape, dtype=self.dtype)

    pkeys = self.variable(
        "state", "previous_keys_" + mode, kv_initializer, kv_shape
    )
    pvals = self.variable(
        "state", "previous_values_" + mode, kv_initializer, kv_shape
    )
    if self.use_importance:
      pimportance = self.variable(
          "state", "previous_importance_" + mode, imp_initializer, imp_shape
      )
    else:
      pimportance = None
    return (pkeys, pvals, pimportance)

  def _initial_kvi(self):
    """Returns initial (zero) keys/values/i that can be passed to prev_kvi."""
    shape = (
        self.batch_size,
        self.window_length,
        self.num_heads,
        self.head_size,
    )

    z = jnp.zeros(shape, dtype=self.dtype)
    if self.use_importance:
      i = jnp.zeros(
          (shape[0], shape[1]), dtype=self.dtype
      )  # (bsize, window_length)
    else:
      i = None
    return (z, z, i)

  def _get_cache(
      self, start_of_sequence: jax.Array, mode: str
  ) -> Optional[attention.KVITuple]:
    """Returns cached (keys, values, importance) from the previous cache."""

    if mode not in self.cached_mem:
      # No cache, return zeros.
      logging.info(
          "kvi-cache: using zero as initial cache value."
      )
      return self._initial_kvi()

    # New documents start with zero_kv.
    # Continuing the same document will attend to previous keys/vals.
    logging.info("kvi-cache: window_length = %d", self.window_length)
    (pkeys, pvals, pimportance) = self.cached_mem[mode]
    (zkeys, zvals, zimportance) = self._initial_kvi()

    # Broadcast start_of_sequence over non-batch dims.
    b = self.batch_size
    start_of_sequence_kv = jnp.reshape(start_of_sequence, [b, 1, 1, 1])
    prev_keys = jnp.where(start_of_sequence_kv, zkeys, pkeys.value)
    prev_vals = jnp.where(start_of_sequence_kv, zvals, pvals.value)
    if self.use_importance:
      start_of_sequence_imp = jnp.reshape(start_of_sequence, [b, 1])
      prev_importance = jnp.where(
          start_of_sequence_imp, zimportance, pimportance.value
      )
    else:
      prev_importance = None
    logging.debug(
        "kvi-cache: start_of_sequence = %s", vshape(start_of_sequence)
    )
    logging.info("kvi-cache: prev_keys[%s] = %s", mode, vshape(prev_keys))
    logging.info(
        "kvi-cache: prev_importance[%s] = %s", mode, vshape(prev_importance)
    )
    return (prev_keys, prev_vals, prev_importance)

  def _set_cache(self, next_kvi: attention.KVITuple, mode: str):
    """Caches the last (keys, values, importance) from the current step."""
    if mode not in self.cached_mem:
      return

    (pkeys, pvals, pimportance) = self.cached_mem[mode]
    (nkeys, nvals, nimportance) = next_kvi   # From last window
    if pkeys.value.shape != nkeys.shape:
      raise ValueError(
          "Shape mismatch for keys on write to cache: "
          + f"{pkeys.value.shape} != {nkeys.shape}"
          + "\nThis could indicate that TransformerTaskConfig.sequence_length"
          " is less than TranformerLayer.window_length, which is a"
          " configuration error."
      )
    if pvals.value.shape != nvals.shape:
      raise ValueError(
          "Shape mismatch for values on write to cache: " +
          f"{pvals.value.shape} != {nvals.shape}")

    logging.info("kvi-cache: next_keys[%s] = %s", mode, vshape(nkeys))
    pkeys.value = nkeys
    pvals.value = nvals
    if self.use_importance:
      nimportance = typing.cast(jax.Array, nimportance)
      logging.info("kvi-cache: next_importance[%s] = %s", mode,
                   vshape(nimportance))
      if pimportance.value.shape != nimportance.shape:
        raise ValueError(
            "Shape mismatch for importance on write to cache: " +
            f"{pimportance.value.shape} != {nimportance.shape}")
      pimportance.value = nimportance


@gin.configurable
class MemoryCache(BaseCache[jax.Array]):
  """Memory cache.

  The cache is a tuple of (memory_size x memory_embedding_dim, None, None).

  Attributes:
    mode: The mode of the model.
    batch_size: The batch size of the model.
    memory_size: The size of the memory. e.g. 1024 for 1024-token memory.
    memory_embedding_dim: The embedding size of the memory.
    init_from_zero: Whether to initialize the memory from zero or from
      embeddings.
    dtype: The data type of the memory.
  """
  batch_size: int = gin.REQUIRED
  memory_size: int = gin.REQUIRED
  memory_embedding_dim: int = gin.REQUIRED
  init_from_zero: bool = True
  dtype: Any = jnp.float32

  def setup(self):

    super().setup()

    # Memory embeddings (if NOT initialized from zero).
    if not self.init_from_zero:
      self.memory_embeddings = self.param(
          "memory_embeddings",
          jax.nn.initializers.truncated_normal(stddev=1.0),
          (self.memory_size, self.memory_embedding_dim),
          self.dtype)

  def _allocate_cache(self, mode: str) -> KVITupleFlaxVars:
    """Allocate (mem, None) which can be cached between steps."""
    # This function is called from setup().

    mem_shape = [self.batch_size,
                 self.memory_size,
                 self.memory_embedding_dim]

    def mem_initializer(shape):
      return jnp.zeros(shape, dtype=self.dtype)

    pmems = self.variable("state", "previous_mems_" + mode,
                          mem_initializer, mem_shape)

    placeholder = None
    return (pmems, placeholder, placeholder)

  def _initial_mem_from_embedding(self):
    ms = jnp.expand_dims(self.memory_embeddings, axis=0)
    ms = jnp.tile(ms, (self.batch_size, 1, 1))
    logging.info("mem-cache: init mem from embedding = %s", vshape(ms))
    return (ms, None, None)

  def _initial_mem_from_zero(self):
    mem_shape = [self.batch_size,
                 self.memory_size,
                 self.memory_embedding_dim]
    ms = jnp.zeros(mem_shape, dtype=self.dtype)
    logging.info("mem-cache: init mem from zero = %s", vshape(ms))
    return (ms, None, None)

  def _initial_mem(self):
    logging.info("mem-cache: bs=%s, mem_size=%s, mem_emb_dim=%s",
                 self.batch_size,
                 self.memory_size,
                 self.memory_embedding_dim)

    if self.init_from_zero:
      return self._initial_mem_from_zero()
    else:
      return self._initial_mem_from_embedding()

  def _get_cache(self,
                 start_of_sequence: jax.Array,
                 mode: str) -> Optional[jax.Array]:
    """Returns cached (keys, values, importance) from the previous step."""
    if mode not in self.cached_mem:
      # No cache, return zeros or learnable embeddings.
      logging.info(
          "mem-cache: using zero/embedding as initial memory cache value."
      )
      return self._initial_mem()

    # New documents start with zeros or learnable embeddings.
    # Continuing the same document will attend to previous memory.
    logging.info("mem-cache: memory_size = %d", self.memory_size)
    (pmems, _, _) = self.cached_mem[mode]
    (zmems, _, _) = self._initial_mem()

    # Broadcast start_of_sequence over non-batch dims.
    b = self.batch_size
    start_of_sequence_mem = jnp.reshape(start_of_sequence, [b, 1, 1])
    prev_mems = jnp.where(start_of_sequence_mem, zmems, pmems.value)

    logging.info(
        "mem-cache: start_of_sequence = %s", vshape(start_of_sequence)
    )
    logging.info("mem-cache: prev_mems[%s] = %s", mode, vshape(prev_mems))

    return prev_mems

  def _set_cache(self, next_kvi: jax.Array, mode: str) -> None:
    """Caches the last (keys, values, importance) from the current step."""
    if mode not in self.cached_mem:
      return

    (pmems, _, _) = self.cached_mem[mode]
    nmems = next_kvi   # From last window

    if pmems.value.shape != nmems.shape:
      raise ValueError(
          "Shape mismatch for memories on write to cache: "
          + f"{pmems.value.shape} != {nmems.shape}"
          + "\nThis could indicate that TransformerTaskConfig.sequence_length"
          " is less than TranformerLayer.window_length, which is a"
          " configuration error."
      )

    logging.info("mem-cache: next_keys[%s] = %s", mode, vshape(nmems))
    pmems.value = nmems

