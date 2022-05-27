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

"""Functions for dealing with relative and absolute positions, and masks."""

from typing import Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np


Array = jnp.ndarray
NpArray = np.ndarray
Dtype = Union[jnp.dtype, str]


def relative_positions(num_queries: int, num_keys: int,
                       offset: Optional[int] = None):
  """Returns an jax array of relative positions between query and key.

  If num_keys >= num_queries, e.g. for transformer XL or sliding window,
  then offset should be (num_keys - num_queries) to make the last N queries
  line up with the last N keys.  This is the default if offset is None.

  Args:
      num_queries: Number of queries.
      num_keys:    Number of keys.
      offset:      Offset of the first query wrt. the first key.

  Returns:
    A /jax/ array of shape [num_queries, num_keys] with the signed distance
    from each query to each key.
  """

  # Get the offset of each query wrt. to each key.
  # If not specified, assume the last N queries line up with the last N keys.
  if offset is None:
    if num_keys < num_queries:
      raise ValueError("Number of keys %d must be greater than queries %d" %
                       (num_keys, num_queries))
    offset = num_keys - num_queries
  qidx = jnp.arange(0, num_queries, dtype=jnp.int32).reshape(num_queries, 1)
  kidx = jnp.arange(0, num_keys, dtype=jnp.int32).reshape(1, num_keys)
  return kidx - (qidx + offset)


def relative_positions_np(num_queries: int, num_keys: int,
                          offset: Optional[int] = None):
  """Returns a numpy array of relative positions between query and key.

  If num_keys >= num_queries, e.g. for transformer XL or sliding window,
  then offset should be (num_keys - num_queries) to make the last N queries
  line up with the last N keys.  This is the default if offset is None.

  Args:
      num_queries: Number of queries.
      num_keys:    Number of keys.
      offset:      Offset of the first query wrt. to the first key.

  Returns:
    A /numpy/ array of shape [num_queries, num_keys] with the signed distance
    from each query to each key.
  """

  # Get the offset of each query wrt. to each key.
  # If not specified, assume the last N queries line up with the last N keys.
  if offset is None:
    if num_keys < num_queries:
      raise ValueError("Number of keys %d must be greater than queries %d" %
                       (num_keys, num_queries))
    offset = num_keys - num_queries
  qidx = np.arange(0, num_queries, dtype=np.int32).reshape(num_queries, 1)
  kidx = np.arange(0, num_keys, dtype=np.int32).reshape(1, num_keys)
  return kidx - (qidx + offset)


def broadcast_mask(mask: Array, attn: Array):
  """Broadcast a mask or bias over all the dimensions of attn."""

  # Add leading dimensions for batch_size, num_heads if necessary.
  if mask.ndim < attn.ndim:
    mask = jnp.expand_dims(mask, axis=tuple(range(0, attn.ndim - mask.ndim)))
  return mask


def causal_mask(num_queries: int, num_keys: int, window_length: int = 0):
  """Returns a causal mask of the same shape as attn."""

  # The mask ranges over the window_length positions prior to each query.
  if window_length == 0:
    window_length = num_queries

  kqpos = relative_positions(num_queries, num_keys)  # 2D mask

  # The causal mask includes only those tokens *before* the current token.
  # This slightly improves perplexity in practice, and simplifies generation.
  # Each token attends to exactly window_length prior tokens.
  mask = (kqpos < 0) & (kqpos >= -window_length)
  return mask


def position_encoding(num_positions: int,
                      input_dim: int,
                      *,
                      offset: int = 0,
                      max_wavelength: float = 0) -> NpArray:
  """Returns a position encoding of shape (num_positions, input_dim).

  Positions are encoded as sin/cos pairs at geometrically increasing
  wavelengths.

  The length of a half-wave (peak to trough) increases geometrically from 1 to
  max_wavelength.  (Technically, it's slightly less; the last sin/cos pair has
  a wavelength of max_wavelength**((d-1)/d), where d = input_dim/2.)

  NOTE: unlike prior published position encodings, we multiply the position of
  each token by pi to convert from fractions of a wave (position/wavelength)
  to radians.  Thus, the highest frequency wave alternates between -1 and 1 on
  every token, whereas in prior published work the highest frequency alternates
  between -1 and 1 every pi tokens.  The max_wavelength is also effectively
  1/pi times as long, so a prior published factor of 10,000
  (e.g. https://arxiv.org/abs/1706.03762) would equate to a max_wavelength
  of 31,416.

  This encoding also does not alternate between sin/cos values, but puts all of
  the cos values on one side, and the sin values on the other.  That makes it
  easier to split the sin,cos values to construct or apply a rotation matrix.

  The default value for max_wavelength is 2 * num_positions.

  Args:
    num_positions:  The number of positions.
    input_dim:      The dimension of the position vector.
    *:  --- The following are keyword arguments only. ---
    offset:         Positions count from offset to (offset + num_positions).
    max_wavelength: The maximum length of a half-wave (peak to trough)

  Returns:
    Numpy matrix of shape (num_positions, input_dim) containing the encodings.
    Position encodings are packed as concat(cos_values, sin_values, axis=1).
  """

  if max_wavelength == 0:
    max_wavelength = 2 * num_positions
  assert max_wavelength > 1

  assert (input_dim % 2) == 0
  idim2 = input_dim // 2

  # t ranges from 0 <= t < 1
  t = np.arange(0, idim2, dtype=np.float32) / idim2

  # wavelength (columns)
  # The length of a half-wave (trough to peak) increases geometrically
  # 1 <= wavelength < max_wavelength
  wavelength = float(max_wavelength)**t
  wavelength = np.reshape(wavelength, (1, idim2))  # broadcast over rows

  # k is the position in the sequence (rows)
  k = np.arange(offset, num_positions + offset, dtype=np.float32)
  k = np.reshape(k, (num_positions, 1))  # broadcast over columns

  # For each position (row) compute an angle (column) at various wavelengths.
  # NOTE: unlike prior published work, we multiply by pi to convert to radians.
  pi_f = np.array(np.pi, dtype=np.float32)
  angles = pi_f * k / wavelength  # shape (num_positions, idim2)
  posx = np.cos(angles, dtype=np.float32)
  posy = np.sin(angles, dtype=np.float32)
  return np.concatenate([posx, posy], axis=1)  # shape (num_positions, idim)


def rotate_kq(keys: Array, queries: Array,
              *,  # the following args must be passed by keyword.
              max_wavelength: float,
              offset: Optional[int] = None,
              dtype: Optional[Dtype] = None) -> Tuple[Array, Array]:
  """Rotate keys and queries by the relative distance between query and key.

  Implements rotary position embeddings (RoPE) https://arxiv.org/abs/2104.09864.

  Args:
    keys: array of shape (batch_size, num_keys, num_heads, head_size)
    queries: aray of shape (batch_size, num_queries, num_heads, head_size)
    max_wavelength: The maximum length of a half-wave (peak to trough)
    offset: The relative positional offset from keys[i] to queries[i].
            Defaults to num_keys - num_queries if not specified.
    dtype: The precision to perform the rotation at.
           Defaults to keys.dtype.

  Returns:
    (keys, queries) after rotation.
  """

  (batch_size, num_keys, num_heads, head_size) = keys.shape
  (_, num_queries, _, _) = queries.shape
  assert queries.shape == (batch_size, num_queries, num_heads, head_size)

  if offset is None:
    assert num_keys >= num_queries
    offset = num_keys - num_queries

  if dtype is None:
    dtype = keys.dtype

  def rotate_k_or_q(kq: Array, num_kq: int, kq_offset: int) -> Array:
    nonlocal max_wavelength
    nonlocal dtype

    # Get position encodings, which can be used to do a rotation.
    kq_pos = position_encoding(num_kq, head_size, offset=kq_offset,
                               max_wavelength=max_wavelength)
    # Broadcast over batch_size and num_heads.
    kq_pos = np.reshape(kq_pos, (1, num_kq, 1, head_size))
    # Split position encoding into separate sin/cos values in order to
    # construct a rotation matrix.
    (cosa, sina) = np.split(kq_pos, 2, axis=-1)
    cosa = jnp.asarray(cosa, dtype=dtype)  # convert from numpy -> jax
    sina = jnp.asarray(sina, dtype=dtype)  # convert from numpy -> jax

    # Split keys/queries into real & imaginary (i.e. x & y) parts.
    (kqx, kqy) = jnp.split(kq, 2, axis=-1)
    # Apply rotation matrix.
    kqx_rot = (kqx * cosa) - (kqy * sina)
    kqy_rot = (kqx * sina) + (kqy * cosa)
    # Concatenate back into keys/queries.
    return jnp.concatenate([kqx_rot, kqy_rot], axis=-1)

  keys = rotate_k_or_q(keys, num_keys, -offset)  # pylint: disable=invalid-unary-operand-type
  queries = rotate_k_or_q(queries, num_queries, 0)
  return (keys, queries)

