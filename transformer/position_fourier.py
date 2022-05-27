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

"""Class for Fourier relative position biases.

This implementation uses the same Fourier position encodings that are used
in the absolute position encoding.  However, instead of adding the positions
to the input, where the position vector and content vectors become entangled,
the relative encoding computes a relative position bias matrix, which is then
added to the content-based attention matrix before applying softmax.

The bias matrix is computed as follows.  First, a learned transformation is
applied to each query position, which transforms it so that it matches a set
of key positions. The relative position bias between query 'i' and key 'j' is
the dot product between the transformed position 'i', and position 'j'.

The learned transformation is designed so that the match between query and key
is a function of the relative distance between the two.  Although absolute
positions are fed as inputs, the rest of the network can't "see" the absolute
positions; it can only transform them by some relative amount.

A position vector consists of a sequence of (sin, cos) pairs, which have
geometrically increasing wavelengths that span from 2 (for the first pair
in each vector) to twice the length of the token sequence (for the last pair).
Each sin/cos pair encodes the (x, y) value of a 2D unit vector at a particular
angle.  For each sin/cos pair in the query position vector, we apply a learned
2x2 rotation matrix, which will rotate and scale the pair by some amount.

The dot product of two (sin, cos) pairs is the cosine of the angle between them.
The dot product of the query position and key position vectors is thus the sum
of such cosines. By rotating and scaling the query position, it is possible to
approximate any function over relative position as a Fourier series: a sum of
cosine waves at different wavelengths. The rotation provides phase, and the
scale provides magnitude.

Put another way, rotating the (sin, cos) pairs of a query position will compute
a relative offset from the /query/ position to some target /key/ position.
"""

from typing import Any, Optional

from flax import linen as nn
import gin
import jax.numpy as jnp
import numpy as np

from transformer import position


Array = jnp.ndarray


def _initialize_frel_rotation_matrix(rng, num_heads, vec_size):
  """Intialize the rotation matrices."""
  # Initialize each rotation matrix to the identity * scale.
  #
  # Initially scale by 1 / number of sine waves = 1/2 the position vector size.
  # With this initialization, the initial position bias terms should be
  # between -1.0 and 1.0 after the rotation matrix has been applied.
  del rng  # required for init function but unused
  scale = float(2.0 / vec_size)
  tmat_a = jnp.ones([num_heads, vec_size // 2], dtype=jnp.float32) * scale
  tmat_b = jnp.zeros([num_heads, vec_size // 2], dtype=jnp.float32)
  return jnp.concatenate([tmat_a, tmat_b], axis=1)


@gin.configurable
class RelativeFourierPositions(nn.Module):
  """A implementation of Fourier relative positions."""

  # The number of attention heads.
  num_heads: int = 8

  # The maximum number of keys to attend to.
  # The sin/cos wavelengths of the position vectors will be tuned to this max.
  max_number_of_keys: int = 1024

  # Size of the position vector. Needs to be large enough to address the keys.
  position_vector_size: int = 128

  # Data type to use for the rotation matrices.
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, num_queries: int, num_keys: int,
               offset: Optional[int] = None,
               bidirectional: bool = True) -> Array:
    """Returns relative positional attention matrix.

    If num_keys >= num_queries, e.g. for transformer XL or sliding window,
    then offset should be (num_keys - num_queries) to make the last N queries
    line up with the last N keys.  This is the default if offset is None.

    Args:
      num_queries: Number of queries.
      num_keys:    Number of keys.
      offset:      Offset of the first query with respect to the first key.
                   (See position.relative_positions() for more info.)
      bidirectional: Unused, included for compatibility.
                     Relative positions are always bidirectional.
    Returns:
      Attention matrix of shape (num_heads, num_queries, num_keys)
    """

    # Get the offset of each query with respect to each key.
    # If not specified, the last N queries line up with the last N keys.
    if offset is None:
      assert num_keys >= num_queries
      offset = num_keys - num_queries
    max_wavelength = 2 * self.max_number_of_keys

    # Compute absolute position vectors for keys.
    # Use numpy to compute these arrays statically.
    # ks : (num_keys, pvec_size)
    ks = position.position_encoding(num_keys,
                                    self.position_vector_size,
                                    offset=0,  # offset of queries wrt. keys
                                    max_wavelength=max_wavelength)

    # Compute absolute position vectors for queries.
    # qs : (num_queries, pvec_size)
    if offset >= 0 and offset + num_queries <= num_keys:
      # Query positions are a subset of the key positions.
      qs = ks[offset:offset + num_queries]
    else:
      # Query positions must be computed separately.
      qs = position.position_encoding(num_queries,
                                      self.position_vector_size,
                                      offset=offset,
                                      max_wavelength=max_wavelength)

    # Split qs into x and y coordinates for rotation.
    (qx, qy) = np.split(qs, 2, axis=-1)
    qs_xs = np.concatenate([qx, qx], axis=-1)
    qs_ys = np.concatenate([qy, qy], axis=-1)
    del qs

    # Convert from numpy to jax.
    ks = jnp.asarray(ks, dtype=self.dtype)
    qs_xs = jnp.asarray(qs_xs, dtype=self.dtype)
    qs_ys = jnp.asarray(qs_ys, dtype=self.dtype)

    # Initialize the rotation matrices to the identity.
    rotation_matrix = self.param("rotation_matrix",
                                 _initialize_frel_rotation_matrix,
                                 self.num_heads,
                                 self.position_vector_size)

    rotation_matrix = jnp.asarray(rotation_matrix, dtype=self.dtype)

    # Unpack rotatation_matrix to a set of 2x2 matrices.
    rmat1 = rotation_matrix  # [rm_a, rm_b]
    (rm_a, rm_b) = jnp.split(rotation_matrix, 2, axis=-1)
    rmat2 = jnp.concatenate([-rm_b, rm_a], axis=-1)

    # Vectors in qs consist of a set of (x,y) (e.g. sin,cos) pairs.
    # We transform each (x,y) pair with a 2D rotation matrix:
    #
    #   x' = a*x + -b*y
    #   y' = b*x + a*y
    #
    # or equivalently, x' + y'i = (a + bi)(x + yi) where i = sqrt(-1).
    #
    # For an angle theta, and scale s, a = cos(theta)*s, b = sin(theta)*s,
    # and a + bi = s*exp(i*theta). We avoid computing sin,cos by training a,b
    # directly.
    #
    # qs_xs = [x0 .. xn;   x0 .. xn]    -- layout of qs_xs
    # qs_ys = [y0 .. yn;   y0 .. yn]
    # rmat1 = [a0 .. an;   b0 .. bn]    -- layout of (a,b) values in rmat1
    # rmat2 = [-b0 .. -bn; a0 .. an]
    #
    # rot_qs: (num_heads, num_queries, pvec_size)

    # Broadcast qs over the number of heads.
    # Broadcast rmat over the number of queries.
    qs_xs = qs_xs[jnp.newaxis, ...]     # (1, num_queries, pvec_size)
    qs_ys = qs_ys[jnp.newaxis, ...]
    rmat1 = rmat1[:, jnp.newaxis, ...]  # (num_heads, 1, pvec_size)
    rmat2 = rmat2[:, jnp.newaxis, ...]
    rot_qs = ((rmat1 * qs_xs) + (rmat2 * qs_ys))

    # Compute the dot product of each position vector in ks by the rotated qs.
    #
    # The dot product of each (x, y) pair in ks, and each (x', y') in rot_qs,
    # is equal to the cosine of the angle between them, times the length
    # of (x', y').
    #
    # The angle of the cosine for each pair depends on:
    #   - The distance between the key and the query, divided by the wavelength.
    #     (From the initial position encoding for ks and qs).
    #   - The rotation performed by (a,b).
    #
    # The length of (x', y') is equal to the scale of (a, b).
    #
    # The dot product of two complete position vectors is the sum of the
    # cosines for all pairs.  The cosines form a progression of geometrically
    # increasing wavelengths, and each wave has a scale and phase provided by
    # the rotation matrix.  The sum of such waves can thus approximate any
    # function of position.
    #
    # pbias: (num_heads, num_queries, num_keys)
    pbias = jnp.einsum("hqd,kd->hqk", rot_qs, ks)

    # Add batch dimension; --> shape (1, num_heads, num_queries, num_keys)
    pbias = jnp.expand_dims(pbias, 0)
    return pbias.astype(self.dtype)

