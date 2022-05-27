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

"""Helper routines for recording various training metrics."""

from typing import Any
import jax.numpy as jnp


Array = Any


def compute_accuracy_sum(logits, targets, valid_loss_mask=None):
  """Compute accuracy for logits and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   valid_loss_mask: None or array of shape bool[batch, length]

  Returns:
    The number of correct tokens in the output.
  """
  if logits.shape[:-1] != targets.shape:
    raise ValueError("Incorrect shapes. Got shape %s logits and %s targets" %
                     logits.shape, targets.shape)
  if valid_loss_mask is not None and valid_loss_mask.shape != targets.shape:
    raise ValueError("Incorrect shapes. Got shape %s targets and %s mask" %
                     targets.shape, valid_loss_mask.shape)

  accuracy = jnp.equal(jnp.argmax(logits, axis=-1), targets)
  if valid_loss_mask is not None:
    accuracy = jnp.logical_and(accuracy, valid_loss_mask)
  return jnp.sum(accuracy)  # Sum of the number of True values.


def reshape_image(image):
  """Reshape image to something that tensorboard recognizes.

  Args:
    image: Array of shape [xsize, size] or [num_images, xsize, ysize]

  Returns:
    Array of shape [num_images, xsize, ysize, 1]
  """

  # Reshape to [num_images, xdim, ydim, rgb] for tensorboard.
  sh = image.shape
  if image.ndim == 2:
    return jnp.reshape(image, [1, sh[0], sh[1], 1]).astype(jnp.float32)
  elif image.ndim == 3:
    return jnp.reshape(image, [sh[0], sh[1], sh[2], 1]).astype(jnp.float32)
  else:
    return None  # Not an image.


def normalize_image(images: Array, as_group: bool = False) -> Array:
  """Rescale the values in images to between 0.0 and 1.0.

  Args:
    images:   Array of size [batch_size, xsize, ysize]
    as_group: Scale all images in the batch by the same amount if True.

  Returns:
    A rescaled image of the same shape.
  """

  images = images.astype(jnp.float32)  # Return images as float32.
  if as_group:
    # Normalize the batch of images as a group.
    min_img = jnp.min(images)
    max_img = jnp.max(images)
  else:
    # Normalize each image in the batch individually.
    min_img = jnp.min(images, axis=(-2, -1), keepdims=True)
    max_img = jnp.max(images, axis=(-2, -1), keepdims=True)
  norm_image = (images - min_img) / (max_img - min_img + 1e-6)
  return jnp.where(jnp.isfinite(norm_image), norm_image, 0.0)


def overlay_images(image1: Array, image2: Array) -> Array:
  """Place image1 on top of image2, broadcasting image2 if necessary.

  Args:
    image1: array of shape [num_images, xsize, ysize]
    image2: array of shape [num_images, xsize, ysize]

  Returns:
    A combined image.
  """

  assert image1.ndim == 3  # (num_images, xsize, ysize)
  assert image2.ndim == 3
  image2 = jnp.broadcast_to(image2, image1.shape)
  return jnp.concatenate([image1, image2], axis=1)


def make_histograms(viz_dicts):
  """Generate image histograms."""
  hist_dict = {}
  for (i, viz_dict) in enumerate(viz_dicts):
    for (k, images) in viz_dict.items():
      hist_dict["h_" + k + "_" + str(i)] = images
  return hist_dict
