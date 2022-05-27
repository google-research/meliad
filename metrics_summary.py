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

"""Class to handle summarizing of metrics over multiple training steps."""

import abc
from typing import Any, Dict, Mapping, Optional, Tuple, Union
from absl import logging
from clu import metric_writers
import gin
import jax
from jax import numpy as jnp
import numpy as np


Array = Union[jnp.ndarray, np.ndarray]


class Aggregator(abc.ABC):  # Superclass for type checks

  @abc.abstractmethod
  def add(self, value: Any):
    pass

  @abc.abstractmethod
  def is_valid(self) -> bool:
    pass

  @abc.abstractmethod
  def to_value(self):
    pass


class _MeanAggregator(Aggregator):
  """Maintains the mean of incoming values."""
  mean: float = 0.0
  weight: float = 0.0

  def add(self, new_value: Any):
    """Aggregates a new value into the mean."""
    if np.ndim(new_value) == 0:  # is a scalar; works with int, float, Array
      val, weight = new_value, 1.0  # assuming weight 1 by default
    else:
      val, weight = new_value
    if weight < 0.0:
      raise ValueError("Adding value with negative weight.")
    total_weight = self.weight + weight
    if total_weight != 0.0 and weight > 0.0:
      delta = (val - self.mean) * weight / total_weight
      self.mean += delta
      self.weight = total_weight

  def is_valid(self) -> bool:
    return self.weight > 0.0

  def to_value(self):
    assert self.weight > 0.0
    return self.mean


class _SumAggregator(_MeanAggregator):
  # We aggregate sum and mean in the same way as a tuple of the form:
  # (weighted mean, total weights). "sum" can then be computed by
  # multiplying the two values.

  def is_valid(self) -> bool:
    return True

  def to_value(self):
    return self.mean * self.weight


class _LastAggregator(Aggregator):
  """Remembers the last value given."""
  last_value: Optional[float] = None

  def add(self, new_value: Any):
    self.last_value = new_value

  def is_valid(self) -> bool:
    return self.last_value is not None

  def to_value(self):
    assert self.last_value is not None
    return self.last_value


@gin.configurable
class MetricsSummary:
  """Summarizes a set of a metrics over multiple training steps."""

  def __init__(self,
               metric_types: Mapping[str, str],
               upscale_images: bool = True,
               remove_outliers: bool = False):
    """Creates a MetricSummarizer.

    Args:
      metric_types: Map from metrics to the type of summary.  Types are:
         "mean" = Compute the cumulative moving average.
         "sum" =  Compute the sum.
         "last" = No summary, just return the last value.
      upscale_images: Upscale small images for easier viewing.
      remove_outliers: Remove outliers from histograms.
    """
    self.metric_dict = {}  # type: Dict[str, Aggregator]
    self.text_dict = {}
    self.metric_types = metric_types
    self.upscale_images = upscale_images
    self.remove_outliers = remove_outliers
    self.constructor_map = {
        "mean": _MeanAggregator,
        "sum": _SumAggregator,
        "last": _LastAggregator,
    }
    logging.debug("Registered metrics: %r", metric_types)

  def current_metric_dict(self) -> Mapping[str, Aggregator]:
    return self.metric_dict

  def _is_image(self, image: Array) -> bool:
    if image.ndim != 4:
      return False
    # Greyscale or RGB image.
    return image.shape[-1] == 1 or image.shape[-1] == 3

  def _upscale_image(self, image: Array) -> Array:
    """Upscale small images to more pixels, for easier viewing."""
    if not self.upscale_images:
      return image
    assert image.ndim == 4  # (num_images, ysize, xsize, num_channels)
    ys = image.shape[1]
    xs = image.shape[2]
    if xs > 512 or ys > 512:
      return image   # No scaling.
    elif xs > 256 or ys > 256:
      scale = 2
    else:
      scale = 4
    yidx = np.arange(ys * scale) // scale
    xidx = np.arange(xs * scale) // scale
    scaled_image = image[:, yidx, :, :][:, :, xidx, :]
    return scaled_image

  def _remove_outliers(self, v, std_range: float = 4):
    if not self.remove_outliers:
      return v
    v_mean = np.mean(v)
    v_std = np.std(v)
    return np.where(np.abs(v) > (v_std * std_range), v_mean, v)

  @staticmethod
  def merge_replicated_metrics(device_metrics: Mapping[str, Any],
                               metric_types: Mapping[str, str]):
    """Merge metrics across devices by psum over "batch" axis.

    Args:
      device_metrics: dictionary of device metrics.
      metric_types: map from the metric name to { "mean", "sum" }

    Returns:
      A dictionary of metrics.
    """
    logging.info("Merging metrics across devices %r: ",
                 [(k, metric_types[k] if k in metric_types else None)
                  for k in device_metrics.keys()])

    def aggregate_sum(value: Array) -> Array:
      assert not isinstance(value, tuple), (
          "Weighted sums are not supported when aggregating over devices.")
      return jax.lax.psum(value, axis_name="batch")

    def aggregate_mean(value: Array, weight: Array) -> Tuple[Array, Array]:
      weighted_value = value * weight
      weighted_value = jax.lax.psum(weighted_value, axis_name="batch")
      weight = jax.lax.psum(weight, axis_name="batch")
      return weighted_value / (weight + 1.0e-6), weight

    aggregated_metrics = dict(device_metrics)
    for k, value in aggregated_metrics.items():
      if k not in metric_types:
        # If no metric type is given, metric remains untouched.
        continue
      if metric_types[k] == "sum":
        aggregated_metrics[k] = aggregate_sum(value)
      elif metric_types[k] == "mean":
        if not isinstance(aggregated_metrics[k], tuple):
          logging.info("Metric '%s' has no weight; assuming 1.0.", k)
          value = (value, jnp.array(1.0))
        aggregated_metrics[k] = aggregate_mean(*value)
      else:
        raise ValueError("Can only aggregate 'sum' and 'mean' over devices. "
                         f"Got {metric_types[k]}.")
    return aggregated_metrics

  def _new_aggregator(self, key) -> Aggregator:
    if key in self.metric_types:
      return self.constructor_map[self.metric_types[key]]()
    else:
      # TODO(mrabe): The default to last_value is not obvious. Force all metric
      # types to be given explicitly.
      logging.debug("No metric type for accumulator: %s", key)
      return _LastAggregator()

  def add(self, metrics: Mapping[str, Any]):
    """Add metrics from the current training step to the summary.

    Args:
      metrics: Dictionary of metrics.
    """
    for k, new_value in metrics.items():
      if k not in self.metric_dict:
        self.metric_dict[k] = self._new_aggregator(k)
      self.metric_dict[k].add(new_value)

  def add_text(self, text_metrics: Mapping[str, str]):
    """Add text metrics from the current step to the summary."""
    for (k, v) in text_metrics.items():
      self.text_dict[k] = str(v)

  def empty(self):
    """Return true if there are no summaries to write."""
    return not (self.metric_dict or self.text_dict)

  def clear(self):
    """Clear acculumated summaries."""
    self.metric_dict = {}
    self.text_dict = {}

  def write(self, writer: metric_writers.MetricWriter, step: int, prefix: str):
    """Write metrics using summary_writer, and clear all summaries."""
    if self.empty():
      return

    # Special logic for organizing metrics under tensorboard.
    # Tensorboard has top-level groups, but doesn't have subgroups.
    # Scalars are put into separate top-level groups for easier viewing.
    # e.g. all scalars in "train", "test", etc.
    # For images, each set of images should be a different top-level group,
    # otherwise all images will get tossed into a single group under,
    # e.g. "generate".
    if prefix:
      s_prefix = prefix + "/"
      i_prefix = prefix + "_"
    else:
      # Each prefix is stored in a separate subdirectory already.
      s_prefix = ""
      i_prefix = ""

    # Split metrics into different types.
    scalars = {}
    images = {}
    histograms = {}
    text_dict = {}

    # Sort metrics into scalars, images, text, and histograms.
    for k, aggregator in self.metric_dict.items():
      if not isinstance(aggregator, Aggregator):
        raise ValueError("Internal error: metric_dict should contain only "
                         "_Aggregator objects; contained %s" % aggregator)
      if not aggregator.is_valid():
        raise ValueError(f"No valid value for metric {k}.")

      v = aggregator.to_value()

      s_key = s_prefix + k
      i_key = i_prefix + k

      finite_mask = np.isfinite(v)
      if not np.all(finite_mask):
        logging.warning("Item %s contains non-finite elements.", k)
        v = np.where(finite_mask, v, np.zeros_like(v))
      if v is None:
        logging.warning("Invalid value for %s", k)
      elif np.ndim(v) == 0:
        scalars[s_key] = v
      elif self._is_image(v):
        images[i_key] = self._upscale_image(v)
      else:
        histograms[s_key] = self._remove_outliers(v)

    # Handle text data.
    for (k, v) in self.text_dict.items():
      s_key = s_prefix + k
      text_dict[s_key] = v

    # Write metrics.
    if scalars:
      writer.write_scalars(step, scalars)
    if images:
      writer.write_images(step, images)
    if histograms:
      writer.write_histograms(step, histograms)
    if text_dict:
      writer.write_texts(step, text_dict)

    # Clear accumulated summaries.
    self.clear()
