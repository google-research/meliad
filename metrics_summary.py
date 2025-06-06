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

"""Class to handle summarizing of metrics over multiple training steps."""
from __future__ import annotations

from typing import Any, Dict, Optional, TypeVar, Union
from absl import logging
from clu import metric_writers
import clu.values
import flax
import gin
import jax
from jax import numpy as jnp
import numpy as np


Array = Union[jnp.ndarray, np.ndarray]
MetricValue = clu.values.Value


M = TypeVar("M", bound="Metric")


class Metric:
  """A metric (e.g. an output) returned from a model.

  This implementation is similar to of clu.Metric, but has a simplified
  interface, and is a bit stricter about how it's supposed to be used.
  ModelInfo expects the model to return a dictionary of Metrics.

  In general, each metric is created in the model using a jax Array, has a
  batch dimension, and may be distributed across devices.  On each training
  step, the reduce() operation combines metrics over the batch dimension from
  multiple devices, and yields a (usually scalar) metric. Thus, implementations
  of reduce() are allowed to use jax-specific operations.

  After reduce() is called, the /reduced metric/ is transferred from
  accelerator to host, and the jax Arrays become numpy arrays.  After each
  training step, the merge() operation is called to combine metrics across
  multiple steps.  The implementation of merge() should thus use polymorphic
  or numpy operations.

  When it is time to log a metric to disk, compute() is used to convert the
  metric to a final (usually scalar) value that can be graphed or otherwise
  displayed within Tensorboard.
  """

  def reduce(self: M, replicate_mode: str) -> M:
    """Combine values from a model along the batch dimension and any replicas.

    This operation is often the same as merge() but not always.  E.g. one may
    wish to sum across the batch dimension and replicas, and still average
    across steps.  The implementaion of reduce() can assume that arrays are
    jax arrays, and use jax operations.

    Args:
      replicate_mode: one of "pmap", "pjit", or "none".
          This argument is supplied by ModelInfo.
          If replicate_mode == "pmap", pmap-appropriate operations (e.g. psum)
          must be used to combine metrics across replicas.

    Returns:
      A new /reduced metric/ which has been reduced along the batch dimension,
      and/or over replicas.
    """
    raise NotImplementedError("Must override reduce()")

  def merge(self: M, other: M) -> M:
    """Combine a new value from the current step into the existing one.

    This operation summarizes metrics over multiple training steps.  It is
    computed on the host by the training loop, and thus should assume that
    any arrays are numpy arrays, and use numpy operations.

    Args:
      other: metrics from the current time step.

    Returns:
      A new metric that combines the current summary with other.
    """
    raise NotImplementedError("Must override merge()")

  def compute(self) -> MetricValue:
    """Compute a clu Value for the metric that can be written to disk."""
    raise NotImplementedError("Must override compute()")

  def info_string(self) -> str:
    """Return information about the given metric, for debugging purposes."""
    raise NotImplementedError("Must override info_string()")


def average_metric(value: Array, weight: Optional[Array] = None) -> "Average":
  """Return a Metric that does a weighted average.

  This metric will compute a weighted average of its input. It will average over
  the batch dimension and replicas, and will maintain a cumulative average over
  steps.

  Args:
    value: A jax array containing the values to be averaged.
    weight: A jax array of weights, of the same shape as value.
        If not specified, defaults to 1.0.

  Returns:
    A Metric.
  """

  # Upcast to float32 to maintain precision.
  value = jnp.array(value, dtype=jnp.float32)

  if weight is None:
    # Use a default weight of 1.0
    weight = jnp.full(shape=value.shape, fill_value=1.0, dtype=jnp.float32)
  else:
    # A weighted average is (a1*w1 ... + ... an*wn) / (w1 ... + ... wn)
    # We premultiply by weight, sum both value and weight, and then divide
    # by the sum of weights in compute().
    weight = jnp.array(weight, dtype=jnp.float32)
    if value.shape != weight.shape:
      raise ValueError(f"The value {vshape(value)} and weight {vshape(weight)} "
                       f"must have the same shape.")
    value = value * weight

  value = jnp.sum(value)
  weight = jnp.sum(weight)
  return Average(value, weight)


def average_metric_np(value: Any, weight: Any = None) -> "Average":
  """Return a reduced (host) metric that does a weighted cumulative average.

  This function returns an already-reduced metric; i.e. it must be a scalar,
  and it uses numpy arrays (on the host) rather than Jax arrays (distributed
  over accelerators).  It is used to log metrics from within code running on
  the host, and thus should not be returned directly from the model.

  Args:
    value: A scalar value that can be converted to a numpy array.
    weight: A scalar weight for a weighted average over time.
        If unspecified, defaults to 1.0.

  Returns:
    An already-reduced Metric.
  """

  value = np.array(value, dtype=np.float32)
  if value.ndim != 0:
    raise ValueError(f"The value {vshape(value)} must be a scalar.")

  if weight is None:
    weight = np.array(1.0, dtype=np.float32)
  else:
    # A weighted average is (a1*w1 ... + ... an*wn) / (w1 ... + ... wn)
    # We premultiply by weight, sum both value and weight, and then divide
    # by the sum of weights in compute().
    weight = np.array(weight, dtype=np.float32)
    if weight.ndim != 0:
      raise ValueError(f"The weight {vshape(weight)} must be a scalar.")
    value = value * weight

  return Average(value, weight)


def average_of_sum_metric(value: Array) -> "AverageOfSum":
  """Return a metric that sums within a step, but averages over steps.

  Similar to average_metric(), but sums over all of the elements of 'value',
  including the batch dimension and any replicas. Like average, it then
  maintains a cumulative average (of the sum) over multiple steps. This metric
  can be used to count things like num_tokens_per_batch.

  There is no weight, since there is no weighted average over 'value'.

  Args:
    value: A jax array of values to sum.

  Returns:
    A Metric.
  """

  # Upcast to float32 if necessary to maintain precision.
  value = jnp.array(value, dtype=jnp.float32)
  # Sum over all elements, including the batch dimension.
  value = jnp.sum(value)
  weight = jnp.array(1.0, dtype=jnp.float32)
  return AverageOfSum(total=value, weight=weight)


def scalar_metric(val: Array) -> "LastScalar":
  """Returns an unreplicated scalar metric with no batch dimension or averaging.

  A scalar metric has no batch dimension, and thus is not reduced or averaged
  over the batch dimension.  When used with pmap, it will return values only
  from the first device on the master host.  It is also not averaged over time;
  it simply reports the value of the most recent step.

  Args:
    val: A scalar value, which can be a jax or numpy array.

  Returns:
    A scalar Metric.
  """

  if isinstance(val, jax.Array):
    if (val.ndim != 0):
      raise ValueError("Argument to scalar_metric must be a scalar.")
  elif isinstance(val, np.ndarray):
    val = np.array(val)   # Fail if can't convert to an np array.
    if (val.ndim != 0):
      raise ValueError("Argument to scalar_metric must be a scalar.")
  return LastScalar(val)


def text_metric(s: str) -> "LastText":
  """Returns a text metric, viewable in Tensorboard's text tab.

  Text metrics obey the semantics of LastValue.  The reduce() operation is a
  no-op, and merge() simply returns the last value.

  Args:
    s: str

  Returns:
    A text Metric.
  """
  return LastText(s)


def output_value_metric(val: Any) -> "OutputValue":
  """Encodes a model output as a metric for convenience.

  This function can be use to return values of arbitrary type (typically
  arrays), as part of a model's MetricDict.  These values will not be logged
  to disk.  It is intended mainly for use with colabs.

  Args:
    val: An object of any jax-compatible type.

  Returns:
    A dummy Metric with the given value.
  """
  return OutputValue(val)


def loss_value(val: Array, loss_type: str = "default") -> "LossValue":
  """Encodes a loss as a metric.

  Like output_value_metric, this is a dummy metric.  Different parts of the
  model can write losses to the metric dictionary.  The model can then collect
  all LossValues into a differentiable loss term.  Loss metrics should be
  removed from the dictionary after they are collected.

  Args:
    val: An object of any jax-compatible type.
    loss_type: A model-specific string describing the type of loss.

  Returns:
    A LossMetric with the given value.
  """
  return LossValue(val, loss_type)


# A model should return a MetricDict as one of its outputs.
MetricDict = Dict[str, Metric]


def add_loss_to_metric_dict(mdict: MetricDict,
                            loss: Array,
                            loss_name: str,
                            loss_type: str = "default"):
  """Add a loss to the metric dict.

  All losses should eventually be collected by the top-level model and summed.
  Failure to call collect_losses_from_metric_dict is an error.

  Args:
    mdict: Dictionary of metrics.
    loss: An array of losses.
    loss_name: The name of the loss.
    loss_type: Optional string to specify a model-specific loss type.
  """

  if loss_name in mdict:
    raise ValueError(f"Loss {loss_name} has already been defined.")
  mdict[loss_name] = loss_value(loss, loss_type)


def collect_losses_from_metric_dict(mdict: MetricDict) -> Optional[Array]:
  """Collect all LossValues from mdict, and return a scalar loss."""

  # Collect and sum all LossValues in the dictionary.
  loss = None
  lkeys = []
  for (k, lv) in mdict.items():
    if isinstance(lv, LossValue):
      if lv.value.ndim != 0:
        raise ValueError(f"Loss {k} is not a scalar loss.")
      else:
        logging.info("Found loss %s of type %s", k, lv.loss_type)
        if loss is None:
          loss = lv.value
        else:
          loss += lv.value
      lkeys.append(k)

  # Remove LossValues from the dictionary.
  for k in lkeys:
    del mdict[k]

  return loss


def reduce_metrics(mdict: MetricDict, replicate_mode: str) -> MetricDict:
  """Reduce all metrics in the given MetricDict.

  Args:
    mdict: A dictionary of metrics.
    replicate_mode: one of "pmap", "pjit", or "none".
        This argument is supplied by ModelInfo.

  Returns:
    A new dectionary of metrics, in which each Metric has been reduced along
    the batch dimension.
  """

  rmdict: MetricDict = {}
  for (k, m) in mdict.items():
    rmdict[k] = m.reduce(replicate_mode)
  return rmdict


def log_metric_info(mdict: MetricDict):
  """Log info on all metrics returned by the model."""

  kmlist = list(mdict.items())
  kmlist.sort()

  logging.info("--- Metrics ---")
  for (k, m) in kmlist:
    if not isinstance(m, Metric):
      logging.info("Invalid metric %s", k)
    else:
      logging.info("Found metric %s = %s", k, m.info_string())


def log_metric_values(mdict: MetricDict):
  """Print metrics returned by the model for debugging purposes."""

  kmlist = list(mdict.items())
  kmlist.sort()

  logging.info("--- Metric Values ---")
  for (k, m) in kmlist:
    if not isinstance(m, Metric):
      logging.info("Invalid metric %s", k)
    elif isinstance(m, OutputValue):
      logging.info("metric %s = %s", k, m.info_string())
    else:
      logging.info("metric %s = %s; compute = %s",
                   k, m.info_string(), m.compute())


@gin.configurable
class MetricsSummary:
  """Summarizes a set of a metrics over multiple training steps.

  MetricsSummary is a dictionary of metrics which persists over multiple
  training steps.  It stores reduced Metrics on the host, and is managed by the
  training loop.
  """

  def __init__(self):
    self.metric_dict: MetricDict = {}

  def current_metric_dict(self) -> MetricDict:
    """Return the dictionary of accumulated metrics for the current step."""
    return self.metric_dict

  def empty(self) -> bool:
    """Return true if there are no summaries to write."""
    return not self.metric_dict

  def clear(self):
    """Clear acculumated summaries."""
    self.metric_dict = {}

  def merge(self, mdict: MetricDict):
    """Merge a dict returned from the model into the current set of metrics."""

    for (k, m) in mdict.items():
      if not isinstance(m, Metric):
        raise ValueError(f"item {k}:{type(m)} in Metric dict is not a Metric.")

      if k in self.metric_dict:
        # Merge with existing metric.
        self.metric_dict[k] = self.metric_dict[k].merge(m)
      else:
        # Add new metric to dict.  (Usually only happens on the first step.)
        self.metric_dict[k] = m

  def write(self, writer: metric_writers.MetricWriter, step: int, prefix: str):
    """Write metrics using summary_writer, and clear all summaries."""
    if self.empty():
      return

    # Special logic for organizing metrics under tensorboard.
    # Tensorboard has top-level groups, but doesn't have subgroups.
    # Scalars are put into separate top-level groups for easier viewing.
    # e.g. all scalars in "train", "test", etc.
    if prefix:
      s_prefix = prefix + "/"
    else:
      s_prefix = ""

    metric_values = {}
    for (k, m) in self.metric_dict.items():
      if not isinstance(m, Metric):
        raise ValueError(f"{k} in metric dictionary is not a metric.")
      if isinstance(m, OutputValue):
        # OutputValues are not actual metrics; ignore them.
        # They populate the metric dictionary for use in colabs.
        continue
      s_key = s_prefix + k
      metric_values[s_key] = m.compute()

    clu.metric_writers.utils.write_values(writer, step, metric_values)
    self.clear()


@flax.struct.dataclass
class Average(Metric):
  """Keeps a cumulative average of a scalar.

  Average will perform a weighted average both over the batch dimension and
  devices, and over time steps.  It expects initial values of shape
  (batch_size,).

  Use average_metric(), above, to construct instances.
  """

  total: Array
  weight: Array

  def reduce(self: M, replicate_mode: str) -> M:
    # Reduce to a scalar by summing across the batch dimension.
    ntotal = jnp.sum(self.total)
    nweight = jnp.sum(self.weight)
    if replicate_mode == "pmap":
      # Further sum across replicas, and broadcast the result to all devices.
      # Metrics are only logged from the master process, and ModelInfo will
      # only read from the first device.
      ntotal = jax.lax.psum(ntotal, axis_name="batch")
      nweight = jax.lax.psum(nweight, axis_name="batch")
    return Average(ntotal, nweight)

  def merge(self: Average, other: Average) -> Average:
    ntotal = self.total + other.total
    nweight = self.weight + other.weight
    # Use type(self) so this method can be inherited by derived classes.
    return type(self)(ntotal, nweight)

  def compute(self) -> MetricValue:
    return clu.values.Scalar(self.total / (self.weight + 1e-6))

  def info_string(self) -> str:
    return f"Average({vshape(self.total)}, {vshape(self.weight)})"


@flax.struct.dataclass
class AverageOfSum(Average):
  """Averages over steps, but sums over the batch dimension.

  Use average_of_sum_metric(), above, to construct instances.
  """

  total: Array
  weight: Array

  def reduce(self, replicate_mode: str) -> AverageOfSum:
    # Reduce to a scalar by summing over the batch dimension.
    ntotal = jnp.sum(self.total)
    if replicate_mode == "pmap":
      # Further sum across replicas, and broadcast the result to all devices.
      # Metrics are only logged from the master process, and ModelInfo will
      # only read from the first device.
      ntotal = jax.lax.psum(ntotal, axis_name="batch")

    # The difference between Average and AverageOfSum is that reduce() only
    # sums the values, not the weights.
    return AverageOfSum(ntotal, jnp.array(1.0))

  def info_string(self) -> str:
    return f"AverageOfSum({vshape(self.total)})"


@flax.struct.dataclass
class LastValue(Metric):
  """Returns the value from the last step.

  LastValue does not do anything to reduce over the batch dimension, or merge
  over steps.  It will return whatever value it had on the the most recent step.
  When used with pmap, it will return a value only from the first device on
  the master host.  I.e. it will return a value from exactly one replica, and
  that value will have whatever shape it usually has within a single replica.

  Jax is incompatible with instances of clu.Value, so there is a subclass
  of LastValue with a different implementation of compute() for every type
  that derives from clu.Value.
  """

  value: Any

  def reduce(self, replicate_mode: str) -> LastValue:
    # Do not reduce across the batch dimension.
    # When using pmap, this will return values only for the master host.
    return self

  def merge(self, other: LastValue) -> LastValue:
    # Returns the most recent value.
    return other


@flax.struct.dataclass
class LastScalar(LastValue):
  """Returns the scalar value from the last step.

  Use scalar_metric(), above, to construct instances.
  """

  def compute(self) -> MetricValue:
    return clu.values.Scalar(self.value)

  def info_string(self) -> str:
    return f"LastScalar({vshape(self.value)})"


@flax.struct.dataclass
class LastText(LastValue):
  """Returns the text from the last step.

  Use text_metric(), above, to construct instances.
  """

  def compute(self) -> MetricValue:
    return clu.values.Text(self.value)

  def info_string(self) -> str:
    return f"LastText({self.value})"


@flax.struct.dataclass
class OutputValue(Metric):
  """A model output that is coded as a Metric, but isn't really.

  This class is useful for returning arbitrary array outputs along with
  the set of metrics.  OutputValues are not actually written to disk with the
  other metrics, because they do not necessarily have a shape that can be
  interpreted by Tensorboard.  Calling compute() is an error.
  """

  value: Any

  def reduce(self, replicate_mode: str) -> OutputValue:
    # Do not reduce across the batch dimension.
    # When using pmap, this will return a value only for the master host.
    return self

  def merge(self, other: OutputValue) -> OutputValue:
    # Returns the most recent value.
    return other

  def info_string(self) -> str:
    if isinstance(self.value, jnp.ndarray):
      s = vshape(self.value)
    else:
      s = str(self.value)
    return f"OutputValue({s})"


@flax.struct.dataclass
class LossValue(Metric):
  """A loss that is coded as a Metric, but isn't really.

  This class is useful for storing various losses within the metric dictionary
  from deeply nested parts of the model.  These losses can then be collected
  and removed from the dictionary by the top-level model code.

  Calling compute(), merge(), or reduce() is an error.
  """

  value: Array
  loss_type: str

  def info_string(self) -> str:
    s = vshape(self.value)
    return f"LossValue({s})"


@gin.configurable
def vshape(x: Any, verbose: bool = False) -> str:
  """Pretty print the shape and/or value of x.

  This function is used throughout meliad to log the shape of arguments when
  the model is being compiled.  If verbose is True (usually set globally True
  via gin), then the value of x will be logged instead.  This can be used to
  show intermediate values for debugging purposes if the model is run in
  immediate mode, rather than being jit-compiled.

  Args:
    x: The value to print.
    verbose: If True, print the value of x, not just the shape.

  Returns:
    A string representation of x.
  """
  if x is None:
    return "None"
  elif verbose:
    return str(x)
  elif hasattr(x, "shape"):
    return "<" + str(x.dtype) + str(x.shape) + ">"
  elif isinstance(x, tuple):
    return "(" + ", ".join([vshape(xi) for xi in x]) + ")"
  else:
    return str(x)
