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

"""TrainingTask encapsulates the state associated with model step."""

import time
from typing import (Any, Callable, Dict, Iterator, Mapping, Optional, Tuple)

from absl import logging
from clu import metric_writers
from flax import optim
from flax import struct
import jax
import  metrics_summary
import numpy as np


@struct.dataclass
class TrainState:
  optimizer: optim.Optimizer   # Trainable parameters.
  state: Any                   # Other state, e.g. XL cache or memory.


PRNGKeys = Any
Metrics = Dict[str, Any]
MetricsSummary = metrics_summary.MetricsSummary

Dataset = Callable[[], Iterator[Any]]
StepFunction = Callable[[TrainState, Any, Any], Tuple[TrainState, Metrics]]
PrettyPrintInputFunction = Optional[Callable[[Any], str]]
ProcessSummariesFunction = Optional[Callable[[Any, str], Any]]
ExtraSummariesFunction = Optional[Callable[[str, int], Mapping[str, Any]]]


def should_run(step: int, every_steps: int) -> bool:
  """Returns true if a periodic action should be run."""
  return (step > 0) and (every_steps > 0) and (step % every_steps == 0)


class TrainingTask:
  """A TrainingTask encapsulates the state associated with a training task.

  Examples of tasks include training steps, test or validation runs,
  or inference (generation).  State includes the input pipeline, and
  summary information that is averaged over multiple steps.
  """

  def __init__(
      self,
      *,  # Pass arguments by keyword only.
      mode: str,
      dataset: Dataset,
      step_function: StepFunction,
      prng_keys: PRNGKeys,
      summary: MetricsSummary,
      extra_summary: MetricsSummary,
      summary_writer: metric_writers.MetricWriter,
      summary_prefix: str = "",
      # --- Options from TrainingLoop ---
      replicate_mode: bool = True,
      print_input_every_steps: int = 0,
      pretty_print_input_function: PrettyPrintInputFunction = None,
      process_summaries_function: ProcessSummariesFunction = None,
      extra_summaries_function: Optional[ExtraSummariesFunction] = None):
    # Local state.
    self.mode = mode
    self.dataset = dataset
    self.step_function = step_function
    self.prng_keys = prng_keys
    self.summary = summary
    self.extra_summary = extra_summary
    self.summary_writer = summary_writer
    self.summary_prefix = summary_prefix

    # Options carried over from TrainingLoop.
    self.replicate_mode = replicate_mode
    self.print_input_every_steps = print_input_every_steps
    self.pretty_print_input_fn = pretty_print_input_function
    self.process_summaries_fn = process_summaries_function
    self.extra_summaries_fn = extra_summaries_function

    # Local state.
    if self.dataset is not None:
      self.ds_iterator = self.dataset()
    self.epoch = 0

  def _get_metrics(self, device_metrics: Metrics) -> Metrics:
    """Read a dictionary of metrics from device."""
    if self.replicate_mode:
      # x[0] gets the metric from device 0 -- the first replica.
      # We assume that merge_replicated_metrics has already combined the
      # metrics from multiple devices.
      device_metrics = jax.tree_map(lambda x: x[0], device_metrics)
    metrics_np = jax.device_get(device_metrics)  # Get numpy arrays.
    return metrics_np

  def get_next_input(self) -> Any:
    """Grab the next input from the data pipeline."""
    if self.dataset is None:
      logging.warning("No dataset for mode %s", self.mode)
      return None

    try:
      x = next(self.ds_iterator)
    except StopIteration:
      logging.info("End of epoch %d for mode %s.", self.epoch, self.mode)
      self.ds_iterator = self.dataset()
      x = next(self.ds_iterator)
      self.epoch += 1
    return x

  def run_step(self, tstate: TrainState, x: Any,
               step: int, sub_step: int = 0) -> Tuple[TrainState, Metrics]:
    """Run the model for a single step.

    Args:
      tstate: The current model state.
      x: The input for the model -- from get_next_input.
      step: The training step number.
      sub_step: For tasks that run multiple iterations within a step.
        E.g. A test cycle will call run_step multiple times to cover the test
        set.  The step counter will not increment, but sub_step will.

    Returns:
      An updated model state.
    """

    start_time = time.perf_counter()

    # Split a batch of inputs among local replicas.
    if self.replicate_mode:
      x = split_batch_dimension(x, jax.local_device_count())

    # Pretty-print the input to the summary and log file every so often.
    if (sub_step == 0 and self.pretty_print_input_fn is not None and
        should_run(step, self.print_input_every_steps)):
      x_first = jax.tree_map(lambda x: x[0], x) if self.replicate_mode else x
      x_strs = self.pretty_print_input_fn(x_first)
      logging.info("[%d] Input (%s) = %s", step, self.mode, x_strs)
      self.summary.add_text({"input": x_strs})

    # Run the step function on the input.
    with jax.profiler.StepTraceAnnotation(self.mode, step_num=step):
      (tstate, metrics) = self.step_function(tstate, x, self.prng_keys)

    # Read metrics from device.
    metrics_np = self._get_metrics(metrics)
    end_time = time.perf_counter()
    metrics_np["step_time"] = end_time - start_time
    if "epoch" not in metrics_np.keys():
      metrics_np["epoch"] = self.epoch

    # Add metrics to the current summary.
    self.summary.add(metrics_np)
    return (tstate, metrics_np)

  def flush(self, step: int):
    """Flush accumulated metric summaries to disk."""

    if self.summary_writer is None:
      self.summary.clear()  # Clear summary if we can't write it.
      return

    if self.summary.empty():
      return

    # Do post-processing of the summaries.
    if self.process_summaries_fn is not None:
      self.summary = self.process_summaries_fn(self.summary, self.mode)  # pylint: disable=not-callable

    # Write and clear summary data.
    logging.info("Writing summaries for mode %s.", self.mode)
    self.summary.write(self.summary_writer, step, prefix=self.summary_prefix)

    # Add extra summaries that are not computed by the step function.
    if self.extra_summaries_fn is not None:
      self.extra_summary.add(self.extra_summaries_fn(self.mode, step))
      self.extra_summary.write(self.summary_writer, step, prefix="")


def split_batch_dimension(inputs: Any, num_replicas: int) -> Any:
  """Splits the leading batch dimension.

  Given inputs of shape [num_replicas * batch_size, ...], it will reshape
  them to [num_replicas, batch_size, ...].  This operation is intended to be
  used right before calling pmap, which will eliminate the num_replicas
  dimension.

  Args:
    inputs: Tuple of inputs to split.
    num_replicas: Number of replicas.

  Returns:
    inputs with extra batch dimension.
  """

  def split_batch_dim(x):
    assert x.ndim > 0
    if (x.shape[0] % num_replicas) != 0:
      raise ValueError(f"Can't split {x.shape} into {num_replicas} replicas.")
    batch_size = x.shape[0] // num_replicas
    split_shape = [num_replicas, batch_size] + list(x.shape[1:])
    return np.reshape(x, split_shape)

  return jax.tree_map(split_batch_dim, inputs)
