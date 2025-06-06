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

"""ModelBase provides a base class for models.
"""

import abc
import functools
from typing import Any, Callable, Dict, Tuple

import flax
import gin
import jax
from jax import numpy as jnp

import  metrics_summary


Array = jax.Array
Metrics = metrics_summary.MetricDict
PRNGKey = Any

CollectionFilter = flax.core.scope.CollectionFilter
FrozenVariableDict = flax.core.scope.FrozenVariableDict
VariableDict = flax.core.scope.VariableDict


# Must be a function that maps from f(**kwargs) --> flax.linen.Module.
# The pytype Callable does not support keyword arguments.
FlaxModuleFactory = Callable[..., flax.linen.Module]


class ModelBase(abc.ABC):
  """Base class for models.

  ModelBase defines a minimal interface that the meliad training loop expects a
  model to obey.

  * model_init() and model_apply() mirror the interface for flax modules,
    although meliad does not require the underlying model to be implemented
    with flax.

  * get_fake_input() is unique to meliad, and is used in conjuction with
    model_init() to compile and initialize the model.

  For models that do use flax, FlaxModuleModel (defined below) implements
  ModelBase, maps model_init() and model_apply() onto their flax equivalents,
  and forwards get_fake_input() to the underlying flax module.

  A common design pattern used in meliad is for one implementation of ModelBase
  to wrap another ModelBase object, and provide a set of operations that are
  common to many models -- like preprocessing of inputs, and computing various
  metrics.  For example, DecoderOnlyLanguageModel (in transformer/models.py)
  uses this design pattern to define a common interface for comparing Meliad
  transformers against T5X baselines.  See also WrappedModel, below.
  """

  def get_fake_input(self, batch_size: int) -> Any:
    """Return a fake input that is a valid argument to model_apply(x=...)

    The fake input must have the correct shape and dtype, and will be used
    to initialize the model with model_init().

    Args:
      batch_size:

    Returns:
      A pytree of Arrays for the example input.
    """
    # Not abstract, because users may use WrappedModel to intercept this call.
    raise NotImplementedError("get_fake_input() is not implemented.")

  @abc.abstractmethod
  def model_init(self, prng_dict: Dict[str, PRNGKey],
                 dummy_input: Any) -> FrozenVariableDict:
    """Initialize the model, using a similar system to flax.Module.init().

    Args:
      prng_dict: Dictionary of random-number generators for initialization.
      dummy_input: A fake input from get_fake_input.

    Returns:
      A dictionary of initialized flax parameters and variable collections.
    """
    pass

  @abc.abstractmethod
  def model_apply(
      self,
      params_state: VariableDict,
      inputs: Any,
      prng_dict: Dict[str, PRNGKey],
      mutable_keys: CollectionFilter,
      *,
      global_info: Dict[str, Any],
  ) -> Tuple[Any, FrozenVariableDict]:
    """Call the model function, using a similar system to flax.Module.apply().

    For a top-level model (i.e. not a nested one), the output_value should
    be a tuple of (loss: Array, metrics: Metrics).

    Args:
      params_state: A dictionary of initialized flax parameters and variables.
      inputs:       A batch of inputs to the model for a given training state.
      prng_dict:    A dictionary of random-number generators.
      mutable_keys: The names of variable collections that are mutable; these
        will be returned in the result.
      global_info:  A dictionary of global information (e.g., current training
        step as "step").

    Returns:
      A nested tuple of (output_value, modified_mutable_variables)
    """
    pass


# Must be a function that maps from f(**kwargs) --> ModelBase.
# The pytype Callable does not support keyword arguments.
WrappedModelFactory = Callable[..., ModelBase]


@gin.configurable
class FlaxModuleModel(ModelBase):
  """A lightweight wrapper that turns a flax Module into a ModelBase object.

  This class provides an implementation of ModelBase that functions as an
  simple adaptor.  It takes any flax module, and turns it into a ModelBase
  object that can be used with the flax training loop.

  The underlying flax module must implement get_fake_input().
  """

  def __init__(self, flax_module_factory: FlaxModuleFactory = gin.REQUIRED,
               **kwargs):
    self.flax_module = flax_module_factory(**kwargs)

  def _accepts_global_info(self):
    return (
        hasattr(self.flax_module, "accepts_global_info")
        and self.flax_module.accepts_global_info()
    )

  def get_fake_input(self, batch_size: int) -> Any:
    """See base class.

    Either the underlying flax module should implement get_fake_input, or this
    class should be wrapped in another model class (like DecoderOnlyModel) that
    implements get_fake_input.

    Args:
      batch_size:

    Returns:
      A pytree of Arrays for the example input.
    """
    # Derived classes may override this method to intercept this call.
    if not hasattr(self.flax_module, "get_fake_input"):
      raise NotImplementedError(
          f"The underlying flax module, of type {type(self.flax_module)}, does"
          " not implement get_fake_input."
      )
    return self.flax_module.get_fake_input(batch_size)

  def model_init(self,
                 prng_dict: Dict[str, PRNGKey],
                 dummy_input: Any) -> FrozenVariableDict:
    init_fn = self.flax_module.init
    if self._accepts_global_info():
      # TODO(cstaats): Don't hardcode the keys of the global_info dict.
      global_info = {"step": jnp.int64(1)}
      init_fn = functools.partial(init_fn, global_info=global_info)
    return init_fn(prng_dict, dummy_input)

  def model_apply(
      self,
      params_state: VariableDict,
      inputs: Any,
      prng_dict: Dict[str, PRNGKey],
      mutable_keys: CollectionFilter,
      *,
      global_info: Dict[str, Any],
  ) -> Tuple[Any, FrozenVariableDict]:
    apply_fn = self.flax_module.apply
    if self._accepts_global_info():
      apply_fn = functools.partial(apply_fn, global_info=global_info)
    return apply_fn(
        params_state,
        inputs,
        rngs=prng_dict,
        mutable=mutable_keys,
    )


@gin.configurable
class WrappedModel(ModelBase):
  """Base class for models that are a wrapper around another ModelBase.

  This class can be used to implement common operations and/or do input
  preprocessing before handing a call off to another model.
  """

  def __init__(self, model_factory: WrappedModelFactory = gin.REQUIRED,
               **kwargs):
    self.wrapped_model = model_factory(**kwargs)

  def get_fake_input(self, batch_size: int) -> Any:
    # Derived classes may override this method to intercept this call.
    return self.wrapped_model.get_fake_input(batch_size)

  def model_init(self,
                 prng_dict: Dict[str, PRNGKey],
                 dummy_input: Any) -> FrozenVariableDict:
    return self.wrapped_model.model_init(prng_dict, dummy_input)

  def model_apply(
      self,
      params_state: VariableDict,
      inputs: Any,
      prng_dict: Dict[str, PRNGKey],
      mutable_keys: CollectionFilter,
      *,
      global_info: Dict[str, Any],
  ) -> Tuple[Any, FrozenVariableDict]:
    return self.wrapped_model.model_apply(params_state, inputs, prng_dict,
                                          mutable_keys, global_info=global_info)
