# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

# pylint: disable=unused-import,g-bad-import-order
"""Contains the base Layer class, from which all layers inherit.

This is a private class and its internal implementation is subject to changes
in the future.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import inspect
import re
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
import six

from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import dtypes


class _Layer(object):
  """Base layer class.

  WARNING: Do not subclass this layer unless you know what you are doing:
  the API is subject to future changes.

  This is the class from which all layers inherit, implementing common
  infrastructure functionality.

  A layer is a class implementing common neural networks operations, such
  as convolution, batch norm, etc. These operations require managing weights,
  losses, and updates, as well as applying TensorFlow ops to input tensors.

  Properties:
    trainable: Whether the layer should be trained (boolean).
    name: The name of the layer (string).
    trainable_weights: List of trainable weights.
    non_trainable_weights: List of non-trainable weights.
    weights: List of all weights of this layer, trainable and non-trainable.
    updates: List of update ops of this layer.
    losses: List of losses added by this layer.
  """

  def __init__(self, trainable=True, name=None,
               dtype=dtypes.float32, **kwargs):
    # We use a kwargs dict here because these kwargs only exist
    # for compatibility reasons.
    # The list of kwargs is subject to changes in the future.
    # We do not want to commit to it or to expose the list to users at all.
    # Note this is exactly as safe as defining kwargs in the function signature,
    # the only difference being that the list of valid kwargs is defined
    # below rather rather in the signature, and default values are defined
    # in calls to kwargs.get().
    allowed_kwargs = {
        '_scope',
        '_reuse_weights',
    }
    for kwarg in kwargs:
      if kwarg not in allowed_kwargs:
        raise TypeError('Keyword argument not understood:', kwarg)

    self._trainable = trainable
    self._built = False
    self._trainable_weights = []
    self._non_trainable_weights = []
    self._updates = []
    self._losses = []
    self._reuse_weights = kwargs.get('_reuse_weights')
    self._dtype = dtype

    # Determine base name (non-unique).
    base_name = name
    if not name:
      base_name = _to_snake_case(self.__class__.__name__)

    # Determine variable scope.
    scope = kwargs.get('_scope')
    if scope:
      self._scope = next(vs.variable_scope(scope).gen)
    else:
      self._scope = next(vs.variable_scope(None, default_name=base_name).gen)

    # Unique name is borrowed from scope to match variable names.
    self._name = self._scope.name

  def __setattr__(self, name, value):
    if hasattr(self, name):
      # Only allow self to update its own attributes
      stack_0_locals = inspect.stack()[1][0].f_locals
      called_from_layer = stack_0_locals.get('self', None) is self
      if not called_from_layer:
        raise AttributeError('Read-only property cannot be set: %s' % name)
    super(_Layer, self).__setattr__(name, value)

  @property
  def name(self):
    return self._name

  @property
  def trainable_weights(self):
    return self._trainable_weights if self.trainable else []

  @property
  def non_trainable_weights(self):
    return self._non_trainable_weights if self.trainable else self.weights

  @property
  def updates(self):
    return self._updates

  @property
  def losses(self):
    return self._losses

  @property
  def built(self):
    return self._built

  @property
  def trainable(self):
    return self._trainable

  @property
  def weights(self):
    """Returns the list of all layer weights, trainable and non-trainable.

    Returns:
      A list of variables.
    """
    return self._trainable_weights + self._non_trainable_weights

  def build(self, _):
    """Creates the weights of the layer.
    """
    self._built = True

  def call(self, inputs, **kwargs):
    """The logic of the layer lives here.

    Arguments:
      inputs: input tensor(s).
     **kwargs: additional keyword arguments.

    Returns:
      Output tensor(s).
    """
    raise NotImplementedError

  def _add_weight(self, name, shape, dtype=None,
                  initializer=None, regularizer=None, trainable=True,
                  variable_getter=vs.get_variable):
    """Adds a new weight variable to the layer.

    Arguments:
      name: weight name.
      shape: weight shape.
      dtype: The type of the weight variable. Defaults to `self._dtype`.
      initializer: initializer instance (callable).
      regularizer: regularizer instance (callable).
      trainable: whether the weight should be part of the layer's
        "trainable_weights" (e.g. weights, biases)
        or "non_trainable_weights" (e.g. BatchNorm mean, stddev).
      variable_getter: The getter to use for TensorFlow variables.

    Returns:
      The created variable.
    """
    if dtype is None:
      dtype = self._dtype
    variable = variable_getter(name,
                               shape=shape,
                               initializer=initializer,
                               dtype=dtype,
                               trainable=trainable and self.trainable)
    if trainable:
      self._trainable_weights.append(variable)
    else:
      self._non_trainable_weights.append(variable)
    if regularizer and not self._reuse_weights:
      with ops.colocate_with(variable.op):
        with ops.name_scope(name + '/Regularizer'):
          regularization = regularizer(variable)
      if regularization is not None:
        self._losses.append(regularization)
        _add_elements_to_collection(
            regularization, ops.GraphKeys.REGULARIZATION_LOSSES)
    return variable

  def __call__(self, inputs, **kwargs):
    """Wraps `call`, applying pre- and post-processing steps.

    Arguments:
      inputs: input tensor(s).
      **kwargs: additional keyword arguments to be passed to `self.call`.

    Returns:
      Output tensor(s).
    """
    # Define a custom getter to override tf.get_variable when creating layer
    # weights. We respect current custom getter, if one is set.
    current_custom_getter = vs.get_variable_scope().custom_getter
    def variable_getter(getter, name, shape, dtype=None, initializer=None,
                        regularizer=None, trainable=True, **kwargs):
      if current_custom_getter is not None:
        getter = functools.partial(current_custom_getter, getter)
      return self._add_weight(
          name, shape, initializer=initializer, regularizer=regularizer,
          dtype=dtype, trainable=trainable,
          variable_getter=functools.partial(getter, **kwargs))

    # Build (if necessary) and call the layer, inside a variable scope.
    with vs.variable_scope(self._scope, reuse=self._reuse_weights,
                           custom_getter=variable_getter) as scope:
      with ops.name_scope(scope.original_name_scope):
        if not self.built:
          input_list = _to_list(inputs)
          input_shapes = [x.get_shape() for x in input_list]
          if len(input_shapes) == 1:
            self.build(input_shapes[0])
          else:
            self.build(input_shapes)
          self._built = True
        outputs = self.call(inputs, **kwargs)

        # Apply activity regularization.
        # Note that it should be applied every time the layer creates a new
        # output, since it is output-specific.
        if hasattr(self, 'activity_regularizer') and self.activity_regularizer:
          output_list = _to_list(outputs)
          for output in output_list:
            with ops.name_scope('ActivityRegularizer'):
              activity_regularization = self.activity_regularizer(output)
            self._losses.append(activity_regularization)
            _add_elements_to_collection(
                activity_regularization, ops.GraphKeys.REGULARIZATION_LOSSES)

    # Update global default collections.
    _add_elements_to_collection(self.updates, ops.GraphKeys.UPDATE_OPS)
    return outputs

  def apply(self, inputs, **kwargs):
    """Apply the layer on a input.

    This simply wraps `self.__call__`.

    Arguments:
      inputs: Input tensor(s).
      **kwargs: additional keyword arguments to be passed to `self.call`.

    Returns:
      Output tensor(s).
    """
    return self.__call__(inputs, **kwargs)


def _to_snake_case(name):
  intermediate = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
  insecure = re.sub('([a-z0-9])([A-Z])', r'\1_\2', intermediate).lower()
  # If the class is private the name starts with "_" which is not secure
  # for creating scopes. We prefix the name with "private" in this case.
  if insecure[0] != '_':
    return insecure
  return 'private' + insecure


def _to_list(x):
  """This normalizes a list/tuple or single element into a list.

  If a single element is passed, we return
  a list of size 1 containing the element.

  Arguments:
    x: list or tuple or single element.

  Returns:
    A list.
  """
  if isinstance(x, (list, tuple)):
    return list(x)
  return [x]


def _add_elements_to_collection(elements, collections):
  elements = _to_list(elements)
  collections = _to_list(collections)
  for name in collections:
    collection = ops.get_collection_ref(name)
    collection_set = set(collection)
    for element in elements:
      if element not in collection_set:
        collection.append(element)
