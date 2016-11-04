# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Subscribe function."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops


def _recursive_apply(tensors, apply_fn):
  """Helper method to recursively apply a function to structure of tensors.

  The structure of the tensors should take the form similar to fetches in
  `tf.Session` and includes single `Tensor`, `list`, nested `list`, `tuple`,
  `namedtuple`, or `dict`.

  Args:
    tensors: Single `Tensor`, `list`, nested `list, `tuple`,
      `namedtuple`, or `dict`.
    apply_fn: Function to apply to each `Tensor` and should return a `Tensor`.
  Returns:
    Returns the modified tensors with the same structure.
  Raises:
    `TypeError` if undefined type in the tensors structure.
  """
  tensors_type = type(tensors)
  if tensors_type is ops.Tensor:
    return apply_fn(tensors)
  elif isinstance(tensors, (list, tuple)):
    tensors = [_recursive_apply(t, apply_fn) for t in tensors]
    if tensors_type is list:
      return list(tensors)
    elif tensors_type is tuple:
      return tuple(tensors)
    return tensors_type(*tensors)  # collections.namedtuple
  elif tensors_type is dict:
    return dict([(k, _recursive_apply(v, apply_fn))
                 for k, v in tensors.iteritems()])
  else:
    raise TypeError('_recursive_apply argument %r has invalid type %r' %
                    (tensors, tensors_type))


def _control_outputs(op):
  """Returns the control_input consumers for the supplied `Operation`.

  Args:
    op: The `Operation` to find consumers of.
  Yields:
    A list of ops that have op as a control dependency.
  """
  for o in op.graph.get_operations():
    if op in o.control_inputs:
      yield o


def _subscribe(tensor, side_effects):
  """Helper method that subscribes a single tensor to a list of side_effects.

  Args:
    tensor: `tf.Tensor`
    side_effects: List of side_effect functions see subscribe for details.
  Returns:
    The modified replacement to the passed in tensor which triggers the side
    effects.
  """
  update_input = []
  for consumer_op in list(tensor.consumers()):  # explicit copy
    update_input.append((consumer_op, list(consumer_op.inputs).index(tensor)))

  update_control_input = list(_control_outputs(tensor.op))

  # Trailing slash on name scope to replace the scope.
  name_scope = tensor.op.name + '/subscription/'
  with ops.name_scope(name_scope):
    outs = []
    for s in side_effects:
      outs += s(tensor)

    with ops.control_dependencies(outs):
      out = array_ops.identity(tensor)

  for consumer_op, index in update_input:
    consumer_op._update_input(index, out)  # pylint: disable=protected-access

  for consumer_op in update_control_input:
    consumer_op._control_inputs.remove(tensor.op)  # pylint: disable=protected-access
    consumer_op._control_inputs.append(out.op)  # pylint: disable=protected-access
    consumer_op._recompute_node_def()  # pylint: disable=protected-access

  return out


def subscribe(tensors, side_effects):
  """Subscribe to a tensor.

  This method will attach side effect graphs to a given set
  of tensors. Set of tensors follows from session.run and supports
  single `Tensor`, `list`, nested `list`, `tuple`, `namedtuple`, or `dict`. It
  returns the tensors in the same passed in structure, but as clones with
  side effects applied. The supplied side effect graphs are specified
  as a constructor function which takes the target tensor and
  constructs a side effect graph and returns a list of ops that should
  be control dependencies on fetching the tensor. It will append
  'subscription' to the name scope of the tensor for every node in
  the side effect graph. These control dependencies are what trigger
  the side effects. Subscribe will construct the additions to your
  graph and return the created identity tensor downstream of the control
  dependencies. Use these tensors as you would normally in the rest of
  your tensorflow code.

  Args:
    tensors: `Tensor` or set of tensors to subscribe to. Set of tensors format
      follows from `Session.run` and supports single `Tensor`, `list`, nested
      `list`, `tuple`, `namedtuple`, or `dict`.
    side_effects: Function(s) that takes a `Tensor`, construct a subgraph, and
      return a nonempty list of control dependencies. This can be a single
      function or list of functions.
  Returns:
    Subscribed tensors, which are identity copies of the passed in tensors
      in the same passed in structure, but the graph has been modified
      such that these are downstream of the control dependencies for
      the side effect graphs. Use these functionally equivelant tensors
      instead of the passed in tensors for further construction or running.
  """
  if not hasattr(side_effects, '__iter__'):
    side_effects = [side_effects]
  return _recursive_apply(tensors, lambda t: _subscribe(t, side_effects))
