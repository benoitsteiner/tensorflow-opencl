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
# ==============================================================================

"""Helpers to manipulate a tensor graph in python.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import re

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import tf_logging as logging

_VARIABLE_OPS = {
    "Assign",
    "AssignAdd",
    "AssignSub",
    "Queue",
    "ScatterAdd",
    "ScatterSub",
    "ScatterUpdate",
    "TruncatedNormal",
    "Variable",
}


def _is_variable_op(op):
  """Returns true if 'op' refers to a Variable node."""
  return op in _VARIABLE_OPS


def set_cpu0(device_string):
  """Creates a new device string based on `device_string' but using /CPU:0.

   If the device is already on /CPU:0, this is a no-op.

   Args:
     device_string: A device string.

   Returns:
     A device string.
  """
  parsed_device = pydev.DeviceSpec.from_string(device_string)
  parsed_device.device_type = "CPU"
  parsed_device.device_index = 0
  return parsed_device.to_string()


def must_run_on_cpu(node, pin_variables_on_cpu=False):
  """Returns True if the given node_def must run on CPU, otherwise False.

  Args:
    node: The node to be assigned to a device. Could be either an ops.Operation
      or NodeDef.
    pin_variables_on_cpu: If True, this function will return False if node_def
      represents a variable-related op.

  Returns:
    True if the given node must run on CPU, otherwise False.
  """

  if isinstance(node, ops.Operation):
    node_def = node.node_def
  else:
    assert isinstance(node, node_def_pb2.NodeDef)
    node_def = node

  # If the op is a variable-related op, should we pin it on CPU?
  if pin_variables_on_cpu and _is_variable_op(node_def.op):
    return True

  # Constant operations producing a string or int32 must run on CPU.
  if node_def.op == "Const":
    # Get the value of the 'dtype' attr
    dtype = node_def.attr["dtype"].type
    if dtype == dtypes.string or dtype == dtypes.int32:
      return True

  if node_def.op == "DynamicStitch":
    dtype = node_def.attr["T"].type
    if dtype == dtypes.int32:
      # DynamicStitch on GPU only works for int32 values.
      return True

  if node_def.op in ["Cast"]:
    dtype = node_def.attr["SrcT"].type
    if dtype == dtypes.int32:
      # Cast on GPU does not works for int32 values.
      return True
  return False


################################################################################
#
# device functions for use in with g.device(...)
#
################################################################################


def _node_name(n):
  if n.startswith("^"):
    return n[1:]
  else:
    return n.split(":")[0]


def extract_sub_graph(graph_def, dest_nodes):
  """Extract the subgraph that can reach any of the nodes in 'dest_nodes'.

  Args:
    graph_def: A graph_pb2.GraphDef proto.
    dest_nodes: A list of strings specifying the destination node names.
  Returns:
    The GraphDef of the sub-graph.

  Raises:
    TypeError: If 'graph_def' is not a graph_pb2.GraphDef proto.
  """

  if not isinstance(graph_def, graph_pb2.GraphDef):
    raise TypeError("graph_def must be a graph_pb2.GraphDef proto.")

  edges = {}  # Keyed by the dest node name.
  name_to_node_map = {}  # Keyed by node name.

  # Keeps track of node sequences. It is important to still output the
  # operations in the original order.
  node_seq = {}  # Keyed by node name.
  seq = 0
  for node in graph_def.node:
    n = _node_name(node.name)
    name_to_node_map[n] = node
    edges[n] = [_node_name(x) for x in node.input]
    node_seq[n] = seq
    seq += 1

  for d in dest_nodes:
    assert d in name_to_node_map, "%s is not in graph" % d

  nodes_to_keep = set()
  # Breadth first search to find all the nodes that we should keep.
  next_to_visit = dest_nodes[:]
  while next_to_visit:
    n = next_to_visit[0]
    del next_to_visit[0]
    if n in nodes_to_keep:
      # Already visited this node.
      continue
    nodes_to_keep.add(n)
    next_to_visit += edges[n]

  nodes_to_keep_list = sorted(list(nodes_to_keep), key=lambda n: node_seq[n])
  # Now construct the output GraphDef
  out = graph_pb2.GraphDef()
  for n in nodes_to_keep_list:
    out.node.extend([copy.deepcopy(name_to_node_map[n])])

  return out


def tensor_shape_from_node_def_name(graph, input_name):
  """Convenience function to get a shape from a NodeDef's input string."""
  # To get a tensor, the name must be in the form <input>:<port>, for example
  # 'Mul:0'. The GraphDef input strings don't always have the port specified
  # though, so if there isn't a colon we need to add a default ':0' to the end.
  if ":" not in input_name:
    canonical_name = input_name + ":0"
  else:
    canonical_name = input_name
  tensor = graph.get_tensor_by_name(canonical_name)
  shape = tensor.get_shape()
  return shape


def convert_variables_to_constants(sess, input_graph_def, output_node_names,
                                   variable_names_whitelist=None):
  """Replaces all the variables in a graph with constants of the same values.

  If you have a trained graph containing Variable ops, it can be convenient to
  convert them all to Const ops holding the same values. This makes it possible
  to describe the network fully with a single GraphDef file, and allows the
  removal of a lot of ops related to loading and saving the variables.

  Args:
    sess: Active TensorFlow session containing the variables.
    input_graph_def: GraphDef object holding the network.
    output_node_names: List of name strings for the result nodes of the graph.
    variable_names_whitelist: The set of variable names to convert (by default,
                              all variables are converted).

  Returns:
    GraphDef containing a simplified version of the original.
  """
  found_variables = {}
  variable_names = []
  variable_dict_names = []
  for node in input_graph_def.node:
    if node.op == "Assign":
      variable_name = node.input[0]
      if (variable_names_whitelist is not None and
          variable_name not in variable_names_whitelist):
        continue
      variable_dict_names.append(variable_name)
      variable_names.append(variable_name + ":0")
  if variable_names:
    returned_variables = sess.run(variable_names)
  else:
    returned_variables = []
  found_variables = dict(zip(variable_dict_names, returned_variables))
  logging.info("Froze %d variables." % len(returned_variables))

  # This graph only includes the nodes needed to evaluate the output nodes, and
  # removes unneeded nodes like those involved in saving and assignment.
  inference_graph = extract_sub_graph(input_graph_def, output_node_names)

  output_graph_def = graph_pb2.GraphDef()
  how_many_converted = 0
  for input_node in inference_graph.node:
    output_node = node_def_pb2.NodeDef()
    if input_node.name in found_variables:
      output_node.op = "Const"
      output_node.name = input_node.name
      dtype = input_node.attr["dtype"]
      data = found_variables[input_node.name]
      output_node.attr["dtype"].CopyFrom(dtype)
      output_node.attr["value"].CopyFrom(attr_value_pb2.AttrValue(
          tensor=tensor_util.make_tensor_proto(data,
                                               dtype=dtype.type,
                                               shape=data.shape)))
      how_many_converted += 1
    else:
      output_node.CopyFrom(input_node)
    output_graph_def.node.extend([output_node])
  print("Converted %d variables to const ops." % how_many_converted)
  return output_graph_def


def remove_training_nodes(input_graph):
  """Prunes out nodes that aren't needed for inference.

  There are nodes like Identity and CheckNumerics that are only useful
  during training, and can be removed in graphs that will be used for
  nothing but inference. Here we identify and remove them, returning an
  equivalent graph. To be specific, CheckNumerics nodes are always removed, and
  Identity nodes that aren't involved in control edges are spliced out so that
  their input and outputs are directly connected.

  Args:
    input_graph: Model to analyze and prune.

  Returns:
    A list of nodes with the unnecessary ones removed.
  """

  types_to_remove = {"CheckNumerics": True}

  input_nodes = input_graph.node
  names_to_remove = {}
  for node in input_nodes:
    if node.op in types_to_remove:
      names_to_remove[node.name] = True

  nodes_after_removal = []
  for node in input_nodes:
    if node.name in names_to_remove:
      continue
    new_node = node_def_pb2.NodeDef()
    new_node.CopyFrom(node)
    input_before_removal = node.input
    del new_node.input[:]
    for full_input_name in input_before_removal:
      input_name = re.sub(r"^\^", "", full_input_name)
      if input_name in names_to_remove:
        continue
      new_node.input.append(full_input_name)
    nodes_after_removal.append(new_node)

  types_to_splice = {"Identity": True}
  names_to_splice = {}
  for node in nodes_after_removal:
    if node.op in types_to_splice:
      # We don't want to remove nodes that have control edge inputs, because
      # they might be involved in subtle dependency issues that removing them
      # will jeopardize.
      has_control_edge = False
      for input_name in node.input:
        if re.match(r"^\^", input_name):
          has_control_edge = True
      if not has_control_edge:
        names_to_splice[node.name] = node.input[0]

  nodes_after_splicing = []
  for node in nodes_after_removal:
    if node.name in names_to_splice:
      continue
    new_node = node_def_pb2.NodeDef()
    new_node.CopyFrom(node)
    input_before_removal = node.input
    del new_node.input[:]
    for full_input_name in input_before_removal:
      input_name = re.sub(r"^\^", "", full_input_name)
      if input_name in names_to_splice:
        new_node.input.append(names_to_splice[input_name])
      else:
        new_node.input.append(full_input_name)
    nodes_after_splicing.append(new_node)

  output_graph = graph_pb2.GraphDef()
  output_graph.node.extend(nodes_after_splicing)
  return output_graph
