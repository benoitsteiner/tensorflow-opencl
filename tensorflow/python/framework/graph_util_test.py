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

"""Tests for tensorflow.python.client.graph_util."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import math_ops  # pylint: disable=unused-import


# Utility device function to use for testing
def test_device_func_pin_variable_to_cpu(op):
  if op.device:
    return op.device
  return "/cpu:0" if op.node_def.op == "Variable" else op.device


class DeviceFunctionsTest(tf.test.TestCase):

  def testTwoDeviceFunctions(self):
    with ops.Graph().as_default() as g:
      var_0 = gen_state_ops._variable(shape=[1], dtype=dtypes.float32, 
          name="var_0", container="", shared_name="")
      with g.device(test_device_func_pin_variable_to_cpu):
        var_1 = gen_state_ops._variable(shape=[1], dtype=dtypes.float32, 
            name="var_1", container="", shared_name="")
      var_2 = gen_state_ops._variable(shape=[1], dtype=dtypes.float32, 
          name="var_2", container="", shared_name="")
      var_3 = gen_state_ops._variable(shape=[1], dtype=dtypes.float32, 
          name="var_3", container="", shared_name="")
      with g.device(test_device_func_pin_variable_to_cpu):
        var_4 = gen_state_ops._variable(shape=[1], dtype=dtypes.float32, 
            name="var_4", container="", shared_name="")
        with g.device("/device:GPU:0"):
          var_5 = gen_state_ops._variable(shape=[1], dtype=dtypes.float32, 
              name="var_5", container="", shared_name="")
        var_6 = gen_state_ops._variable(shape=[1], dtype=dtypes.float32, 
            name="var_6", container="", shared_name="")

    self.assertDeviceEqual(var_0.device, None)
    self.assertDeviceEqual(var_1.device, "/device:CPU:0")
    self.assertDeviceEqual(var_2.device, None)
    self.assertDeviceEqual(var_3.device, None)
    self.assertDeviceEqual(var_4.device, "/device:CPU:0")
    self.assertDeviceEqual(var_5.device, "/device:GPU:0")
    self.assertDeviceEqual(var_6.device, "/device:CPU:0")

  def testNestedDeviceFunctions(self):
    with tf.Graph().as_default():
      var_0 = tf.Variable(0)
      with tf.device(test_device_func_pin_variable_to_cpu):
        var_1 = tf.Variable(1)
        with tf.device(lambda op: "/gpu:0"):
          var_2 = tf.Variable(2)
        with tf.device("/gpu:0"):  # Implicit merging device function.
          var_3 = tf.Variable(3)

    self.assertDeviceEqual(var_0.device, None)
    self.assertDeviceEqual(var_1.device, "/device:CPU:0")
    self.assertDeviceEqual(var_2.device, "/device:GPU:0")
    self.assertDeviceEqual(var_3.device, "/device:GPU:0")

  def testExplicitDevice(self):
    with ops.Graph().as_default() as g:
      const_0 = constant_op.constant(5.0)
      with g.device("/device:GPU:0"):
        const_1 = constant_op.constant(5.0)
      with g.device("/device:GPU:1"):
        const_2 = constant_op.constant(5.0)
      with g.device("/device:CPU:0"):
        const_3 = constant_op.constant(5.0)
      with g.device("/device:CPU:1"):
        const_4 = constant_op.constant(5.0)
      with g.device("/job:ps"):
        const_5 = constant_op.constant(5.0)

    self.assertDeviceEqual(const_0.device, None)
    self.assertDeviceEqual(const_1.device, "/device:GPU:0")
    self.assertDeviceEqual(const_2.device, "/device:GPU:1")
    self.assertDeviceEqual(const_3.device, "/device:CPU:0")
    self.assertDeviceEqual(const_4.device, "/device:CPU:1")
    self.assertDeviceEqual(const_5.device, "/job:ps")

  def testDefaultDevice(self):
    with ops.Graph().as_default() as g, g.device(
        test_device_func_pin_variable_to_cpu):
      with g.device("/job:ps"):
        const_0 = constant_op.constant(5.0)
      with g.device("/device:GPU:0"):
        const_1 = constant_op.constant(5.0)
      with g.device("/device:GPU:1"):
        const_2 = constant_op.constant(5.0)
      with g.device("/device:CPU:0"):
        const_3 = constant_op.constant(5.0)
      with g.device("/device:CPU:1"):
        const_4 = constant_op.constant(5.0)
      with g.device("/replica:0"):
        const_5 = constant_op.constant(5.0)

    self.assertDeviceEqual(const_0.device, "/job:ps")
    self.assertDeviceEqual(const_1.device, "/device:GPU:0")
    self.assertDeviceEqual(const_2.device, "/device:GPU:1")
    self.assertDeviceEqual(const_3.device, "/device:CPU:0")
    self.assertDeviceEqual(const_4.device, "/device:CPU:1")
    self.assertDeviceEqual(const_5.device, "/replica:0")

  def testExtractSubGraph(self):
    graph_def = tf.GraphDef()
    n1 = graph_def.node.add()
    n1.name = "n1"
    n1.input.extend(["n5"])
    n2 = graph_def.node.add()
    n2.name = "n2"
    # Take the first output of the n1 node as the input.
    n2.input.extend(["n1:0"])
    n3 = graph_def.node.add()
    n3.name = "n3"
    # Add a control input (which isn't really needed by the kernel, but
    # rather to enforce execution order between nodes).
    n3.input.extend(["^n2"])
    n4 = graph_def.node.add()
    n4.name = "n4"

    # It is fine to have a loops in the graph as well.
    n5 = graph_def.node.add()
    n5.name = "n5"
    n5.input.extend(["n1"])

    sub_graph = graph_util.extract_sub_graph(graph_def, ["n3"])
    self.assertEqual("n1", sub_graph.node[0].name)
    self.assertEqual("n2", sub_graph.node[1].name)
    self.assertEqual("n3", sub_graph.node[2].name)
    self.assertEqual("n5", sub_graph.node[3].name)

  def testConvertVariablesToConsts(self):
    with tf.Graph().as_default():
      variable_node = tf.Variable(1.0, name="variable_node")
      _ = tf.Variable(1.0, name="unused_variable_node")
      output_node = tf.mul(variable_node, 2.0, name="output_node")
      with tf.Session() as sess:
        init = tf.initialize_variables([variable_node])
        sess.run(init)
        output = sess.run(output_node)
        self.assertNear(2.0, output, 0.00001)
        variable_graph_def = sess.graph.as_graph_def()
        # First get the constant_graph_def when variable_names_whitelist is set,
        # note that if variable_names_whitelist is not set an error will be
        # thrown because unused_variable_node is not initialized.
        constant_graph_def = graph_util.convert_variables_to_constants(
            sess, variable_graph_def, ["output_node"],
            variable_names_whitelist=set(["variable_node"]))

        # Then initialize the unused variable, and get another
        # constant_graph_def when variable_names_whitelist is not set.
        sess.run(tf.initialize_all_variables())
        constant_graph_def_without_variable_whitelist = (
            graph_util.convert_variables_to_constants(
                sess, variable_graph_def, ["output_node"]))

        # The unused variable should be cleared so the two graphs should be
        # equivalent.
        self.assertEqual(str(constant_graph_def),
                         str(constant_graph_def_without_variable_whitelist))

    # Now we make sure the variable is now a constant, and that the graph still
    # produces the expected result.
    with tf.Graph().as_default():
      _ = tf.import_graph_def(constant_graph_def, name="")
      self.assertEqual(4, len(constant_graph_def.node))
      for node in constant_graph_def.node:
        self.assertNotEqual("Variable", node.op)
      with tf.Session() as sess:
        output_node = sess.graph.get_tensor_by_name("output_node:0")
        output = sess.run(output_node)
        self.assertNear(2.0, output, 0.00001)

  def create_node_def(self, op, name, inputs):
    new_node = tf.NodeDef()
    new_node.op = op
    new_node.name = name
    for input_name in inputs:
      new_node.input.extend([input_name])
    return new_node

  def create_constant_node_def(self, name, value, dtype, shape=None):
    node = self.create_node_def("Const", name, [])
    self.set_attr_dtype(node, "dtype", dtype)
    self.set_attr_tensor(node, "value", value, dtype, shape)
    return node

  def set_attr_dtype(self, node, key, value):
    node.attr[key].CopyFrom(tf.AttrValue(type=value.as_datatype_enum))

  def set_attr_tensor(self, node, key, value, dtype, shape=None):
    node.attr[key].CopyFrom(tf.AttrValue(
        tensor=tensor_util.make_tensor_proto(value,
                                             dtype=dtype,
                                             shape=shape)))

  def testRemoveTrainingNodes(self):
    a_constant_name = "a_constant"
    b_constant_name = "b_constant"
    a_check_name = "a_check"
    b_check_name = "b_check"
    a_identity_name = "a_identity"
    b_identity_name = "b_identity"
    add_name = "add"
    graph_def = tf.GraphDef()
    a_constant = self.create_constant_node_def(a_constant_name,
                                               value=1,
                                               dtype=tf.float32,
                                               shape=[])
    graph_def.node.extend([a_constant])
    a_check_node = self.create_node_def("CheckNumerics", a_check_name,
                                        [a_constant_name])
    graph_def.node.extend([a_check_node])
    a_identity_node = self.create_node_def("Identity", a_identity_name,
                                           [a_constant_name,
                                            "^" + a_check_name])
    graph_def.node.extend([a_identity_node])
    b_constant = self.create_constant_node_def(b_constant_name,
                                               value=1,
                                               dtype=tf.float32,
                                               shape=[])
    graph_def.node.extend([b_constant])
    b_check_node = self.create_node_def("CheckNumerics", b_check_name,
                                        [b_constant_name])
    graph_def.node.extend([b_check_node])
    b_identity_node = self.create_node_def("Identity", b_identity_name,
                                           [b_constant_name,
                                            "^" + b_check_name])
    graph_def.node.extend([b_identity_node])
    add_node = self.create_node_def("Add", add_name,
                                    [a_identity_name,
                                     b_identity_name])
    self.set_attr_dtype(add_node, "T", tf.float32)
    graph_def.node.extend([add_node])

    expected_output = tf.GraphDef()
    a_constant = self.create_constant_node_def(a_constant_name,
                                               value=1,
                                               dtype=tf.float32,
                                               shape=[])
    expected_output.node.extend([a_constant])
    b_constant = self.create_constant_node_def(b_constant_name,
                                               value=1,
                                               dtype=tf.float32,
                                               shape=[])
    expected_output.node.extend([b_constant])
    add_node = self.create_node_def("Add", add_name,
                                    [a_constant_name,
                                     b_constant_name])
    self.set_attr_dtype(add_node, "T", tf.float32)
    expected_output.node.extend([add_node])

    output = graph_util.remove_training_nodes(graph_def)
    self.assertProtoEquals(expected_output, output)


if __name__ == "__main__":
  tf.test.main()
