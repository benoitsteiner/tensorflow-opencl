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
"""Tests for tf.contrib.tensor_forest.ops.grow_tree_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, 'getdlopenflags') and hasattr(sys, 'setdlopenflags'):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

from tensorflow.contrib.tensor_forest.python.ops import tensor_forest_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


class GrowTreeTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.tree = variables.Variable([[1, 0], [-1, 0], [-1, 0], [-2, 0], [-2, 0],
                                    [-2, 0], [-2, 0]])
    self.tree_thresholds = variables.Variable([0., 0., 0., 0., 0., 0., 0.])
    self.eot = variables.Variable([3])
    self.node_map = [-1, 0, 1, -1, -1, -1, -1]
    self.finished = [1, 2]
    self.best_splits = [2, 3]
    self.split_features = [[1, 2, 3, 4], [5, 6, 7, 8]]
    self.split_thresholds = [[10., 20., 30., 40.], [50., 60., 70., 80.]]

  def testSimple(self):
    with self.test_session():
      variables.global_variables_initializer().run()
      update_list, tree_updates, threshold_updates, new_eot = (
          tensor_forest_ops.grow_tree(self.eot, self.node_map, self.finished,
                                      self.best_splits, self.split_features,
                                      self.split_thresholds))

      self.assertAllEqual([1, 3, 4, 2, 5, 6], update_list.eval())
      self.assertAllEqual(
          [[3, 3], [-1, -1], [-1, -1], [5, 8], [-1, -1], [-1, -1]],
          tree_updates.eval())
      self.assertAllEqual([30.0, 0.0, 0.0, 80.0, 0.0, 0.0],
                          threshold_updates.eval())
      self.assertAllEqual([7], new_eot.eval())

  def testNoRoomToGrow(self):
    with self.test_session():
      variables.global_variables_initializer().run()
      # Even though there's one free node, there needs to be 2 to grow.
      state_ops.assign(self.eot, [6]).eval()

      update_list, tree_updates, threshold_updates, new_eot = (
          tensor_forest_ops.grow_tree(self.eot, self.node_map, self.finished,
                                      self.best_splits, self.split_features,
                                      self.split_thresholds))

      self.assertAllEqual([], update_list.eval())
      self.assertEquals((0, 2), tree_updates.eval().shape)
      self.assertAllEqual([], threshold_updates.eval())
      self.assertAllEqual([6], new_eot.eval())

  def testNoFinished(self):
    with self.test_session():
      variables.global_variables_initializer().run()

      update_list, tree_updates, threshold_updates, new_eot = (
          tensor_forest_ops.grow_tree(self.eot, self.node_map, [], [],
                                      self.split_features,
                                      self.split_thresholds))

      self.assertAllEqual([], update_list.eval())
      self.assertAllEqual((0, 2), tree_updates.eval().shape)
      self.assertAllEqual([], threshold_updates.eval())
      self.assertAllEqual([3], new_eot.eval())

  def testBadInput(self):
    with self.test_session():
      variables.global_variables_initializer().run()
      with self.assertRaisesOpError(
          'Number of finished nodes should be the same in finished and '
          'best_splits.'):
        update_list, _, _, _ = (tensor_forest_ops.grow_tree(
            self.eot, self.node_map, [], self.best_splits, self.split_features,
            self.split_thresholds))
        self.assertAllEqual([], update_list.eval())


if __name__ == '__main__':
  googletest.main()
