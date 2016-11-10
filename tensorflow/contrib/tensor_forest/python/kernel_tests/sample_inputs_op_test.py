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
"""Tests for tf.contrib.tensor_forest.ops.sample_inputs_op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.tensor_forest.python.ops import training_ops

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class SampleInputsTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.input_data = [[-1., 10.], [-10., 2.],  # node 1
                       [20., 50.], [1., -2.]]  # node 2
    self.node_map = [-1, 0, 1]
    self.leaves = [1, 1, 2, 2]
    self.split_features = [[-1, -1, -1], [1, 0, -1], [-1, -1, -1]]
    self.split_thresholds = [[0., 0., 0.], [5., -2., 0.], [0., 0., 0.]]
    self.ops = training_ops.Load()

  def testSimple(self):
    with self.test_session():
      tf.global_variables_initializer().run()
      indices, feature_updates, threshold_updates = (self.ops.sample_inputs(
          self.input_data, [], [], [], [],
          self.node_map,
          self.leaves,
          self.split_features,
          self.split_thresholds,
          split_initializations_per_input=1,
          split_sampling_random_seed=3))
      self.assertAllEqual([1, 0], indices.eval())
      self.assertAllEqual([[1, 0, 1], [0, 0, -1]],
                          feature_updates.eval())
      self.assertAllEqual([[5., -2., 50.], [-1., -10., 0.]],
                          threshold_updates.eval())

  def testSparse(self):
    sparse_shape = [4, 10]
    sparse_indices = [[0, 0], [0, 4], [0, 9],
                      [1, 0], [1, 7],
                      [2, 0],
                      [3, 1], [3, 4]]
    sparse_values = [3.0, -1.0, 0.5,
                     1.5, 6.0,
                     -2.0,
                     -0.5, 2.0]

    with self.test_session():
      tf.global_variables_initializer().run()
      indices, feature_updates, threshold_updates = (self.ops.sample_inputs(
          [],
          sparse_indices,
          sparse_values,
          sparse_shape, [],
          self.node_map,
          self.leaves,
          self.split_features,
          self.split_thresholds,
          split_initializations_per_input=1,
          split_sampling_random_seed=3))
      self.assertAllEqual([1, 0], indices.eval())
      self.assertAllEqual([[1, 0, 0], [4, 7, -1]],
                          feature_updates.eval())
      self.assertAllEqual([[5., -2., -2.], [-1., 6., 0.]],
                          threshold_updates.eval())

  def testWeights(self):
    with self.test_session():
      tf.global_variables_initializer().run()
      indices, feature_updates, threshold_updates = (self.ops.sample_inputs(
          self.input_data, [], [], [], [0.5, 0.1, 0.8, 0.7],
          self.node_map,
          self.leaves,
          self.split_features,
          self.split_thresholds,
          split_initializations_per_input=1,
          split_sampling_random_seed=3))
      self.assertAllEqual([1, 0], indices.eval())
      self.assertAllEqual([[1, 0, 0], [-1, -1, -1]], feature_updates.eval())
      self.assertAllEqual([[5., -2., 20.], [0., 0., 0.]],
                          threshold_updates.eval())

  def testNoAccumulators(self):
    with self.test_session():
      tf.global_variables_initializer().run()
      indices, feature_updates, threshold_updates = (self.ops.sample_inputs(
          self.input_data, [], [], [], [], [-1] * 3,
          self.leaves,
          self.split_features,
          self.split_thresholds,
          split_initializations_per_input=1,
          split_sampling_random_seed=3))
      self.assertAllEqual([], indices.eval())
      self.assertAllEqual((0, 3), feature_updates.eval().shape)
      self.assertAllEqual((0, 3), threshold_updates.eval().shape)

  def testBadInput(self):
    del self.split_features[1]
    with self.test_session():
      tf.global_variables_initializer().run()
      with self.assertRaisesOpError(
          'split_features and split_thresholds should be the same shape.'):
        indices, _, _ = self.ops.sample_inputs(
            self.input_data, [], [], [], [],
            self.node_map,
            self.leaves,
            self.split_features,
            self.split_thresholds,
            split_initializations_per_input=1,
            split_sampling_random_seed=3)
        self.assertAllEqual([], indices.eval())


if __name__ == '__main__':
  googletest.main()
