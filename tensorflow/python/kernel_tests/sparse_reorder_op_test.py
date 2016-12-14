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

"""Tests for SparseReorder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class SparseReorderTest(tf.test.TestCase):

  def _SparseTensorPlaceholder(self):
    return tf.SparseTensor(
        tf.placeholder(tf.int64),
        tf.placeholder(tf.float64),
        tf.placeholder(tf.int64))

  def _SparseTensorValue_5x6(self, permutation):
    ind = np.array([
        [0, 0],
        [1, 0], [1, 3], [1, 4],
        [3, 2], [3, 3]]).astype(np.int64)
    val = np.array([0, 10, 13, 14, 32, 33]).astype(np.float64)

    ind = ind[permutation]
    val = val[permutation]

    shape = np.array([5, 6]).astype(np.int64)
    return tf.SparseTensorValue(ind, val, shape)

  def testAlreadyInOrder(self):
    with self.test_session(use_gpu=False) as sess:
      input_val = self._SparseTensorValue_5x6(np.arange(6))
      sp_output = tf.sparse_reorder(input_val)

      output_val = sess.run(sp_output)
      self.assertAllEqual(output_val.indices, input_val.indices)
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.shape, input_val.shape)

  def testFeedAlreadyInOrder(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6(np.arange(6))
      sp_output = tf.sparse_reorder(sp_input)

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices, input_val.indices)
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.shape, input_val.shape)

  def testOutOfOrder(self):
    expected_output_val = self._SparseTensorValue_5x6(np.arange(6))
    with self.test_session(use_gpu=False) as sess:
      for _ in range(5):  # To test various random permutations
        input_val = self._SparseTensorValue_5x6(np.random.permutation(6))
        sp_output = tf.sparse_reorder(input_val)

        output_val = sess.run(sp_output)
        self.assertAllEqual(output_val.indices, expected_output_val.indices)
        self.assertAllEqual(output_val.values, expected_output_val.values)
        self.assertAllEqual(output_val.shape, expected_output_val.shape)

  def testFeedOutOfOrder(self):
    expected_output_val = self._SparseTensorValue_5x6(np.arange(6))
    with self.test_session(use_gpu=False) as sess:
      for _ in range(5):  # To test various random permutations
        sp_input = self._SparseTensorPlaceholder()
        input_val = self._SparseTensorValue_5x6(np.random.permutation(6))
        sp_output = tf.sparse_reorder(sp_input)

        output_val = sess.run(sp_output, {sp_input: input_val})
        self.assertAllEqual(output_val.indices, expected_output_val.indices)
        self.assertAllEqual(output_val.values, expected_output_val.values)
        self.assertAllEqual(output_val.shape, expected_output_val.shape)

  def testGradients(self):
    with self.test_session(use_gpu=False):
      for _ in range(5):  # To test various random permutations
        input_val = self._SparseTensorValue_5x6(np.random.permutation(6))
        sp_input = tf.SparseTensor(
            input_val.indices, input_val.values, input_val.dense_shape)
        sp_output = tf.sparse_reorder(sp_input)

        err = tf.test.compute_gradient_error(
            sp_input.values,
            input_val.values.shape,
            sp_output.values,
            input_val.values.shape,
            x_init_value=input_val.values)
        self.assertLess(err, 1e-11)


if __name__ == "__main__":
  tf.test.main()
