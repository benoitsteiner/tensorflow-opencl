# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for 1D LSTM."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf
lstm1d = tf.contrib.ndlstm.lstm1d


def _rand(*size):
  return np.random.uniform(size=size).astype("f")


class Lstm1DTest(tf.test.TestCase):

  def testSequenceToSequenceDims(self):
    with self.test_session():
      inputs = tf.constant(_rand(17, 1, 5))
      outputs = lstm1d.ndlstm_base(inputs, 8)
      tf.global_variables_initializer().run()
      names = [v.name for v in tf.trainable_variables()]
      self.assertEqual(len(names), 2)
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (17, 1, 8))

  def testSequenceToSequenceGradient(self):
    with self.test_session():
      size = (17, 1, 15)
      output_size = (17, 1, 8)
      inputs = tf.constant(_rand(*size))
      outputs = lstm1d.ndlstm_base(inputs, 8, dynamic=False)
      tf.global_variables_initializer().run()
      gradients = tf.gradients(outputs, inputs)
      if 1:  # pylint: disable=using-constant-test
        gradients = tf.gradients(outputs, inputs)[0].eval()
        self.assertEqual(gradients.shape, size)
      else:
        # TODO(tmb) tf.test.compute_gradient error is currently broken
        # with dynamic_rnn. Enable this test case eventually.
        err = tf.test.compute_gradient_error(inputs,
                                             size,
                                             outputs,
                                             output_size,
                                             delta=1e-4)
        self.assert_(not np.isnan(err))
        self.assert_(err < 0.1)

  def testSequenceToSequenceGradientReverse(self):
    with self.test_session():
      size = (17, 1, 15)
      output_size = (17, 1, 8)
      inputs = tf.constant(_rand(*size))
      outputs = lstm1d.ndlstm_base(inputs, 8, reverse=1, dynamic=False)
      tf.global_variables_initializer().run()
      if 1:  # pylint: disable=using-constant-test
        gradients = tf.gradients(outputs, inputs)[0].eval()
        self.assertEqual(gradients.shape, size)
      else:
        # TODO(tmb) tf.test.compute_gradient error is currently broken
        # with dynamic_rnn. Enable this test case eventually.
        err = tf.test.compute_gradient_error(inputs,
                                             size,
                                             outputs,
                                             output_size,
                                             delta=1e-4)
        self.assert_(not np.isnan(err))
        self.assert_(err < 0.1)

  def testSequenceToFinalDims(self):
    with self.test_session():
      inputs = tf.constant(_rand(17, 6, 5))
      outputs = lstm1d.sequence_to_final(inputs, 8)
      tf.global_variables_initializer().run()
      names = [v.name for v in tf.trainable_variables()]
      self.assertEqual(len(names), 2)
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (6, 8))

  def testSequenceSoftmaxDims(self):
    with self.test_session():
      inputs = tf.constant(_rand(17, 1, 5))
      outputs = lstm1d.sequence_softmax(inputs, 8)
      tf.global_variables_initializer().run()
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (17, 1, 8))


if __name__ == "__main__":
  tf.test.main()
