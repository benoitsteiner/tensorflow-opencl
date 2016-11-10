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
"""Tests for 2D LSTMs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
lstm2d = tf.contrib.ndlstm.lstm2d


def _rand(*size):
  return np.random.uniform(size=size).astype("f")


class Lstm2DTest(test_util.TensorFlowTestCase):

  def testImagesToSequenceDims(self):
    with self.test_session():
      inputs = tf.constant(_rand(2, 7, 11, 5))
      outputs = lstm2d.images_to_sequence(inputs)
      tf.global_variables_initializer().run()
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (11, 14, 5))

  def testSequenceToImagesDims(self):
    with self.test_session():
      inputs = tf.constant(_rand(11, 14, 5))
      outputs = lstm2d.sequence_to_images(inputs, 2)
      tf.global_variables_initializer().run()
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (2, 7, 11, 5))

  def testImagesAndSequenceDims(self):
    with self.test_session():
      size = (2, 7, 11, 5)
      inputs = tf.constant(_rand(*size))
      sequence = lstm2d.images_to_sequence(inputs)
      outputs = lstm2d.sequence_to_images(sequence, size[0])
      tf.global_variables_initializer().run()
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), size)

  def testSeparableLstmDims(self):
    with self.test_session():
      inputs = tf.constant(_rand(2, 7, 11, 5))
      outputs = lstm2d.separable_lstm(inputs, 8)
      tf.global_variables_initializer().run()
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (2, 7, 11, 8))

  def testReduceToSequenceDims(self):
    with self.test_session():
      inputs = tf.constant(_rand(2, 7, 11, 5))
      outputs = lstm2d.reduce_to_sequence(inputs, 8)
      tf.global_variables_initializer().run()
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (2, 11, 8))

  def testReduceToFinalDims(self):
    with self.test_session():
      inputs = tf.constant(_rand(2, 7, 11, 5))
      outputs = lstm2d.reduce_to_final(inputs, 8, 12)
      tf.global_variables_initializer().run()
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (2, 8))


if __name__ == "__main__":
  tf.test.main()
