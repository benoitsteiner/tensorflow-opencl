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

"""Tests for tensorflow.ops.reshape_op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class ReshapeTest(tf.test.TestCase):

  def _testReshape(self, x, y, use_gpu=False):
    with self.test_session(use_gpu=use_gpu):
      np_ans = x.reshape(y)
      tf_ans = tf.reshape(x, y)
      out = tf_ans.eval()
      self.assertEqual(tf_ans.get_shape(), out.shape)
      self.assertShapeEqual(np_ans, tf_ans)

  def _testBothReshape(self, x, y):
    self._testReshape(x, y, False)
    self._testReshape(x, y, True)

  def testFloatBasic(self):
    x = np.arange(1., 7.).reshape([1, 6]).astype(np.float32)
    self._testBothReshape(x, [2, 3])

  def testDoubleBasic(self):
    x = np.arange(1., 7.).reshape([1, 6]).astype(np.float64)
    self._testBothReshape(x, [2, 3])

  def testInt32Basic(self):
    x = np.arange(1., 7.).reshape([1, 6]).astype(np.int32)
    self._testBothReshape(x, [2, 3])

  def testComplex64Basic(self):
    x = np.arange(1., 7.).reshape([1, 6]).astype(np.complex64)
    self._testBothReshape(x, [2, 3])

  def testComplex128Basic(self):
    x = np.arange(1., 7.).reshape([1, 6]).astype(np.complex128)
    self._testBothReshape(x, [2, 3])

  def testFloatReshapeThreeDimensions(self):
    x = np.arange(1., 28.).reshape([1, 27]).astype(np.float32)
    self._testBothReshape(x, [3, 3, 3])

  def testFloatUnspecifiedDimOnly(self):
    x = np.arange(1., 7.).reshape([6]).astype(np.float32)
    self._testBothReshape(x, [-1])

  def testFloatUnspecifiedDimBegin(self):
    x = np.arange(1., 7.).reshape([6]).astype(np.float32)
    self._testBothReshape(x, [-1, 2])

  def testFloatUnspecifiedDimEnd(self):
    x = np.arange(1., 7.).reshape([6]).astype(np.float32)
    self._testBothReshape(x, [3, -1])

  # TODO(vrv): Add tests for failure conditions once python test_util
  # reports errors.

  def testFloatReshapeGradThreeDimensions(self):
    x = np.arange(1., 25.).reshape([2, 3, 4]).astype(np.float32)
    s = list(np.shape(x))
    with self.test_session():
      input_tensor = tf.constant(x)
      reshape_out = tf.reshape(input_tensor, [1, 8, 3])
      err = tf.test.compute_gradient_error(input_tensor,
                                           s,
                                           reshape_out,
                                           s,
                                           x_init_value=x)
    print("Reshape gradient error = " % err)
    self.assertLess(err, 1e-3)

  def testFloatEmpty(self):
    x = np.empty((0, 0, 0, 0), dtype=np.float32)
    self._testBothReshape(x, [1, 2, 3, 0])
    self._testBothReshape(x, [1, 0, 0, 4])
    self._testBothReshape(x, [0, 0, 0, 0])
    self._testBothReshape(x, [1, 2, 0])
    self._testBothReshape(x, [0, 0, 0])
    self._testBothReshape(x, [1, -1, 5])

  def testErrors(self):
    y = tf.constant(0.0, shape=[23, 29, 31])
    with self.assertRaisesRegexp(ValueError, "must be evenly divisible by 17"):
      tf.reshape(y, [17, -1])

    z = tf.constant(0.0, shape=[32, 128])
    with self.assertRaisesRegexp(ValueError,
                                 "Cannot reshape a tensor with 4096 elements"):
      tf.reshape(z, [4095])

  def testPartialShapes(self):
    x = tf.placeholder(tf.float32)

    # Unknown input shape, partial new shape.
    y = tf.reshape(x, [1, 1, -1, 1])
    self.assertEqual([1, 1, None, 1], y.get_shape().as_list())

    # Unknown input shape, unknown new shape.
    y = tf.reshape(x, tf.placeholder(tf.int32))
    self.assertEqual(None, y.get_shape().ndims)

    # Unknown input shape, known rank for new shape.
    y = tf.reshape(x, tf.placeholder(tf.int32, shape=(3,)))
    self.assertEqual([None, None, None], y.get_shape().as_list())

    # Unknown input shape, partial new shape using `tf.stack()`.
    y = tf.reshape(x, [tf.placeholder(tf.int32), 37])
    self.assertEqual([None, 37], y.get_shape().as_list())

    # Unknown input shape, partial new shape using `tf.concat()`.
    y = tf.reshape(x, tf.concat(0, [tf.placeholder(tf.int32, shape=(2,)),
                                    [37, 42]]))
    self.assertEqual([None, None, 37, 42], y.get_shape().as_list())

    # Unknown input shape, partial new shape using `tf.shape()`.
    y = tf.reshape(x, tf.shape(tf.placeholder(tf.float32,
                                              shape=[None, 37, None])))
    self.assertEqual([None, 37, None], y.get_shape().as_list())


if __name__ == "__main__":
  tf.test.main()
