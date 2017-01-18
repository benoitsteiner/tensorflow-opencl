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
"""Tests for SoftmaxCrossEntropyWithLogits op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class XentTest(test.TestCase):

  def _npXent(self, features, labels, dim=-1):
    if dim is -1:
      dim = len(features.shape) - 1
    one_only_on_dim = list(features.shape)
    one_only_on_dim[dim] = 1
    e = np.exp(features - np.reshape(
        np.amax(
            features, axis=dim), one_only_on_dim))
    probs = e / np.reshape(np.sum(e, axis=dim), one_only_on_dim)
    bp = (probs - labels)
    l = -np.sum(labels * np.log(probs + 1.0e-20), axis=dim)
    return l, bp

  def _testXent(self, np_features, np_labels, use_gpu=False):
    np_loss, np_backprop = self._npXent(np_features, np_labels)
    with self.test_session(use_gpu=use_gpu) as sess:
      loss, backprop = gen_nn_ops._softmax_cross_entropy_with_logits(
          np_features, np_labels)
      tf_loss, tf_backprop = sess.run([loss, backprop])
    self.assertAllCloseAccordingToType(np_loss, tf_loss)
    self.assertAllCloseAccordingToType(np_backprop, tf_backprop)

  def _testXentWrapper(self, np_features, np_labels, dim=-1, use_gpu=False):
    np_loss, _ = self._npXent(np_features, np_labels, dim=dim)
    with self.test_session(use_gpu=use_gpu) as sess:
      loss = nn_ops.softmax_cross_entropy_with_logits(
          labels=np_labels, logits=np_features, dim=dim)
      tf_loss = sess.run(loss)
    print("np_loss:", np_loss)
    print("tf_loss:", tf_loss)
    self.assertAllCloseAccordingToType(np_loss, tf_loss)

  def _testAll(self, features, labels):
    self._testXent(features, labels, use_gpu=False)
    self._testXent(features, labels, use_gpu=True)

  def _testSingleClass(self, use_gpu=False):
    for dtype in np.float16, np.float32:
      with self.test_session(use_gpu=use_gpu) as sess:
        loss, backprop = gen_nn_ops._softmax_cross_entropy_with_logits(
            np.array([[1.], [-1.], [0.]]).astype(dtype),
            np.array([[-1.], [0.], [1.]]).astype(dtype))
        tf_loss, tf_backprop = sess.run([loss, backprop])
      self.assertAllClose([0.0, 0.0, 0.0], tf_loss)
      self.assertAllClose([[2.0], [1.0], [0.0]], tf_backprop)

  def testSingleClass(self):
    self._testSingleClass(True)
    self._testSingleClass(False)

  def testRankTooLarge(self):
    for dtype in np.float16, np.float32:
      np_features = np.array(
          [[[1., 1., 1., 1.]], [[1., 2., 3., 4.]]]).astype(dtype)
      np_labels = np.array(
          [[[0., 0., 0., 1.]], [[0., .5, .5, 0.]]]).astype(dtype)
      self.assertRaisesRegexp(ValueError, "must be rank 2",
                              gen_nn_ops._softmax_cross_entropy_with_logits,
                              np_features, np_labels)

  def testNpXent(self):
    # We create 2 batches of logits for testing.
    # batch 0 is the boring uniform distribution: 1, 1, 1, 1, with target 3.
    # batch 1 has a bit of difference: 1, 2, 3, 4, with soft targets (1, 2).
    features = [[1., 1., 1., 1.], [1., 2., 3., 4.]]
    labels = [[0., 0., 0., 1.], [0., .5, .5, 0.]]

    # For batch 0, we expect the uniform distribution: 0.25, 0.25, 0.25, 0.25
    # With a hard target 3, the backprop is [0.25, 0.25, 0.25, -0.75]
    # The loss for this batch is -log(0.25) = 1.386
    #
    # For batch 1, we have:
    # exp(0) = 1
    # exp(1) = 2.718
    # exp(2) = 7.389
    # exp(3) = 20.085
    # SUM = 31.192
    # So we have as probabilities:
    # exp(0) / SUM = 0.032
    # exp(1) / SUM = 0.087
    # exp(2) / SUM = 0.237
    # exp(3) / SUM = 0.644
    # With a soft target (1, 2), the backprop is
    # [0.032, 0.087 - 0.5 = -0.413, 0.237 - 0.5 = -0.263, 0.644]
    # The loss for this batch is [0.5 * -log(0.087), 0.5 * -log(0.237)]
    # = [1.3862, 1.9401]
    np_loss, np_backprop = self._npXent(np.array(features), np.array(labels))
    self.assertAllClose(
        np.array([[0.25, 0.25, 0.25, -0.75],
                  [0.0321, -0.4129, -0.2632, 0.6439]]),
        np_backprop,
        rtol=1.e-3,
        atol=1.e-3)
    self.assertAllClose(
        np.array([1.3862, 1.9401]), np_loss, rtol=1.e-3, atol=1.e-3)

  def testShapeMismatch(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        gen_nn_ops._softmax_cross_entropy_with_logits(
            [[0., 1.], [2., 3.]], [[0., 1., 0.], [1., 0., 0.]])

  def testNotMatrix(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        gen_nn_ops._softmax_cross_entropy_with_logits([0., 1., 2., 3.],
                                                      [0., 1., 0., 1.])

  def testHalf(self):
    self._testAll(
        np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float16),
        np.array([[0., 0., 0., 1.], [0., .5, .5, 0.]]).astype(np.float16))

  def testFloat(self):
    self._testAll(
        np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float32),
        np.array([[0., 0., 0., 1.], [0., .5, .5, 0.]]).astype(np.float32))

  def testDouble(self):
    self._testAll(
        np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float64),
        np.array([[0., 0., 0., 1.], [0., .5, .5, 0.]]).astype(np.float64))

  def testGradient(self):
    with self.test_session():
      l = constant_op.constant(
          [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5],
          shape=[3, 4],
          dtype=dtypes.float64,
          name="l")
      f = constant_op.constant(
          [0.1, 0.2, 0.3, 0.4, 0.1, 0.4, 0.9, 1.6, 0.1, 0.8, 2.7, 6.4],
          shape=[3, 4],
          dtype=dtypes.float64,
          name="f")
      x = nn_ops.softmax_cross_entropy_with_logits(labels=l, logits=f,
                                                   name="xent")
      err = gradient_checker.compute_gradient_error(f, [3, 4], x, [3])
    print("cross entropy gradient err = ", err)
    self.assertLess(err, 5e-8)

  def testSecondGradient(self):
    with self.test_session():
      l = constant_op.constant([0.0, 0.0, 1.0, 0.0,
                                1.0, 0.0, 0.0, 0.0,
                                0.0, 0.5, 0.0, 0.5], shape=[12],
                               dtype=dtypes.float64, name="l")
      f = constant_op.constant([0.1, 0.2, 0.3, 0.4,
                                0.1, 0.4, 0.9, 1.6,
                                0.1, 0.8, 2.7, 6.4], shape=[12],
                               dtype=dtypes.float64, name="f")
      x = nn_ops.softmax_cross_entropy_with_logits(labels=l, logits=f,
                                                   name="xent")
      loss = math_ops.reduce_mean(x)

    # Taking ths second gradient should fail, since it is not
    # yet supported.
    with self.assertRaisesRegexp(LookupError,
                                 ".*No gradient defined.*PreventGradient.*"):
      _ = gradients_impl.hessians(loss, [f])

  def testWrapper(self):
    features = np.array(
        [[[1., 1., 1., 1.], [1., 2., 3., 4.]],
         [[2., 3., 4., 5.], [6., 7., 8., 9.]],
         [[5., 4., 3., 2.], [1., 2., 3., 4.]]]).astype(np.float32)
    labels = np.array([[[0., 0., 0., 1.], [0., 1., 0., 0.]],
                       [[0., 0.5, 0.5, 0.], [0.5, 0.5, 0., 0.]],
                       [[0., 1., 0., 0.], [0., 0., 1., 0.]]]).astype(np.float32)
    self._testXentWrapper(features, labels, dim=0, use_gpu=False)
    self._testXentWrapper(features, labels, dim=0, use_gpu=True)
    self._testXentWrapper(features, labels, dim=1, use_gpu=False)
    self._testXentWrapper(features, labels, dim=1, use_gpu=True)
    self._testXentWrapper(features, labels, dim=-1, use_gpu=False)
    self._testXentWrapper(features, labels, dim=-1, use_gpu=True)


if __name__ == "__main__":
  test.main()
