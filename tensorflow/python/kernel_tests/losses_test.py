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
"""Tests for losses."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.losses.python.losses import loss_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.ops.losses import util
from tensorflow.python.platform import test
from tensorflow.python.training import momentum as momentum_lib


class AbsoluteDifferenceLossTest(test.TestCase):

  def setUp(self):
    self._predictions = constant_op.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
    self._labels = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))

  def testValueErrorThrownWhenWeightIsNone(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        losses.absolute_difference(
            self._predictions, self._predictions, weights=None)

  def testAllCorrectNoLossWeight(self):
    loss = losses.absolute_difference(self._predictions, self._predictions)
    with self.test_session():
      self.assertAlmostEqual(0.0, loss.eval(), 3)

  def testNonZeroLoss(self):
    loss = losses.absolute_difference(self._labels, self._predictions)
    with self.test_session():
      self.assertAlmostEqual(5.5, loss.eval(), 3)

  def testNonZeroLossWithPythonScalarWeight(self):
    weights = 2.3
    loss = losses.absolute_difference(self._labels, self._predictions, weights)
    with self.test_session():
      self.assertAlmostEqual(5.5 * weights, loss.eval(), 3)

  def testNonZeroLossWithScalarTensorWeight(self):
    weights = 2.3
    loss = losses.absolute_difference(self._labels, self._predictions,
                                      constant_op.constant(weights))
    with self.test_session():
      self.assertAlmostEqual(5.5 * weights, loss.eval(), 3)

  def testNonZeroLossWithOneDimBatchSpecificWeights(self):
    weights = constant_op.constant([1.2, 0.0], shape=[2,])
    loss = losses.absolute_difference(self._labels, self._predictions, weights)
    with self.test_session():
      self.assertAlmostEqual(5.6, loss.eval(), 3)

  def testNonZeroLossWithTwoDimBatchSpecificWeights(self):
    weights = constant_op.constant([1.2, 0.0], shape=[2, 1])
    loss = losses.absolute_difference(self._labels, self._predictions, weights)
    with self.test_session():
      self.assertAlmostEqual(5.6, loss.eval(), 3)

  def testNonZeroLossWithSampleSpecificWeights(self):
    weights = constant_op.constant([3, 6, 5, 0, 4, 2], shape=[2, 3])
    loss = losses.absolute_difference(self._labels, self._predictions, weights)
    with self.test_session():
      self.assertAlmostEqual(16.6, loss.eval(), 3)

  def testNonZeroLossWithSampleSpecificWeightsMostZero(self):
    weights = constant_op.constant([0, 0, 0, 0, 0, 2], shape=[2, 3])
    loss = losses.absolute_difference(self._labels, self._predictions, weights)
    with self.test_session():
      self.assertAlmostEqual(6.0, loss.eval(), 3)

  def testLossWithSampleSpecificWeightsAllZero(self):
    weights = array_ops.zeros((2, 3))
    loss = losses.absolute_difference(self._labels, self._predictions, weights)
    with self.test_session():
      self.assertAlmostEqual(0.0, loss.eval(), 3)


class SoftmaxCrossEntropyLossTest(test.TestCase):

  def testNoneWeightRaisesValueError(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    with self.test_session():
      with self.assertRaises(ValueError):
        losses.softmax_cross_entropy(labels, logits, weights=None)

  def testAllCorrect(self):
    with self.test_session():
      logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                     [0.0, 0.0, 10.0]])
      labels = constant_op.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
      loss = losses.softmax_cross_entropy(labels, logits)
      self.assertEquals('softmax_cross_entropy_loss/value', loss.op.name)
      self.assertAlmostEqual(loss.eval(), 0.0, 3)

  def testAllWrong(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    with self.test_session():
      loss = losses.softmax_cross_entropy(labels, logits)
      self.assertEquals(loss.op.name, 'softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(loss.eval(), 10.0, 3)

  def testNonZeroLossWithPythonScalarWeight(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    weights = 2.3
    with self.test_session():
      loss = losses.softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual(weights * 10.0, loss.eval(), 3)

  def testNonZeroLossWithScalarTensorWeight(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    weights = 2.3
    with self.test_session():
      loss = losses.softmax_cross_entropy(labels, logits,
                                          constant_op.constant(weights))
      self.assertAlmostEqual(weights * 10.0, loss.eval(), 3)

  def testNonZeroLossWithOneDimBatchSpecificWeights(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    weights = constant_op.constant([1.2, 3.4, 5.6], shape=[3])
    with self.test_session():
      loss = losses.softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual((1.2 + 3.4 + 5.6) * 10.0 / 3.0, loss.eval(), 3)

  def testAllWrongAllWeightsMissing(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    weights = constant_op.constant([0, 0, 0], shape=[3])
    with self.test_session():
      loss = losses.softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual(0.0, loss.eval(), 3)

  def testSomeWeightsMissing(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    weights = constant_op.constant([1.2, 0, 0], shape=[3])
    with self.test_session():
      loss = losses.softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual(12.0, loss.eval(), 3)

  def testSoftmaxWithMeasurementSpecificWeightsRaisesException(self):
    with self.test_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0],
                                     [-100.0, 100.0, -100.0],
                                     [-100.0, -100.0, 100.0]])
      labels = constant_op.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
      weights = constant_op.constant([[3, 4, 5], [2, 6, 0], [8, 0, 1]])

      with self.assertRaises(ValueError):
        losses.softmax_cross_entropy(labels, logits, weights=weights).eval()

  def testSoftmaxLabelSmoothing(self):
    with self.test_session():
      # Softmax Cross Entropy Loss is:
      #   -\sum_i p_i \log q_i
      # where for a softmax activation
      # \log q_i = x_i - \log \sum_j \exp x_j
      #          = x_i - x_max - \log \sum_j \exp (x_j - x_max)
      # For our activations, [100, -100, -100] the log partion function becomes
      # \log ( exp(0) + exp(-200) + exp(-200) ) = 0
      # so our log softmaxes become: [0, -200, -200]
      # so our cross entropy loss is:
      # -(1 - L + L/n) * 0 + 400 * L/n = 400 L/n
      logits = constant_op.constant([[100.0, -100.0, -100.0]])
      labels = constant_op.constant([[1, 0, 0]])
      label_smoothing = 0.1
      loss = losses.softmax_cross_entropy(
          labels, logits, label_smoothing=label_smoothing)
      self.assertEquals(loss.op.name, 'softmax_cross_entropy_loss/value')
      expected_value = 400.0 * label_smoothing / 3.0
      self.assertAlmostEqual(loss.eval(), expected_value, 3)


class SparseSoftmaxCrossEntropyLossTest(test.TestCase):

  def testNoneWeightRaisesValueError(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[0], [1], [2]])
    with self.test_session():
      with self.assertRaises(ValueError):
        losses.sparse_softmax_cross_entropy(labels, logits, weights=None)

  def testAllCorrectInt32Labels(self):
    with self.test_session():
      logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                     [0.0, 0.0, 10.0]])
      labels = constant_op.constant([[0], [1], [2]], dtype=dtypes.int32)
      loss = losses.sparse_softmax_cross_entropy(labels, logits)
      self.assertEquals(loss.op.name, 'sparse_softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(loss.eval(), 0.0, 3)

  def testAllCorrectInt64Labels(self):
    with self.test_session():
      logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                     [0.0, 0.0, 10.0]])
      labels = constant_op.constant([[0], [1], [2]], dtype=dtypes.int64)
      loss = losses.sparse_softmax_cross_entropy(labels, logits)
      self.assertEquals(loss.op.name, 'sparse_softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(loss.eval(), 0.0, 3)

  def testAllCorrectNonColumnLabels(self):
    with self.test_session():
      logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                     [0.0, 0.0, 10.0]])
      labels = constant_op.constant([0, 1, 2])
      loss = losses.sparse_softmax_cross_entropy(labels, logits)
      self.assertEquals(loss.op.name, 'sparse_softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(loss.eval(), 0.0, 3)

  def testAllWrongInt32Labels(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[2], [0], [1]], dtype=dtypes.int32)

    with self.test_session():
      loss = losses.sparse_softmax_cross_entropy(labels, logits)
      self.assertEquals(loss.op.name, 'sparse_softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(loss.eval(), 10.0, 3)

  def testAllWrongInt64Labels(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[2], [0], [1]], dtype=dtypes.int64)

    with self.test_session():
      loss = losses.sparse_softmax_cross_entropy(labels, logits)
      self.assertEquals(loss.op.name, 'sparse_softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(loss.eval(), 10.0, 3)

  def testAllWrongNonColumnLabels(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([2, 0, 1])

    with self.test_session():
      loss = losses.sparse_softmax_cross_entropy(labels, logits)
      self.assertEquals(loss.op.name, 'sparse_softmax_cross_entropy_loss/value')
      self.assertAlmostEqual(loss.eval(), 10.0, 3)

  def testNonZeroLossWithPythonScalarWeight(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[2], [0], [1]])
    weights = 2.3
    with self.test_session():
      loss = losses.sparse_softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual(weights * 10.0, loss.eval(), 3)

  def testNonZeroLossWithScalarTensorWeight(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[2], [0], [1]])
    weights = 2.3
    with self.test_session():
      loss = losses.sparse_softmax_cross_entropy(labels, logits,
                                                 constant_op.constant(weights))
      self.assertAlmostEqual(weights * 10.0, loss.eval(), 3)

  def testNonZeroLossWithOneDimBatchSpecificWeights(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[2], [0], [1]])
    weights = constant_op.constant([1.2, 3.4, 5.6], shape=[3])
    with self.test_session():
      loss = losses.sparse_softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual((1.2 + 3.4 + 5.6) * 10.0 / 3.0, loss.eval(), 3)

  def testNonZeroLossWithColumnWeights(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[2], [0], [1]])
    weights = constant_op.constant([[1.2], [3.4], [5.6]])
    with self.test_session():
      loss = losses.sparse_softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual((1.2 + 3.4 + 5.6) * 10.0 / 3.0, loss.eval(), 3)

  def testAllWrongAllWeightsMissing(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[2], [0], [1]])
    weights = constant_op.constant([0, 0, 0], shape=[3])
    with self.test_session():
      loss = losses.sparse_softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual(0.0, loss.eval(), 3)

  def testSomeWeightsMissing(self):
    logits = constant_op.constant([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                                   [0.0, 0.0, 10.0]])
    labels = constant_op.constant([[2], [0], [1]])
    weights = constant_op.constant([1.2, 0, 0], shape=[3])
    with self.test_session():
      loss = losses.sparse_softmax_cross_entropy(labels, logits, weights)
      self.assertAlmostEqual(12.0, loss.eval(), 3)

  def testMeasurementSpecificWeightsRaisesException(self):
    with self.test_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0],
                                     [-100.0, 100.0, -100.0],
                                     [-100.0, -100.0, 100.0]])
      labels = constant_op.constant([[0], [1], [2]])
      weights = constant_op.constant([[3, 4, 5], [2, 6, 0], [8, 0, 1]])

      with self.assertRaises(ValueError):
        losses.sparse_softmax_cross_entropy(
            labels, logits, weights=weights).eval()

  def testInconsistentWeightSizeRaisesException(self):
    """The weight tensor has incorrect number of elements."""
    with self.test_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0],
                                     [-100.0, 100.0, -100.0],
                                     [-100.0, -100.0, 100.0]])
      labels = constant_op.constant([[0], [1], [2]])
      weights = constant_op.constant([1.2, 3.4, 5.6, 7.8])

      with self.assertRaises(ValueError):
        losses.sparse_softmax_cross_entropy(
            labels, logits, weights=weights).eval()

  def testInconsistentLabelSizeRaisesException(self):
    """The label tensor has incorrect number of elements."""
    with self.test_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0],
                                     [-100.0, 100.0, -100.0],
                                     [-100.0, -100.0, 100.0]])
      labels = constant_op.constant([[0], [1], [2], [3]])
      weights = constant_op.constant([1.2, 3.4, 5.6])

      with self.assertRaises(ValueError):
        losses.sparse_softmax_cross_entropy(
            labels, logits, weights=weights).eval()

  def testInconsistentWeightShapeRaisesException(self):
    """The weight tensor has incorrect shape."""
    with self.test_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0, -100.0],
                                     [-100.0, 100.0, -100.0, -100.0],
                                     [-100.0, -100.0, 100.0, -100.0],
                                     [-100.0, -100.0, -100.0, 100.0]])
      labels = constant_op.constant([[0], [1], [2], [3]])
      weights = constant_op.constant([[1.2, 3.4], [5.6, 7.8]])

      with self.assertRaises(ValueError):
        losses.sparse_softmax_cross_entropy(
            labels, logits, weights=weights).eval()

  def testInconsistentLabelShapeRaisesException(self):
    """The label tensor has incorrect shape."""
    with self.test_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0, -100.0],
                                     [-100.0, 100.0, -100.0, -100.0],
                                     [-100.0, -100.0, 100.0, -100.0],
                                     [-100.0, -100.0, -100.0, 100.0]])
      labels = constant_op.constant([[0, 1], [2, 3]])
      weights = constant_op.constant([1.2, 3.4, 5.6, 7.8])

      with self.assertRaises(errors_impl.InvalidArgumentError):
        losses.sparse_softmax_cross_entropy(
            labels, logits, weights=weights).eval()


class SigmoidCrossEntropyLossTest(test.TestCase):

  def testAllCorrectSigmoid(self):
    with self.test_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0],
                                     [-100.0, 100.0, -100.0],
                                     [-100.0, -100.0, 100.0]])
      labels = constant_op.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
      loss = losses.sigmoid_cross_entropy(labels, logits)
      self.assertEquals(loss.op.name, 'sigmoid_cross_entropy_loss/value')
      self.assertAlmostEqual(0.0, loss.eval(), 3)

  def testLossWithSingleDimPlaceholderForLogitsAndWeights1(self):
    logits = array_ops.placeholder(dtypes.float32, shape=(None, 1))
    labels = array_ops.placeholder(dtypes.float32, shape=(None, 1))
    weights = array_ops.ones_like(logits, dtype=dtypes.float32)

    loss = losses.sigmoid_cross_entropy(labels, logits, weights)

    with self.test_session() as sess:
      loss = sess.run(loss,
                      feed_dict={
                          logits: np.ones((32, 1)),
                          labels: np.ones((32, 1)),
                      })
      self.assertAlmostEqual(0.313, loss, 3)

  def testLossWithSingleDimPlaceholderForLogitsAndWeights2(self):
    logits = array_ops.placeholder(dtypes.float32, shape=(None, 2))
    labels = array_ops.placeholder(dtypes.float32, shape=(None, 2))
    weights = array_ops.ones_like(logits, dtype=dtypes.float32)

    loss = losses.sigmoid_cross_entropy(labels, logits, weights)

    with self.test_session() as sess:
      loss = sess.run(loss,
                      feed_dict={
                          logits: np.ones((32, 2)),
                          labels: np.ones((32, 2)),
                      })
      self.assertAlmostEqual(0.313, loss, 3)

  def testAllWrongSigmoid(self):
    with self.test_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0],
                                     [-100.0, 100.0, -100.0],
                                     [-100.0, -100.0, 100.0]])
      labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
      loss = losses.sigmoid_cross_entropy(labels, logits)
      self.assertEquals(loss.op.name, 'sigmoid_cross_entropy_loss/value')
      self.assertAlmostEqual(loss.eval(), 600.0 / 9.0, 3)

  def testAllWrongSigmoidWithMeasurementSpecificWeights(self):
    with self.test_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0],
                                     [-100.0, 100.0, -100.0],
                                     [-100.0, -100.0, 100.0]])
      labels = constant_op.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
      weights = constant_op.constant([[3, 4, 5], [2, 6, 0], [8, 0, 1]])
      loss = losses.sigmoid_cross_entropy(labels, logits, weights)
      self.assertEquals(loss.op.name, 'sigmoid_cross_entropy_loss/value')
      self.assertAlmostEqual(1700.0 / 7.0, loss.eval(), 3)

  def testMultiCorrectSigmoid(self):
    logits = constant_op.constant([[100.0, -100.0, 100.0],
                                   [100.0, 100.0, -100.0],
                                   [-100.0, 100.0, 100.0]])
    labels = constant_op.constant([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
    loss = losses.sigmoid_cross_entropy(labels, logits)
    self.assertEquals(loss.op.name, 'sigmoid_cross_entropy_loss/value')

    with self.test_session():
      self.assertAlmostEqual(loss.eval(), 0.0, 3)

  def testSigmoidLabelSmoothingCorrect(self):
    with self.test_session():
      logits = constant_op.constant([[100.0, -100.0, -100.0]])
      labels = constant_op.constant([[1, 0, 1]])
      # Sigmoid cross entropy loss is:
      #   max(x,0) - x*z + log(1 + exp(-abs(x)))
      # The new labels are:
      #    z' = z * (1 - L) + 0.5 L
      #    1 -> 1 - 0.5 L
      #    0 -> 0.5 L
      # here we expect:
      # 1/3 * (100 - 100 * (1 - 0.5 L)  + 0
      #       + 0  + 100 * (0.5 L)      + 0
      #       + 0  + 100 * (1 - 0.5 L)  + 0)
      # = 1/3 * (100 + 50 L)
      label_smoothing = 0.1
      loss = losses.sigmoid_cross_entropy(
          labels, logits, label_smoothing=label_smoothing)
      self.assertEquals(loss.op.name, 'sigmoid_cross_entropy_loss/value')
      expected_value = (100.0 + 50.0 * label_smoothing) / 3.0
      self.assertAlmostEqual(loss.eval(), expected_value, 3)

  def testSigmoidLabelSmoothingEqualsSoftmaxTwoLabel(self):
    with self.test_session():
      label_smoothing = 0.1
      sigmoid_logits = constant_op.constant([[100.0, -100.0, -100.0]])
      sigmoid_labels = constant_op.constant([[1, 0, 1]])
      sigmoid_loss = losses.sigmoid_cross_entropy(
          sigmoid_labels, sigmoid_logits, label_smoothing=label_smoothing)

      softmax_logits = constant_op.constant(
          [[0.0, 100.0], [100.0, 0.0], [100.0, 0.0]])
      softmax_labels = constant_op.constant([[0, 1], [1, 0], [0, 1]])
      softmax_loss = losses.softmax_cross_entropy(
          softmax_labels, softmax_logits, label_smoothing=label_smoothing)
      self.assertAlmostEqual(sigmoid_loss.eval(), softmax_loss.eval(), 3)


class LogLossTest(test.TestCase):

  def setUp(self):
    predictions = np.asarray([.9, .2, .2, .8, .4, .6]).reshape((2, 3))
    labels = np.asarray([1.0, 0.0, 1.0, 1.0, 0.0, 0.0]).reshape((2, 3))

    self._np_predictions = predictions
    self._np_labels = labels

    epsilon = 1e-7
    self._expected_losses = np.multiply(
        labels, np.log(predictions + epsilon)) + np.multiply(
            1 - labels, np.log(1 - predictions + epsilon))

    self._predictions = constant_op.constant(predictions)
    self._labels = constant_op.constant(labels)

  def testValueErrorThrownWhenWeightIsNone(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        losses.log_loss(self._labels, self._labels, weights=None)

  def testAllCorrectNoLossWeight(self):
    loss = losses.log_loss(self._labels, self._labels)
    with self.test_session():
      self.assertAlmostEqual(0.0, loss.eval(), 3)

  def testAllCorrectNoLossWeightWithPlaceholder(self):
    tf_predictions = array_ops.placeholder(
        dtypes.float32, shape=self._np_labels.shape)
    loss = losses.log_loss(self._labels, tf_predictions)
    with self.test_session():
      self.assertAlmostEqual(
          0.0, loss.eval(feed_dict={tf_predictions: self._np_labels}), 3)

  def testNonZeroLoss(self):
    loss = losses.log_loss(self._labels, self._predictions)
    with self.test_session():
      self.assertAlmostEqual(-np.sum(self._expected_losses) / 6.0,
                             loss.eval(), 3)

  def testNonZeroLossWithPythonScalarWeight(self):
    weights = 2.3
    loss = losses.log_loss(self._labels, self._predictions, weights)
    with self.test_session():
      self.assertAlmostEqual(weights * -np.sum(self._expected_losses) / 6.0,
                             loss.eval(), 3)

  def testNonZeroLossWithScalarTensorWeight(self):
    weights = 2.3
    loss = losses.log_loss(self._labels, self._predictions,
                           constant_op.constant(weights))
    with self.test_session():
      self.assertAlmostEqual(weights * -np.sum(self._expected_losses) / 6.0,
                             loss.eval(), 3)

  def testNonZeroLossWithScalarTensorWeightAndPlaceholder(self):
    tf_predictions = array_ops.placeholder(
        dtypes.float32, shape=self._np_predictions.shape)
    weights = 2.3
    loss = losses.log_loss(self._labels, tf_predictions,
                           constant_op.constant(weights))
    with self.test_session() as sess:
      loss = sess.run(loss, feed_dict={tf_predictions: self._np_predictions})
      self.assertAlmostEqual(weights * -np.sum(self._expected_losses) / 6.0,
                             loss, 3)

  def testNonZeroLossWithScalarTensorWeightAndPlaceholderWithRankOnly(self):
    tf_predictions = array_ops.placeholder(dtypes.float32, shape=[None, None])
    weights = 2.3
    loss = losses.log_loss(self._labels, tf_predictions,
                           constant_op.constant(weights))
    with self.test_session() as sess:
      loss = sess.run(loss, feed_dict={tf_predictions: self._np_predictions})
      self.assertAlmostEqual(weights * -np.sum(self._expected_losses) / 6.0,
                             loss, 3)

  def testNonZeroLossWithOneDimBatchSpecificWeights(self):
    weights = constant_op.constant([1.2, 3.4], shape=[2])
    expected_losses = np.multiply(
        self._expected_losses,
        np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape((2, 3)))
    loss = losses.log_loss(self._labels, self._predictions, weights)
    with self.test_session():
      self.assertAlmostEqual(-np.sum(expected_losses) / 6.0, loss.eval(), 3)

  def testNonZeroLossWithOneDimBatchSpecificWeightsSomeZero(self):
    weights = constant_op.constant([1.2, 0], shape=[2])
    expected_losses = np.multiply(self._expected_losses,
                                  np.asarray([1.2, 1.2, 1.2, 0, 0, 0]).reshape(
                                      (2, 3)))
    loss = losses.log_loss(self._labels, self._predictions, weights)
    with self.test_session():
      self.assertAlmostEqual(-np.sum(expected_losses) / 3.0, loss.eval(), 3)

  def testNonZeroLossWithTwoDimBatchSpecificWeightsSomeZero(self):
    weights = constant_op.constant([1.2, 0], shape=[2, 1])
    expected_losses = np.multiply(self._expected_losses,
                                  np.asarray([1.2, 1.2, 1.2, 0, 0, 0]).reshape(
                                      (2, 3)))
    loss = losses.log_loss(self._labels, self._predictions, weights)
    with self.test_session():
      self.assertAlmostEqual(-np.sum(expected_losses) / 3.0, loss.eval(), 3)

  def testWeightsWithSameNumDimsButWrongShapeThrowsException(self):
    weights = constant_op.constant(np.random.normal(size=(2, 4)), shape=[2, 4])
    with self.test_session():
      with self.assertRaises(ValueError):
        losses.log_loss(self._labels, self._predictions, weights)

  def testNonZeroLossWithMeasurementSpecificWeights(self):
    weights = np.array([3, 6, 5, 0, 4, 2]).reshape((2, 3))
    expected_losses = np.multiply(self._expected_losses, weights)

    loss = losses.log_loss(
        self._labels,
        self._predictions,
        constant_op.constant(
            weights, shape=(2, 3)))
    with self.test_session():
      self.assertAlmostEqual(-np.sum(expected_losses) / 5.0, loss.eval(), 3)

  def testNonZeroLossWithMeasurementSpecificWeightsWithPlaceholder(self):
    weights = np.array([3, 6, 5, 0, 4, 2]).reshape((2, 3))
    expected_losses = np.multiply(self._expected_losses, weights)

    tf_predictions = array_ops.placeholder(dtypes.float32, shape=[2, 3])
    loss = losses.log_loss(
        self._labels,
        tf_predictions,
        constant_op.constant(
            weights, shape=(2, 3)))

    with self.test_session() as sess:
      loss = sess.run(loss, feed_dict={tf_predictions: self._np_predictions})
      self.assertAlmostEqual(-np.sum(expected_losses) / 5.0, loss, 3)

  def testNonZeroLossWithSampleSpecificWeightsMostZero(self):
    weights = np.array([0, 0, 0, 0, 0, 2]).reshape((2, 3))
    expected_losses = np.multiply(self._expected_losses, weights)

    loss = losses.log_loss(
        self._labels,
        self._predictions,
        constant_op.constant(
            weights, shape=(2, 3)))
    with self.test_session():
      self.assertAlmostEqual(-np.sum(expected_losses), loss.eval(), 3)

  def testNonZeroLossWithSampleSpecificWeightsMostZeroWithPlaceholder(self):
    weights = np.array([0, 0, 0, 0, 0, 2]).reshape((2, 3))
    expected_losses = np.multiply(self._expected_losses, weights)

    tf_predictions = array_ops.placeholder(dtypes.float32, shape=[2, 3])
    tf_weights = constant_op.constant(weights, shape=(2, 3))
    loss = losses.log_loss(self._labels, tf_predictions, tf_weights)

    with self.test_session() as sess:
      loss = sess.run(loss, feed_dict={tf_predictions: self._np_predictions})
      self.assertAlmostEqual(-np.sum(expected_losses), loss, 3)

  def testLossWithSampleSpecificWeightsAllZero(self):
    tf_weights = array_ops.zeros(shape=(2, 3))
    loss = losses.log_loss(self._labels, self._predictions, tf_weights)
    with self.test_session():
      self.assertAlmostEqual(0.0, loss.eval(), 3)


class HingeLossTest(test.TestCase):

  def testIncompatibleShapes(self):
    with self.test_session():
      logits = constant_op.constant([[-1.0], [2.1]])
      labels = constant_op.constant([0.0, 1.0])
      with self.assertRaises(ValueError):
        _ = losses.hinge_loss(labels, logits).eval()

  def testAllOutsideMargin(self):
    with self.test_session():
      logits = constant_op.constant([1.2, -1.4, -1.0, 2.1])
      labels = constant_op.constant([1.0, 0.0, 0.0, 1.0])
      loss = losses.hinge_loss(labels, logits)
      self.assertAllClose(loss.eval(), 0.0, atol=1e-3)

  def testSomeInsideMargin(self):
    with self.test_session():
      logits = constant_op.constant([[-0.7], [-1.4], [1.4], [0.6]])
      labels = constant_op.constant([[0.0], [0.0], [1.0], [1.0]])
      loss = losses.hinge_loss(labels, logits)
      # Examples 1 and 4 are on the correct side of the hyperplane but within
      # the margin so they incur some (small) loss.
      self.assertAllClose(loss.eval(), 0.175, atol=1e-3)

  def testSomeMisclassified(self):
    with self.test_session():
      logits = constant_op.constant([[[1.2], [0.4], [-1.0], [-1.1]]])
      labels = constant_op.constant([[[1.0], [0.0], [0.0], [1.0]]])
      loss = losses.hinge_loss(labels, logits)
      # Examples 2 and 4 are on the wrong side of the hyperplane so they incur
      # some (fairly large) loss.
      self.assertAllClose(loss.eval(), 0.875, atol=1e-3)


class MeanSquaredErrorTest(test.TestCase):

  def setUp(self):
    self._predictions = constant_op.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
    self._labels = constant_op.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))

  def testValueErrorThrownWhenWeightIsNone(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        losses.mean_squared_error(
            self._predictions, self._predictions, weights=None)

  def testAllCorrectNoLossWeight(self):
    loss = losses.mean_squared_error(self._predictions, self._predictions)
    with self.test_session():
      self.assertAlmostEqual(0.0, loss.eval(), 3)

  def testNonZeroLoss(self):
    loss = losses.mean_squared_error(self._labels, self._predictions)
    with self.test_session():
      self.assertAlmostEqual(49.5, loss.eval(), 3)

  def testNonZeroLossWithPythonScalarWeight(self):
    weights = 2.3
    loss = losses.mean_squared_error(self._labels, self._predictions, weights)
    with self.test_session():
      self.assertAlmostEqual(49.5 * weights, loss.eval(), 3)

  def testNonZeroLossWithScalarTensorWeight(self):
    weights = 2.3
    loss = losses.mean_squared_error(self._labels, self._predictions,
                                     constant_op.constant(weights))
    with self.test_session():
      self.assertAlmostEqual(49.5 * weights, loss.eval(), 3)

  def testNonZeroLossWithOneDimBatchSpecificWeights(self):
    weights = constant_op.constant([1.2, 3.4], shape=[2,])
    loss = losses.mean_squared_error(self._labels, self._predictions, weights)
    with self.test_session():
      self.assertAlmostEqual(767.8 / 6.0, loss.eval(), 3)

  def testNonZeroLossWithTwoDimBatchSpecificWeights(self):
    weights = constant_op.constant([1.2, 3.4], shape=[2, 1])
    loss = losses.mean_squared_error(self._labels, self._predictions, weights)
    with self.test_session():
      self.assertAlmostEqual(767.8 / 6.0, loss.eval(), 3)

  def testNonZeroLossWithSampleSpecificWeights(self):
    weights = constant_op.constant([3, 6, 5, 0, 4, 2], shape=[2, 3])
    loss = losses.mean_squared_error(self._labels, self._predictions, weights)
    with self.test_session():
      self.assertAlmostEqual(587 / 5.0, loss.eval(), 3)

  def testNonZeroLossWithSampleSpecificWeightsMostZero(self):
    weights = constant_op.constant([0, 0, 0, 0, 0, 2], shape=[2, 3])
    loss = losses.mean_squared_error(self._labels, self._predictions, weights)
    with self.test_session():
      self.assertAlmostEqual(18.0, loss.eval(), 3)

  def testLossWithSampleSpecificWeightsAllZero(self):
    weights = array_ops.zeros((2, 3))
    loss = losses.mean_squared_error(self._labels, self._predictions, weights)
    with self.test_session():
      self.assertAlmostEqual(0.0, loss.eval(), 3)


class MeanPairwiseSquaresErrorTest(test.TestCase):

  def setUp(self):
    self._predictions = np.array([[4, 8, 12], [8, 1, 3]])
    self._labels = np.array([[1, 9, 2], [-5, -5, 7]])

    batch_size, dims = self._labels.shape

    # Compute the expected loss 'manually'.
    total = np.zeros((batch_size, 1))
    for b in range(batch_size):
      for i in range(dims):
        for j in range(dims):
          x = self._predictions[b, i].item() - self._predictions[b, j].item()
          y = self._labels[b, i].item() - self._labels[b, j].item()
          tmp = (x - y) * (x - y)
          total[b] += tmp

    self._expected_losses = np.divide(total, 9.0)

  def testValueErrorThrownWhenWeightIsNone(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        losses.mean_pairwise_squared_error(
            predictions=constant_op.constant(self._labels),
            labels=constant_op.constant(self._labels),
            weights=None)

  def testAllCorrectNoLossWeight(self):
    loss = losses.mean_pairwise_squared_error(
        predictions=constant_op.constant(self._labels),
        labels=constant_op.constant(self._labels))
    with self.test_session():
      self.assertAlmostEqual(0.0, loss.eval(), 3)

  def testNonZeroLoss(self):
    loss = losses.mean_pairwise_squared_error(
        predictions=constant_op.constant(self._predictions),
        labels=constant_op.constant(self._labels))
    with self.test_session():
      self.assertAlmostEqual(np.sum(self._expected_losses), loss.eval(), 3)

  def testGradientWithZeroWeight(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)

      inputs = array_ops.ones((2, 3))
      weights = variable_scope.get_variable(
          'weights',
          shape=[3, 4],
          initializer=init_ops.truncated_normal_initializer())
      predictions = math_ops.matmul(inputs, weights)

      optimizer = momentum_lib.MomentumOptimizer(
          learning_rate=0.001, momentum=0.9)
      loss = losses.mean_pairwise_squared_error(predictions, predictions, 0)

      gradients_to_variables = optimizer.compute_gradients(loss)

      init_op = variables.global_variables_initializer()

      with self.test_session() as sess:
        sess.run(init_op)
        for grad, _ in gradients_to_variables:
          np_grad = sess.run(grad)
          self.assertFalse(np.isnan(np_grad).any())

  def testNonZeroLossWithPythonScalarWeight(self):
    weights = 2.3
    loss = losses.mean_pairwise_squared_error(
        predictions=constant_op.constant(self._predictions),
        labels=constant_op.constant(self._labels),
        weights=weights)
    with self.test_session():
      self.assertAlmostEqual(weights * np.sum(self._expected_losses),
                             loss.eval(), 3)

  def testNonZeroLossWithScalarTensorWeight(self):
    weights = 2.3
    loss = losses.mean_pairwise_squared_error(
        predictions=constant_op.constant(self._predictions),
        labels=constant_op.constant(self._labels),
        weights=constant_op.constant(weights))
    with self.test_session():
      self.assertAlmostEqual(weights * np.sum(self._expected_losses),
                             loss.eval(), 3)

  def testNonZeroLossWithScalarZeroWeight(self):
    weights = 0
    loss = losses.mean_pairwise_squared_error(
        predictions=constant_op.constant(self._predictions),
        labels=constant_op.constant(self._labels),
        weights=constant_op.constant(weights))
    with self.test_session():
      self.assertAlmostEqual(0, loss.eval(), 3)

  def testNonZeroLossWithScalarTensorWeightWithPlaceholder(self):
    weights = 2.3
    tf_predictions = array_ops.placeholder(
        dtypes.float32, shape=self._predictions.shape)
    tf_labels = array_ops.placeholder(dtypes.float32, shape=self._labels.shape)
    loss = losses.mean_pairwise_squared_error(
        predictions=tf_predictions,
        labels=tf_labels,
        weights=constant_op.constant(weights))
    with self.test_session() as sess:
      loss = sess.run(loss,
                      feed_dict={
                          tf_predictions: self._predictions,
                          tf_labels: self._labels,
                      })
      self.assertAlmostEqual(weights * np.sum(self._expected_losses), loss, 3)

  def testNonZeroLossWithOneDimBatchSpecificWeights(self):
    weights = np.asarray([2.0, 1.0]).reshape((2, 1))
    expected_losses = np.multiply(weights, self._expected_losses)

    loss = losses.mean_pairwise_squared_error(
        predictions=constant_op.constant(self._predictions),
        labels=constant_op.constant(self._labels),
        weights=constant_op.constant(
            weights, shape=[2]))
    with self.test_session():
      self.assertAlmostEqual(np.sum(expected_losses), loss.eval(), 3)

  def testZeroLossWithOneDimBatchZeroWeights(self):
    weights = np.asarray([0.0, 0.0]).reshape((2, 1))
    loss = losses.mean_pairwise_squared_error(
        predictions=constant_op.constant(self._predictions),
        labels=constant_op.constant(self._labels),
        weights=constant_op.constant(
            weights, shape=[2]))
    with self.test_session():
      self.assertAlmostEqual(0, loss.eval(), 3)

  def testNonZeroLossWithOneDimBatchSpecificWeightsAndPlaceholders(self):
    weights = np.asarray([1.2, 3.4]).reshape((2, 1))
    expected_losses = np.multiply(weights, self._expected_losses)

    tf_predictions = array_ops.placeholder(
        dtypes.float32, shape=self._predictions.shape)
    tf_labels = array_ops.placeholder(dtypes.int32, shape=self._labels.shape)
    loss = losses.mean_pairwise_squared_error(
        predictions=tf_predictions,
        labels=tf_labels,
        weights=constant_op.constant(
            weights, shape=[2]))

    with self.test_session() as sess:
      loss = sess.run(loss,
                      feed_dict={
                          tf_predictions: self._predictions,
                          tf_labels: self._labels,
                      })
      self.assertAlmostEqual(np.sum(expected_losses), loss, 3)

  def testLossWithAllZeroBatchSpecificWeights(self):
    weights = np.zeros((2, 1))
    loss = losses.mean_pairwise_squared_error(
        predictions=constant_op.constant(self._predictions),
        labels=constant_op.constant(self._labels),
        weights=constant_op.constant(
            weights, shape=[2]))
    with self.test_session():
      self.assertAlmostEqual(0.0, loss.eval(), 3)


class CosineDistanceLossTest(test.TestCase):

  def setUp(self):
    self._predictions = np.asarray([
        [1, 0, 0],  # Batch 1
        [0, 0, -1],
        [1, 0, 0],  # Batch 2
        [1, 0, 0],
        [0, 0, -1],  # Batch 3
        [1, 0, 0]
    ]).reshape((3, 2, 3))

    self._labels = np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0],
                               [0, 0, 1], [0, 1, 0]]).reshape((3, 2, 3))

  def testValueErrorThrownWhenWeightIsNone(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        losses.cosine_distance(
            predictions=constant_op.constant(self._labels),
            labels=constant_op.constant(self._labels),
            dim=2,
            weights=None)

  def testAllCorrectNoWeights(self):
    loss = losses.cosine_distance(
        predictions=constant_op.constant(self._labels),
        labels=constant_op.constant(self._labels),
        dim=2)
    with self.test_session():
      self.assertAlmostEqual(0, loss.eval(), 5)

  def testPartiallyCorrectWithIntegerValues(self):
    loss = losses.cosine_distance(
        predictions=constant_op.constant(self._predictions),
        labels=constant_op.constant(self._labels),
        dim=2)
    with self.test_session():
      self.assertAlmostEqual(1, loss.eval(), 5)

  def testPartiallyCorrectFloatingPointValues(self):
    predictions = np.matrix(
        ('0.819031913261206 0.567041924552012 0.087465312324590;'
         '-0.665139432070255 -0.739487441769973 -0.103671883216994;'
         '0.707106781186548 -0.707106781186548 0'))
    labels = np.matrix(('0.819031913261206 0.567041924552012 0.087465312324590;'
                        '0.665139432070255 0.739487441769973 0.103671883216994;'
                        '0.707106781186548 0.707106781186548 0'))

    tf_preds = constant_op.constant(
        predictions, shape=(3, 1, 3), dtype=dtypes.float32)
    tf_labels = constant_op.constant(
        labels, shape=(3, 1, 3), dtype=dtypes.float32)
    loss = losses.cosine_distance(tf_labels, tf_preds, dim=2)

    with self.test_session():
      self.assertAlmostEqual(1.0, loss.eval(), 5)

  def testSampleSpecificWeights(self):
    loss = losses.cosine_distance(
        predictions=constant_op.constant(self._predictions),
        labels=constant_op.constant(self._labels),
        dim=2,
        weights=constant_op.constant([1, 0, 0]))
    with self.test_session():
      self.assertEqual(1.0, loss.eval())

  def testMeasurementSpecificWeights(self):
    loss = losses.cosine_distance(
        predictions=constant_op.constant(self._predictions),
        labels=constant_op.constant(self._labels),
        dim=2,
        weights=constant_op.constant(
            [1, 0, 0, 1, 1, 1], shape=(3, 2)))
    with self.test_session():
      self.assertEqual(3.0 / 4.0, loss.eval())

  def testValueErrorThrownWithShapelessPlaceholder(self):
    tf_predictions = array_ops.placeholder(dtypes.float32)
    with self.test_session():
      with self.assertRaises(ValueError):
        losses.cosine_distance(
            predictions=tf_predictions,
            labels=constant_op.constant(self._labels),
            dim=2,
            weights=constant_op.constant(
                [1, 0, 0, 1, 1, 1], shape=(3, 2)))

  def testMeasurementSpecificWeightsWithPlaceholderWithShape(self):
    tf_predictions = array_ops.placeholder(
        dtypes.float32, shape=self._labels.shape)
    loss = losses.cosine_distance(
        predictions=tf_predictions,
        labels=constant_op.constant(self._labels),
        dim=2,
        weights=constant_op.constant(
            [1, 0, 0, 1, 1, 1], shape=(3, 2)))
    with self.test_session() as sess:
      loss = sess.run(loss, feed_dict={tf_predictions: self._predictions})
      self.assertEqual(3.0 / 4.0, loss)

  def testZeroLossWhenAllSampleSpecificWeightsAreZero(self):
    loss = losses.cosine_distance(
        predictions=constant_op.constant(self._predictions),
        labels=constant_op.constant(self._labels),
        dim=2,
        weights=array_ops.zeros((3,)))
    with self.test_session():
      self.assertEqual(0, loss.eval())

  def testZeroLossWhenAllMeasurementSpecificWeightsAreZero(self):
    loss = losses.cosine_distance(
        predictions=constant_op.constant(self._predictions),
        labels=constant_op.constant(self._labels),
        dim=2,
        weights=array_ops.zeros((3, 2)))
    with self.test_session():
      self.assertEqual(0, loss.eval())


class AddLossTest(test.TestCase):

  def testNoCollectLossesBatch2(self):
    logits = constant_op.constant([[1.2, 0.4, -1.0, -1.1]] * 2)
    labels = constant_op.constant([[1.0, 0.0, 0.0, 1.0]] * 2)
    self.assertFalse(util.get_losses())
    losses.absolute_difference(logits, labels, loss_collection=None)
    losses.log_loss(logits, labels, loss_collection=None)
    losses.mean_squared_error(logits, labels, loss_collection=None)
    losses.sigmoid_cross_entropy(logits, labels, loss_collection=None)
    losses.softmax_cross_entropy(logits, labels, loss_collection=None)
    self.assertFalse(util.get_losses())


class ComputeWeightedLossTest(test.TestCase):

  def setUp(self):
    self._shape = (3, 2, 4)
    raw_losses = np.zeros(self._shape)
    next_loss = 0.0
    for i in range(self._shape[0]):
      for j in range(self._shape[1]):
        for k in range(self._shape[2]):
          raw_losses[i][j][k] = next_loss
          next_loss += 1.0
    raw_losses.setflags(write=False)
    self._raw_losses = raw_losses
    self._unweighted_loss = np.mean(self._raw_losses)

  def testUnweighted(self):
    with ops.Graph().as_default():
      self.assertEqual(0, len(loss_ops.get_losses()))
      raw_losses = self._raw_losses
      shape = self._shape
      unweighted_losses = (losses.compute_weighted_loss(raw_losses),
                           losses.compute_weighted_loss(
                               raw_losses, weights=1.0),
                           losses.compute_weighted_loss(
                               raw_losses, weights=np.ones(shape=shape[0:1])),
                           losses.compute_weighted_loss(
                               raw_losses, weights=np.ones(shape=shape[0:2])),
                           losses.compute_weighted_loss(
                               raw_losses, weights=np.ones(shape=shape)))
      self.assertEqual(5, len(loss_ops.get_losses()))
      with self.test_session():
        for unweighted_loss in unweighted_losses:
          self.assertAllClose(self._unweighted_loss, unweighted_loss.eval())

  def testScalarWeight(self):
    with ops.Graph().as_default():
      self.assertEqual(0, len(loss_ops.get_losses()))
      weight = 17.0
      weighted_loss = losses.compute_weighted_loss(
          self._raw_losses, weights=weight)
      self.assertEqual(1, len(loss_ops.get_losses()))
      with self.test_session():
        self.assertAllClose(
            np.mean(weight * self._raw_losses), weighted_loss.eval())

  # TODO(b/33556118): Bug: `loss1` should be the same as `testUnweighted`, and
  # `loss17` should be the same as `testScalarWeight`.
  def testScalar1DWeight(self):
    with ops.Graph().as_default():
      self.assertEqual(0, len(loss_ops.get_losses()))
      loss1 = losses.compute_weighted_loss(self._raw_losses, weights=(1.0,))
      self.assertEqual(1, len(loss_ops.get_losses()))
      weight = 17.0
      loss17 = losses.compute_weighted_loss(self._raw_losses, weights=(weight,))
      self.assertEqual(2, len(loss_ops.get_losses()))
      with self.test_session():
        self.assertAllClose(self._unweighted_loss * self._shape[0],
                            loss1.eval())
        self.assertAllClose(
            np.mean(weight * self._raw_losses) * self._shape[0], loss17.eval())

  def testInvalid1DWeight(self):
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(ValueError, 'Dimensions must be equal'):
        losses.compute_weighted_loss(self._raw_losses, weights=(17.0, 31.0))

  def testInvalid4DWeight(self):
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(ValueError, 'Invalid weights shape'):
        losses.compute_weighted_loss(
            self._raw_losses, weights=np.zeros(shape=(2, 2, 2, 2)))

  def test3Weight(self):
    with ops.Graph().as_default():
      self.assertEqual(0, len(loss_ops.get_losses()))
      weights3 = (17.0, 5.0, 2.0)
      weighted_loss = losses.compute_weighted_loss(
          self._raw_losses, weights=weights3)
      self.assertEqual(1, len(loss_ops.get_losses()))
      with self.test_session():
        weights3x1x1 = np.reshape(weights3, (3, 1, 1))
        self.assertAllClose(
            np.mean(weights3x1x1 * self._raw_losses), weighted_loss.eval())

  def test3x1Weight(self):
    with ops.Graph().as_default():
      self.assertEqual(0, len(loss_ops.get_losses()))
      weights3x1 = (
          (17.0,),
          (5.0,),
          (2.0,),)
      weighted_loss = losses.compute_weighted_loss(
          self._raw_losses, weights=weights3x1)
      self.assertEqual(1, len(loss_ops.get_losses()))
      with self.test_session():
        weights3x1x1 = np.reshape(weights3x1, (3, 1, 1))
        self.assertAllClose(
            np.mean(weights3x1x1 * self._raw_losses), weighted_loss.eval())

  # TODO(ptucker): Bug: this should be the same as `test3x1Weight`.
  def test3x1x1Weight(self):
    with ops.Graph().as_default():
      self.assertEqual(0, len(loss_ops.get_losses()))
      weights3x1x1 = (
          ((17.0,),),
          ((5.0,),),
          ((2.0,),),)
      weighted_loss = losses.compute_weighted_loss(
          self._raw_losses, weights=weights3x1x1)
      self.assertEqual(1, len(loss_ops.get_losses()))
      with self.test_session():
        self.assertAllClose(
            np.mean(weights3x1x1 * self._raw_losses) * self._shape[1],
            weighted_loss.eval())

  def test3x2Weight(self):
    with ops.Graph().as_default():
      self.assertEqual(0, len(loss_ops.get_losses()))
      weights3x2 = (
          (17.0, 3.0),
          (5.0, 31.0),
          (2.0, 7.0),)
      weighted_loss = losses.compute_weighted_loss(
          self._raw_losses, weights=weights3x2)
      self.assertEqual(1, len(loss_ops.get_losses()))
      with self.test_session():
        weights3x2x1 = np.reshape(weights3x2, (3, 2, 1))
        self.assertAllClose(
            np.mean(weights3x2x1 * self._raw_losses), weighted_loss.eval())

  # TODO(b/33556118): Bug: this should be averaged across all dimensions, not
  # summed across dim 0.
  def test1x2Weight(self):
    with ops.Graph().as_default():
      self.assertEqual(0, len(loss_ops.get_losses()))
      weights1x2 = ((
          17.0,
          3.0,),)
      weighted_loss = losses.compute_weighted_loss(
          self._raw_losses, weights=weights1x2)
      self.assertEqual(1, len(loss_ops.get_losses()))
      with self.test_session():
        weights1x2x1 = np.reshape(weights1x2, (1, 2, 1))
        self.assertAllClose(
            np.mean(weights1x2x1 * self._raw_losses) * self._shape[0],
            weighted_loss.eval())

  # TODO(b/33556118): Bug: this should be averaged across all dimensions, not
  # summed across dim 0.
  def test1x2x1Weight(self):
    with ops.Graph().as_default():
      self.assertEqual(0, len(loss_ops.get_losses()))
      weights1x2x1 = ((
          (17.0,),
          (3.0,),),)
      weighted_loss = losses.compute_weighted_loss(
          self._raw_losses, weights=weights1x2x1)
      self.assertEqual(1, len(loss_ops.get_losses()))
      with self.test_session():
        self.assertAllClose(
            np.mean(weights1x2x1 * self._raw_losses) * self._shape[0],
            weighted_loss.eval())

  # TODO(b/33556118): Bug: this should be averaged across all dimensions, not
  # summed across dims 0 & 1.
  def test1x1x4Weight(self):
    with ops.Graph().as_default():
      self.assertEqual(0, len(loss_ops.get_losses()))
      weights1x1x4 = (((17.0, 13.0, 2.0, 5.0),),)
      weighted_loss = losses.compute_weighted_loss(
          self._raw_losses, weights=weights1x1x4)
      self.assertEqual(1, len(loss_ops.get_losses()))
      shape = self._shape
      with self.test_session():
        self.assertAllClose(
            np.mean(weights1x1x4 * self._raw_losses) * shape[0] * shape[1],
            weighted_loss.eval())

  def test3x2x1Weight(self):
    with ops.Graph().as_default():
      self.assertEqual(0, len(loss_ops.get_losses()))
      weights3x2x1 = (
          ((17.0,), (3.0,)),
          ((5.0,), (31.0,)),
          ((2.0,), (7.0,)),
      )
      weighted_loss = loss_ops.compute_weighted_loss(
          self._raw_losses, weights=weights3x2x1)
      self.assertEqual(1, len(loss_ops.get_losses()))
      with self.test_session():
        self.assertAllClose(
            np.mean(weights3x2x1 * self._raw_losses),
            weighted_loss.eval())

  # TODO(b/33556118): Bug: this should be averaged across all dimensions, not
  # summed across dim 1.
  def test3x1x4Weight(self):
    with ops.Graph().as_default():
      self.assertEqual(0, len(loss_ops.get_losses()))
      weights3x1x4 = (
          ((17.0, 13.0, 2.0, 5.0),),
          ((5.0, 31.0, 17.0, 5.0),),
          ((7.0, 3.0, 11.0, 5.0),),
      )
      weighted_loss = loss_ops.compute_weighted_loss(
          self._raw_losses, weights=weights3x1x4)
      self.assertEqual(1, len(loss_ops.get_losses()))
      with self.test_session():
        self.assertAllClose(
            np.mean(weights3x1x4 * self._raw_losses) * self._shape[1],
            weighted_loss.eval())

  # TODO(b/33556118): Bug: this should be averaged across all dimensions, not
  # summed across dim 0.
  def test1x2x4Weight(self):
    with ops.Graph().as_default():
      self.assertEqual(0, len(loss_ops.get_losses()))
      weights1x2x4 = ((
          (17.0, 13.0, 2.0, 5.0),
          (3.0, 13.0, 11.0, 2.0),),)
      weighted_loss = losses.compute_weighted_loss(
          self._raw_losses, weights=weights1x2x4)
      self.assertEqual(1, len(loss_ops.get_losses()))
      with self.test_session():
        self.assertAllClose(
            np.mean(weights1x2x4 * self._raw_losses) * self._shape[0],
            weighted_loss.eval())

  def test3x2x4Weight(self):
    with ops.Graph().as_default():
      self.assertEqual(0, len(loss_ops.get_losses()))
      weights3x2x4 = (
          (
              (17.0, 13.0, 2.0, 5.0),
              (3.0, 13.0, 11.0, 2.0),),
          (
              (5.0, 31.0, 17.0, 5.0),
              (13.0, 3.0, 1.0, 11.0),),
          (
              (7.0, 3.0, 11.0, 5.0),
              (13.0, 11.0, 1.0, 7.0),),)
      weighted_loss = losses.compute_weighted_loss(
          self._raw_losses, weights=weights3x2x4)
      self.assertEqual(1, len(loss_ops.get_losses()))
      with self.test_session():
        self.assertAllClose(
            np.mean(weights3x2x4 * self._raw_losses), weighted_loss.eval())


if __name__ == '__main__':
  test.main()
