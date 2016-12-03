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
"""Tests for metric_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.metrics.python.ops import metric_ops

NAN = float('nan')

metrics = tf.contrib.metrics


def _enqueue_vector(sess, queue, values, shape=None):
  if not shape:
    shape = (1, len(values))
  dtype = queue.dtypes[0]
  sess.run(queue.enqueue(tf.constant(values, dtype=dtype, shape=shape)))


def _binary_2d_label_to_sparse_value(labels):
  """Convert dense 2D binary indicator tensor to sparse tensor.

  Only 1 values in `labels` are included in result.

  Args:
    labels: Dense 2D binary indicator tensor.

  Returns:
    `SparseTensorValue` whose values are indices along the last dimension of
    `labels`.
  """
  indices = []
  values = []
  batch = 0
  for row in labels:
    label = 0
    xi = 0
    for x in row:
      if x == 1:
        indices.append([batch, xi])
        values.append(label)
        xi += 1
      else:
        assert x == 0
      label += 1
    batch += 1
  shape = [len(labels), len(labels[0])]
  return tf.SparseTensorValue(
      np.array(indices, np.int64),
      np.array(values, np.int64),
      np.array(shape, np.int64))


def _binary_2d_label_to_sparse(labels):
  """Convert dense 2D binary indicator tensor to sparse tensor.

  Only 1 values in `labels` are included in result.

  Args:
    labels: Dense 2D binary indicator tensor.

  Returns:
    `SparseTensor` whose values are indices along the last dimension of
    `labels`.
  """
  return tf.SparseTensor.from_value(_binary_2d_label_to_sparse_value(labels))


def _binary_3d_label_to_sparse_value(labels):
  """Convert dense 3D binary indicator tensor to sparse tensor.

  Only 1 values in `labels` are included in result.

  Args:
    labels: Dense 2D binary indicator tensor.

  Returns:
    `SparseTensorValue` whose values are indices along the last dimension of
    `labels`.
  """
  indices = []
  values = []
  for d0, labels_d0 in enumerate(labels):
    for d1, labels_d1 in enumerate(labels_d0):
      d2 = 0
      for class_id, label in enumerate(labels_d1):
        if label == 1:
          values.append(class_id)
          indices.append([d0, d1, d2])
          d2 += 1
        else:
          assert label == 0
  shape = [len(labels), len(labels[0]), len(labels[0][0])]
  return tf.SparseTensorValue(
      np.array(indices, np.int64),
      np.array(values, np.int64),
      np.array(shape, np.int64))


def _binary_3d_label_to_sparse(labels):
  """Convert dense 3D binary indicator tensor to sparse tensor.

  Only 1 values in `labels` are included in result.

  Args:
    labels: Dense 2D binary indicator tensor.

  Returns:
    `SparseTensor` whose values are indices along the last dimension of
    `labels`.
  """
  return tf.SparseTensor.from_value(_binary_3d_label_to_sparse_value(labels))


def _assert_nan(test_case, actual):
  test_case.assertTrue(math.isnan(actual), 'Expected NAN, got %s.' % actual)


def _assert_local_variables(test_case, expected):
  test_case.assertEquals(
      set(expected), set(v.name for v in tf.local_variables()))


class StreamingMeanTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def testVars(self):
    metrics.streaming_mean(tf.ones([4, 3]))
    _assert_local_variables(self, ('mean/count:0', 'mean/total:0'))

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = metrics.streaming_mean(
        tf.ones([4, 3]),
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [mean])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.streaming_mean(
        tf.ones([4, 3]),
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testBasic(self):
    with self.test_session() as sess:
      values_queue = tf.FIFOQueue(4, dtypes=tf.float32, shapes=(1, 2))
      _enqueue_vector(sess, values_queue, [0, 1])
      _enqueue_vector(sess, values_queue, [-4.2, 9.1])
      _enqueue_vector(sess, values_queue, [6.5, 0])
      _enqueue_vector(sess, values_queue, [-3.2, 4.0])
      values = values_queue.dequeue()

      mean, update_op = metrics.streaming_mean(values)

      sess.run(tf.local_variables_initializer())
      for _ in range(4):
        sess.run(update_op)
      self.assertAlmostEqual(1.65, sess.run(mean), 5)

  def testUpdateOpsReturnsCurrentValue(self):
    with self.test_session() as sess:
      values_queue = tf.FIFOQueue(4, dtypes=tf.float32, shapes=(1, 2))
      _enqueue_vector(sess, values_queue, [0, 1])
      _enqueue_vector(sess, values_queue, [-4.2, 9.1])
      _enqueue_vector(sess, values_queue, [6.5, 0])
      _enqueue_vector(sess, values_queue, [-3.2, 4.0])
      values = values_queue.dequeue()

      mean, update_op = metrics.streaming_mean(values)

      sess.run(tf.local_variables_initializer())

      self.assertAlmostEqual(0.5, sess.run(update_op), 5)
      self.assertAlmostEqual(1.475, sess.run(update_op), 5)
      self.assertAlmostEqual(12.4/6.0, sess.run(update_op), 5)
      self.assertAlmostEqual(1.65, sess.run(update_op), 5)

      self.assertAlmostEqual(1.65, sess.run(mean), 5)

  def test1dWeightedValues(self):
    with self.test_session() as sess:
      # Create the queue that populates the values.
      values_queue = tf.FIFOQueue(4, dtypes=tf.float32, shapes=(1, 2))
      _enqueue_vector(sess, values_queue, [0, 1])
      _enqueue_vector(sess, values_queue, [-4.2, 9.1])
      _enqueue_vector(sess, values_queue, [6.5, 0])
      _enqueue_vector(sess, values_queue, [-3.2, 4.0])
      values = values_queue.dequeue()

      # Create the queue that populates the weighted labels.
      weights_queue = tf.FIFOQueue(4, dtypes=tf.float32, shapes=(1, 1))
      _enqueue_vector(sess, weights_queue, [1])
      _enqueue_vector(sess, weights_queue, [0])
      _enqueue_vector(sess, weights_queue, [0])
      _enqueue_vector(sess, weights_queue, [1])
      weights = weights_queue.dequeue()

      mean, update_op = metrics.streaming_mean(values, weights)

      tf.local_variables_initializer().run()
      for _ in range(4):
        update_op.eval()
      self.assertAlmostEqual((0 + 1 - 3.2 + 4.0) / 4.0, mean.eval(), 5)

  def test1dWeightedValues_placeholders(self):
    with self.test_session() as sess:
      # Create the queue that populates the values.
      feed_values = (
          (0, 1),
          (-4.2, 9.1),
          (6.5, 0),
          (-3.2, 4.0)
      )
      values = tf.placeholder(dtype=tf.float32)

      # Create the queue that populates the weighted labels.
      weights_queue = tf.FIFOQueue(4, dtypes=tf.float32, shapes=(1, 1))
      _enqueue_vector(sess, weights_queue, [1])
      _enqueue_vector(sess, weights_queue, [0])
      _enqueue_vector(sess, weights_queue, [0])
      _enqueue_vector(sess, weights_queue, [1])
      weights = weights_queue.dequeue()

      mean, update_op = metrics.streaming_mean(values, weights)

      tf.local_variables_initializer().run()
      for i in range(4):
        update_op.eval(feed_dict={values: feed_values[i]})
      self.assertAlmostEqual((0 + 1 - 3.2 + 4.0) / 4.0, mean.eval(), 5)

  def test2dWeightedValues(self):
    with self.test_session() as sess:
      # Create the queue that populates the values.
      values_queue = tf.FIFOQueue(4, dtypes=tf.float32, shapes=(1, 2))
      _enqueue_vector(sess, values_queue, [0, 1])
      _enqueue_vector(sess, values_queue, [-4.2, 9.1])
      _enqueue_vector(sess, values_queue, [6.5, 0])
      _enqueue_vector(sess, values_queue, [-3.2, 4.0])
      values = values_queue.dequeue()

      # Create the queue that populates the weighted labels.
      weights_queue = tf.FIFOQueue(4, dtypes=tf.float32, shapes=(1, 2))
      _enqueue_vector(sess, weights_queue, [1, 1])
      _enqueue_vector(sess, weights_queue, [1, 0])
      _enqueue_vector(sess, weights_queue, [0, 1])
      _enqueue_vector(sess, weights_queue, [0, 0])
      weights = weights_queue.dequeue()

      mean, update_op = metrics.streaming_mean(values, weights)

      tf.local_variables_initializer().run()
      for _ in range(4):
        update_op.eval()
      self.assertAlmostEqual((0 + 1 - 4.2 + 0) / 4.0, mean.eval(), 5)

  def test2dWeightedValues_placeholders(self):
    with self.test_session() as sess:
      # Create the queue that populates the values.
      feed_values = (
          (0, 1),
          (-4.2, 9.1),
          (6.5, 0),
          (-3.2, 4.0)
      )
      values = tf.placeholder(dtype=tf.float32)

      # Create the queue that populates the weighted labels.
      weights_queue = tf.FIFOQueue(4, dtypes=tf.float32, shapes=(1, 2))
      _enqueue_vector(sess, weights_queue, [1, 1])
      _enqueue_vector(sess, weights_queue, [1, 0])
      _enqueue_vector(sess, weights_queue, [0, 1])
      _enqueue_vector(sess, weights_queue, [0, 0])
      weights = weights_queue.dequeue()

      mean, update_op = metrics.streaming_mean(values, weights)

      tf.local_variables_initializer().run()
      for i in range(4):
        update_op.eval(feed_dict={values: feed_values[i]})
      self.assertAlmostEqual((0 + 1 - 4.2 + 0) / 4.0, mean.eval(), 5)


class StreamingMeanTensorTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def testVars(self):
    metrics.streaming_mean_tensor(tf.ones([4, 3]))
    _assert_local_variables(self, (
        'mean/total_tensor:0', 'mean/count_tensor:0'))

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = metrics.streaming_mean_tensor(
        tf.ones([4, 3]),
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [mean])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.streaming_mean_tensor(
        tf.ones([4, 3]),
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testBasic(self):
    with self.test_session() as sess:
      values_queue = tf.FIFOQueue(4, dtypes=tf.float32, shapes=(1, 2))
      _enqueue_vector(sess, values_queue, [0, 1])
      _enqueue_vector(sess, values_queue, [-4.2, 9.1])
      _enqueue_vector(sess, values_queue, [6.5, 0])
      _enqueue_vector(sess, values_queue, [-3.2, 4.0])
      values = values_queue.dequeue()

      mean, update_op = metrics.streaming_mean_tensor(values)

      sess.run(tf.local_variables_initializer())
      for _ in range(4):
        sess.run(update_op)
      self.assertAllClose([[-0.9/4., 3.525]], sess.run(mean))

  def testMultiDimensional(self):
    with self.test_session() as sess:
      values_queue = tf.FIFOQueue(2, dtypes=tf.float32, shapes=(2, 2, 2))
      _enqueue_vector(sess,
                      values_queue,
                      [[[1, 2], [1, 2]], [[1, 2], [1, 2]]],
                      shape=(2, 2, 2))
      _enqueue_vector(sess,
                      values_queue,
                      [[[1, 2], [1, 2]], [[3, 4], [9, 10]]],
                      shape=(2, 2, 2))
      values = values_queue.dequeue()

      mean, update_op = metrics.streaming_mean_tensor(values)

      sess.run(tf.local_variables_initializer())
      for _ in range(2):
        sess.run(update_op)
      self.assertAllClose([[[1, 2], [1, 2]], [[2, 3], [5, 6]]],
                          sess.run(mean))

  def testUpdateOpsReturnsCurrentValue(self):
    with self.test_session() as sess:
      values_queue = tf.FIFOQueue(4, dtypes=tf.float32, shapes=(1, 2))
      _enqueue_vector(sess, values_queue, [0, 1])
      _enqueue_vector(sess, values_queue, [-4.2, 9.1])
      _enqueue_vector(sess, values_queue, [6.5, 0])
      _enqueue_vector(sess, values_queue, [-3.2, 4.0])
      values = values_queue.dequeue()

      mean, update_op = metrics.streaming_mean_tensor(values)

      sess.run(tf.local_variables_initializer())

      self.assertAllClose([[0, 1]], sess.run(update_op), 5)
      self.assertAllClose([[-2.1, 5.05]], sess.run(update_op), 5)
      self.assertAllClose([[2.3/3., 10.1/3.]], sess.run(update_op), 5)
      self.assertAllClose([[-0.9/4., 3.525]], sess.run(update_op), 5)

      self.assertAllClose([[-0.9/4., 3.525]], sess.run(mean), 5)

  def testWeighted1d(self):
    with self.test_session() as sess:
      # Create the queue that populates the values.
      values_queue = tf.FIFOQueue(4, dtypes=tf.float32, shapes=(1, 2))
      _enqueue_vector(sess, values_queue, [0, 1])
      _enqueue_vector(sess, values_queue, [-4.2, 9.1])
      _enqueue_vector(sess, values_queue, [6.5, 0])
      _enqueue_vector(sess, values_queue, [-3.2, 4.0])
      values = values_queue.dequeue()

      # Create the queue that populates the weights.
      weights_queue = tf.FIFOQueue(4, dtypes=tf.float32, shapes=(1, 1))
      _enqueue_vector(sess, weights_queue, [[1]])
      _enqueue_vector(sess, weights_queue, [[0]])
      _enqueue_vector(sess, weights_queue, [[1]])
      _enqueue_vector(sess, weights_queue, [[0]])
      weights = weights_queue.dequeue()

      mean, update_op = metrics.streaming_mean_tensor(values, weights)

      sess.run(tf.local_variables_initializer())
      for _ in range(4):
        sess.run(update_op)
      self.assertAllClose([[3.25, 0.5]], sess.run(mean), 5)

  def testWeighted2d_1(self):
    with self.test_session() as sess:
      # Create the queue that populates the values.
      values_queue = tf.FIFOQueue(4, dtypes=tf.float32, shapes=(1, 2))
      _enqueue_vector(sess, values_queue, [0, 1])
      _enqueue_vector(sess, values_queue, [-4.2, 9.1])
      _enqueue_vector(sess, values_queue, [6.5, 0])
      _enqueue_vector(sess, values_queue, [-3.2, 4.0])
      values = values_queue.dequeue()

      # Create the queue that populates the weights.
      weights_queue = tf.FIFOQueue(4, dtypes=tf.float32, shapes=(1, 2))
      _enqueue_vector(sess, weights_queue, [1, 1])
      _enqueue_vector(sess, weights_queue, [1, 0])
      _enqueue_vector(sess, weights_queue, [0, 1])
      _enqueue_vector(sess, weights_queue, [0, 0])
      weights = weights_queue.dequeue()

      mean, update_op = metrics.streaming_mean_tensor(values, weights)

      sess.run(tf.local_variables_initializer())
      for _ in range(4):
        sess.run(update_op)
      self.assertAllClose([[-2.1, 0.5]], sess.run(mean), 5)

  def testWeighted2d_2(self):
    with self.test_session() as sess:
      # Create the queue that populates the values.
      values_queue = tf.FIFOQueue(4, dtypes=tf.float32, shapes=(1, 2))
      _enqueue_vector(sess, values_queue, [0, 1])
      _enqueue_vector(sess, values_queue, [-4.2, 9.1])
      _enqueue_vector(sess, values_queue, [6.5, 0])
      _enqueue_vector(sess, values_queue, [-3.2, 4.0])
      values = values_queue.dequeue()

      # Create the queue that populates the weights.
      weights_queue = tf.FIFOQueue(4, dtypes=tf.float32, shapes=(1, 2))
      _enqueue_vector(sess, weights_queue, [0, 1])
      _enqueue_vector(sess, weights_queue, [0, 0])
      _enqueue_vector(sess, weights_queue, [0, 1])
      _enqueue_vector(sess, weights_queue, [0, 0])
      weights = weights_queue.dequeue()

      mean, update_op = metrics.streaming_mean_tensor(values, weights)

      sess.run(tf.local_variables_initializer())
      for _ in range(4):
        sess.run(update_op)
      self.assertAllClose([[0, 0.5]], sess.run(mean), 5)


class StreamingAccuracyTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def testVars(self):
    metrics.streaming_accuracy(
        predictions=tf.ones((10, 1)), labels=tf.ones((10, 1)),
        name='my_accuracy')
    _assert_local_variables(self, (
        'my_accuracy/count:0', 'my_accuracy/total:0'))

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = metrics.streaming_accuracy(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [mean])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.streaming_accuracy(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testPredictionsAndLabelsOfDifferentSizeRaisesValueError(self):
    predictions = tf.ones((10, 3))
    labels = tf.ones((10, 4))
    with self.assertRaises(ValueError):
      metrics.streaming_accuracy(predictions, labels)

  def testPredictionsAndWeightsOfDifferentSizeRaisesValueError(self):
    predictions = tf.ones((10, 3))
    labels = tf.ones((10, 3))
    weights = tf.ones((9, 3))
    with self.assertRaises(ValueError):
      metrics.streaming_accuracy(predictions, labels, weights)

  def testValueTensorIsIdempotent(self):
    predictions = tf.random_uniform((10, 3), maxval=3, dtype=tf.int64, seed=1)
    labels = tf.random_uniform((10, 3), maxval=3, dtype=tf.int64, seed=1)
    accuracy, update_op = metrics.streaming_accuracy(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        sess.run(update_op)

      # Then verify idempotency.
      initial_accuracy = accuracy.eval()
      for _ in range(10):
        self.assertEqual(initial_accuracy, accuracy.eval())

  def testMultipleUpdates(self):
    with self.test_session() as sess:
      # Create the queue that populates the predictions.
      preds_queue = tf.FIFOQueue(4, dtypes=tf.float32, shapes=(1, 1))
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [2])
      _enqueue_vector(sess, preds_queue, [1])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      labels_queue = tf.FIFOQueue(4, dtypes=tf.float32, shapes=(1, 1))
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [2])
      labels = labels_queue.dequeue()

      accuracy, update_op = metrics.streaming_accuracy(
          predictions, labels)

      sess.run(tf.local_variables_initializer())
      for _ in xrange(3):
        sess.run(update_op)
      self.assertEqual(0.5, sess.run(update_op))
      self.assertEqual(0.5, accuracy.eval())

  def testEffectivelyEquivalentSizes(self):
    predictions = tf.ones((40, 1))
    labels = tf.ones((40,))
    with self.test_session() as sess:
      accuracy, update_op = metrics.streaming_accuracy(
          predictions, labels)

      sess.run(tf.local_variables_initializer())
      self.assertEqual(1.0, update_op.eval())
      self.assertEqual(1.0, accuracy.eval())

  def testEffectivelyEquivalentSizesWithStaicShapedWeight(self):
    predictions = tf.convert_to_tensor([1, 1, 1])  # shape 3,
    labels = tf.expand_dims(tf.convert_to_tensor([1, 0, 0]), 1)  # shape 3, 1
    weights = tf.expand_dims(tf.convert_to_tensor([100, 1, 1]), 1)  # shape 3, 1

    with self.test_session() as sess:
      accuracy, update_op = metrics.streaming_accuracy(
          predictions, labels, weights)

      sess.run(tf.local_variables_initializer())
      # if streaming_accuracy does not flatten the weight, accuracy would be
      # 0.33333334 due to an intended broadcast of weight. Due to flattening,
      # it will be higher than .95
      self.assertGreater(update_op.eval(), .95)
      self.assertGreater(accuracy.eval(), .95)

  def testEffectivelyEquivalentSizesWithDynamicallyShapedWeight(self):
    predictions = tf.convert_to_tensor([1, 1, 1])  # shape 3,
    labels = tf.expand_dims(tf.convert_to_tensor([1, 0, 0]), 1)  # shape 3, 1

    weights = [[100], [1], [1]]  # shape 3, 1
    weights_placeholder = tf.placeholder(dtype=tf.int32, name='weights')
    feed_dict = {weights_placeholder: weights}

    with self.test_session() as sess:
      accuracy, update_op = metrics.streaming_accuracy(
          predictions, labels, weights_placeholder)

      sess.run(tf.local_variables_initializer())
      # if streaming_accuracy does not flatten the weight, accuracy would be
      # 0.33333334 due to an intended broadcast of weight. Due to flattening,
      # it will be higher than .95
      self.assertGreater(update_op.eval(feed_dict=feed_dict), .95)
      self.assertGreater(accuracy.eval(feed_dict=feed_dict), .95)

  def testMultipleUpdatesWithWeightedValues(self):
    with self.test_session() as sess:
      # Create the queue that populates the predictions.
      preds_queue = tf.FIFOQueue(4, dtypes=tf.float32, shapes=(1, 1))
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [2])
      _enqueue_vector(sess, preds_queue, [1])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      labels_queue = tf.FIFOQueue(4, dtypes=tf.float32, shapes=(1, 1))
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [2])
      labels = labels_queue.dequeue()

      # Create the queue that populates the weights.
      weights_queue = tf.FIFOQueue(4, dtypes=tf.int64, shapes=(1, 1))
      _enqueue_vector(sess, weights_queue, [1])
      _enqueue_vector(sess, weights_queue, [1])
      _enqueue_vector(sess, weights_queue, [0])
      _enqueue_vector(sess, weights_queue, [0])
      weights = weights_queue.dequeue()

      accuracy, update_op = metrics.streaming_accuracy(
          predictions, labels, weights)

      sess.run(tf.local_variables_initializer())
      for _ in xrange(3):
        sess.run(update_op)
      self.assertEqual(1.0, sess.run(update_op))
      self.assertEqual(1.0, accuracy.eval())


class StreamingTruePositivesTest(tf.test.TestCase):

  def setUp(self):
    np.random.seed(1)
    tf.reset_default_graph()

  def testVars(self):
    metrics.streaming_true_positives((0, 1, 0), (0, 1, 1))
    _assert_local_variables(self, ('true_positives/count:0',))

  def testUnweighted(self):
    predictions = tf.constant((
        (1, 0, 1, 0),
        (0, 1, 1, 1),
        (0, 0, 0, 0)))
    labels = tf.constant((
        (0, 1, 1, 0),
        (1, 0, 0, 0),
        (0, 0, 0, 0)))
    tp, tp_update_op = metrics.streaming_true_positives(predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertEqual(0, tp.eval())
      self.assertEqual(1, tp_update_op.eval())
      self.assertEqual(1, tp.eval())

  def testWeighted(self):
    predictions = tf.constant((
        (1, 0, 1, 0),
        (0, 1, 1, 1),
        (0, 0, 0, 0)))
    labels = tf.constant((
        (0, 1, 1, 0),
        (1, 0, 0, 0),
        (0, 0, 0, 0)))
    tp, tp_update_op = metrics.streaming_true_positives(
        predictions, labels, weights=(37.0,))

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertEqual(0, tp.eval())
      self.assertEqual(37.0, tp_update_op.eval())
      self.assertEqual(37.0, tp.eval())


class StreamingFalseNegativesTest(tf.test.TestCase):

  def setUp(self):
    np.random.seed(1)
    tf.reset_default_graph()

  def testVars(self):
    metrics.streaming_false_negatives((0, 1, 0), (0, 1, 1))
    _assert_local_variables(self, ('false_negatives/count:0',))

  def testUnweighted(self):
    predictions = tf.constant((
        (1, 0, 1, 0),
        (0, 1, 1, 1),
        (0, 0, 0, 0)))
    labels = tf.constant((
        (0, 1, 1, 0),
        (1, 0, 0, 0),
        (0, 0, 0, 0)))
    fn, fn_update_op = metrics.streaming_false_negatives(predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertEqual(0, fn.eval())
      self.assertEqual(2, fn_update_op.eval())
      self.assertEqual(2, fn.eval())

  def testWeighted(self):
    predictions = tf.constant((
        (1, 0, 1, 0),
        (0, 1, 1, 1),
        (0, 0, 0, 0)))
    labels = tf.constant((
        (0, 1, 1, 0),
        (1, 0, 0, 0),
        (0, 0, 0, 0)))
    fn, fn_update_op = metrics.streaming_false_negatives(
        predictions, labels, weights=((3.0,), (5.0,), (7.0,)))

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertEqual(0, fn.eval())
      self.assertEqual(8.0, fn_update_op.eval())
      self.assertEqual(8.0, fn.eval())


class StreamingFalsePositivesTest(tf.test.TestCase):

  def setUp(self):
    np.random.seed(1)
    tf.reset_default_graph()

  def testVars(self):
    metrics.streaming_false_positives((0, 1, 0), (0, 1, 1))
    _assert_local_variables(self, ('false_positives/count:0',))

  def testUnweighted(self):
    predictions = tf.constant((
        (1, 0, 1, 0),
        (0, 1, 1, 1),
        (0, 0, 0, 0)))
    labels = tf.constant((
        (0, 1, 1, 0),
        (1, 0, 0, 0),
        (0, 0, 0, 0)))
    fp, fp_update_op = metrics.streaming_false_positives(predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertEqual(0, fp.eval())
      self.assertEqual(4, fp_update_op.eval())
      self.assertEqual(4, fp.eval())

  def testWeighted(self):
    predictions = tf.constant((
        (1, 0, 1, 0),
        (0, 1, 1, 1),
        (0, 0, 0, 0)))
    labels = tf.constant((
        (0, 1, 1, 0),
        (1, 0, 0, 0),
        (0, 0, 0, 0)))
    fp, fp_update_op = metrics.streaming_false_positives(
        predictions, labels, weights=(
            (1.0, 2.0, 3.0, 5.0),
            (7.0, 11.0, 13.0, 17.0),
            (19.0, 23.0, 29.0, 31.0)))

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertEqual(0, fp.eval())
      self.assertEqual(42.0, fp_update_op.eval())
      self.assertEqual(42.0, fp.eval())


class StreamingTrueNegativesTest(tf.test.TestCase):

  def setUp(self):
    np.random.seed(1)
    tf.reset_default_graph()

  def testVars(self):
    metrics.streaming_true_negatives((0, 1, 0), (0, 1, 1))
    _assert_local_variables(self, ('true_negatives/count:0',))

  def testUnweighted(self):
    predictions = tf.constant((
        (1, 0, 1, 0),
        (0, 1, 1, 1),
        (0, 0, 0, 0)))
    labels = tf.constant((
        (0, 1, 1, 0),
        (1, 0, 0, 0),
        (0, 0, 0, 0)))
    tn, tn_update_op = metrics.streaming_true_negatives(predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertEqual(0, tn.eval())
      self.assertEqual(5, tn_update_op.eval())
      self.assertEqual(5, tn.eval())

  def testWeighted(self):
    predictions = tf.constant((
        (1, 0, 1, 0),
        (0, 1, 1, 1),
        (0, 0, 0, 0)))
    labels = tf.constant((
        (0, 1, 1, 0),
        (1, 0, 0, 0),
        (0, 0, 0, 0)))
    tn, tn_update_op = metrics.streaming_true_negatives(
        predictions, labels, weights=(0.0, 2.0, 3.0, 5.0))

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertEqual(0, tn.eval())
      self.assertEqual(15.0, tn_update_op.eval())
      self.assertEqual(15.0, tn.eval())


class StreamingTruePositivesAtThresholdsTest(tf.test.TestCase):

  def setUp(self):
    np.random.seed(1)
    tf.reset_default_graph()

  def testVars(self):
    metrics.streaming_true_positives_at_thresholds(
        (0.0, 1.0, 0.0), (0, 1, 1), thresholds=(0.15, 0.5, 0.85,))
    _assert_local_variables(self, ('true_positives:0',))

  def testUnweighted(self):
    predictions = tf.constant((
        (0.9, 0.2, 0.8, 0.1),
        (0.2, 0.9, 0.7, 0.6),
        (0.1, 0.2, 0.4, 0.3)))
    labels = tf.constant((
        (0, 1, 1, 0),
        (1, 0, 0, 0),
        (0, 0, 0, 0)))
    tp, tp_update_op = metrics.streaming_true_positives_at_thresholds(
        predictions, labels, thresholds=(0.15, 0.5, 0.85,))

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertAllEqual((0, 0, 0), tp.eval())
      self.assertAllEqual((3, 1, 0), tp_update_op.eval())
      self.assertAllEqual((3, 1, 0), tp.eval())

  def testWeighted(self):
    predictions = tf.constant((
        (0.9, 0.2, 0.8, 0.1),
        (0.2, 0.9, 0.7, 0.6),
        (0.1, 0.2, 0.4, 0.3)))
    labels = tf.constant((
        (0, 1, 1, 0),
        (1, 0, 0, 0),
        (0, 0, 0, 0)))
    tp, tp_update_op = metrics.streaming_true_positives_at_thresholds(
        predictions, labels, weights=(37.0,), thresholds=(0.15, 0.5, 0.85,))

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertAllEqual((0.0, 0.0, 0.0), tp.eval())
      self.assertAllEqual((111.0, 37.0, 0.0), tp_update_op.eval())
      self.assertAllEqual((111.0, 37.0, 0.0), tp.eval())


class StreamingFalseNegativesAtThresholdsTest(tf.test.TestCase):

  def setUp(self):
    np.random.seed(1)
    tf.reset_default_graph()

  def testVars(self):
    metrics.streaming_false_negatives_at_thresholds(
        (0.0, 1.0, 0.0), (0, 1, 1), thresholds=(0.15, 0.5, 0.85,))
    _assert_local_variables(self, ('false_negatives:0',))

  def testUnweighted(self):
    predictions = tf.constant((
        (0.9, 0.2, 0.8, 0.1),
        (0.2, 0.9, 0.7, 0.6),
        (0.1, 0.2, 0.4, 0.3)))
    labels = tf.constant((
        (0, 1, 1, 0),
        (1, 0, 0, 0),
        (0, 0, 0, 0)))
    fn, fn_update_op = metrics.streaming_false_negatives_at_thresholds(
        predictions, labels, thresholds=(0.15, 0.5, 0.85,))

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertAllEqual((0, 0, 0), fn.eval())
      self.assertAllEqual((0, 2, 3), fn_update_op.eval())
      self.assertAllEqual((0, 2, 3), fn.eval())

  def testWeighted(self):
    predictions = tf.constant((
        (0.9, 0.2, 0.8, 0.1),
        (0.2, 0.9, 0.7, 0.6),
        (0.1, 0.2, 0.4, 0.3)))
    labels = tf.constant((
        (0, 1, 1, 0),
        (1, 0, 0, 0),
        (0, 0, 0, 0)))
    fn, fn_update_op = metrics.streaming_false_negatives_at_thresholds(
        predictions, labels, weights=((3.0,), (5.0,), (7.0,)),
        thresholds=(0.15, 0.5, 0.85,))

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertAllEqual((0.0, 0.0, 0.0), fn.eval())
      self.assertAllEqual((0.0, 8.0, 11.0), fn_update_op.eval())
      self.assertAllEqual((0.0, 8.0, 11.0), fn.eval())


class StreamingFalsePositivesAtThresholdsTest(tf.test.TestCase):

  def setUp(self):
    np.random.seed(1)
    tf.reset_default_graph()

  def testVars(self):
    metrics.streaming_false_positives_at_thresholds(
        (0.0, 1.0, 0.0), (0, 1, 1), thresholds=(0.15, 0.5, 0.85,))
    _assert_local_variables(self, ('false_positives:0',))

  def testUnweighted(self):
    predictions = tf.constant((
        (0.9, 0.2, 0.8, 0.1),
        (0.2, 0.9, 0.7, 0.6),
        (0.1, 0.2, 0.4, 0.3)))
    labels = tf.constant((
        (0, 1, 1, 0),
        (1, 0, 0, 0),
        (0, 0, 0, 0)))
    fp, fp_update_op = metrics.streaming_false_positives_at_thresholds(
        predictions, labels, thresholds=(0.15, 0.5, 0.85,))

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertAllEqual((0, 0, 0), fp.eval())
      self.assertAllEqual((7, 4, 2), fp_update_op.eval())
      self.assertAllEqual((7, 4, 2), fp.eval())

  def testWeighted(self):
    predictions = tf.constant((
        (0.9, 0.2, 0.8, 0.1),
        (0.2, 0.9, 0.7, 0.6),
        (0.1, 0.2, 0.4, 0.3)))
    labels = tf.constant((
        (0, 1, 1, 0),
        (1, 0, 0, 0),
        (0, 0, 0, 0)))
    fp, fp_update_op = metrics.streaming_false_positives_at_thresholds(
        predictions, labels, weights=(
            (1.0, 2.0, 3.0, 5.0),
            (7.0, 11.0, 13.0, 17.0),
            (19.0, 23.0, 29.0, 31.0)), thresholds=(0.15, 0.5, 0.85,))

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertAllEqual((0.0, 0.0, 0.0), fp.eval())
      self.assertAllEqual((125.0, 42.0, 12.0), fp_update_op.eval())
      self.assertAllEqual((125.0, 42.0, 12.0), fp.eval())


class StreamingTrueNegativesAtThresholdsTest(tf.test.TestCase):

  def setUp(self):
    np.random.seed(1)
    tf.reset_default_graph()

  def testVars(self):
    metrics.streaming_true_negatives_at_thresholds(
        (0.0, 1.0, 0.0), (0, 1, 1), thresholds=(0.15, 0.5, 0.85,))
    _assert_local_variables(self, ('true_negatives:0',))

  def testUnweighted(self):
    predictions = tf.constant((
        (0.9, 0.2, 0.8, 0.1),
        (0.2, 0.9, 0.7, 0.6),
        (0.1, 0.2, 0.4, 0.3)))
    labels = tf.constant((
        (0, 1, 1, 0),
        (1, 0, 0, 0),
        (0, 0, 0, 0)))
    tn, tn_update_op = metrics.streaming_true_negatives_at_thresholds(
        predictions, labels, thresholds=(0.15, 0.5, 0.85,))

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertAllEqual((0, 0, 0), tn.eval())
      self.assertAllEqual((2, 5, 7), tn_update_op.eval())
      self.assertAllEqual((2, 5, 7), tn.eval())

  def testWeighted(self):
    predictions = tf.constant((
        (0.9, 0.2, 0.8, 0.1),
        (0.2, 0.9, 0.7, 0.6),
        (0.1, 0.2, 0.4, 0.3)))
    labels = tf.constant((
        (0, 1, 1, 0),
        (1, 0, 0, 0),
        (0, 0, 0, 0)))
    tn, tn_update_op = metrics.streaming_true_negatives_at_thresholds(
        predictions, labels, weights=(0.0, 2.0, 3.0, 5.0),
        thresholds=(0.15, 0.5, 0.85,))

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertAllEqual((0.0, 0.0, 0.0), tn.eval())
      self.assertAllEqual((5.0, 15.0, 23.0), tn_update_op.eval())
      self.assertAllEqual((5.0, 15.0, 23.0), tn.eval())


class StreamingPrecisionTest(tf.test.TestCase):

  def setUp(self):
    np.random.seed(1)
    tf.reset_default_graph()

  def testVars(self):
    metrics.streaming_precision(
        predictions=tf.ones((10, 1)), labels=tf.ones((10, 1)))
    _assert_local_variables(self, (
        'precision/false_positives/count:0',
        'precision/true_positives/count:0'
    ))

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = metrics.streaming_precision(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [mean])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.streaming_precision(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testValueTensorIsIdempotent(self):
    predictions = tf.random_uniform((10, 3), maxval=1, dtype=tf.int64, seed=1)
    labels = tf.random_uniform((10, 3), maxval=1, dtype=tf.int64, seed=1)
    precision, update_op = metrics.streaming_precision(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        sess.run(update_op)

      # Then verify idempotency.
      initial_precision = precision.eval()
      for _ in range(10):
        self.assertEqual(initial_precision, precision.eval())

  def testAllCorrect(self):
    inputs = np.random.randint(0, 2, size=(100, 1))

    predictions = tf.constant(inputs)
    labels = tf.constant(inputs)
    precision, update_op = metrics.streaming_precision(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertAlmostEqual(1, sess.run(update_op))
      self.assertAlmostEqual(1, precision.eval())

  def testSomeCorrect(self):
    predictions = tf.constant([1, 0, 1, 0], shape=(1, 4))
    labels = tf.constant([0, 1, 1, 0], shape=(1, 4))
    precision, update_op = metrics.streaming_precision(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertAlmostEqual(0.5, update_op.eval())
      self.assertAlmostEqual(0.5, precision.eval())

  def testWeighted1d(self):
    predictions = tf.constant([[1, 0, 1, 0], [1, 0, 1, 0]])
    labels = tf.constant([[0, 1, 1, 0], [1, 0, 0, 1]])
    precision, update_op = metrics.streaming_precision(
        predictions, labels, weights=tf.constant([[2], [5]]))

    with self.test_session():
      tf.local_variables_initializer().run()
      weighted_tp = 2.0 + 5.0
      weighted_positives = (2.0 + 2.0) + (5.0 + 5.0)
      expected_precision = weighted_tp / weighted_positives
      self.assertAlmostEqual(expected_precision, update_op.eval())
      self.assertAlmostEqual(expected_precision, precision.eval())

  def testWeighted1d_placeholders(self):
    predictions = tf.placeholder(dtype=tf.float32)
    labels = tf.placeholder(dtype=tf.float32)
    feed_dict = {
        predictions: ((1, 0, 1, 0), (1, 0, 1, 0)),
        labels: ((0, 1, 1, 0), (1, 0, 0, 1))
    }
    precision, update_op = metrics.streaming_precision(
        predictions, labels, weights=tf.constant([[2], [5]]))

    with self.test_session():
      tf.local_variables_initializer().run()
      weighted_tp = 2.0 + 5.0
      weighted_positives = (2.0 + 2.0) + (5.0 + 5.0)
      expected_precision = weighted_tp / weighted_positives
      self.assertAlmostEqual(
          expected_precision, update_op.eval(feed_dict=feed_dict))
      self.assertAlmostEqual(
          expected_precision, precision.eval(feed_dict=feed_dict))

  def testWeighted2d(self):
    predictions = tf.constant([[1, 0, 1, 0], [1, 0, 1, 0]])
    labels = tf.constant([[0, 1, 1, 0], [1, 0, 0, 1]])
    precision, update_op = metrics.streaming_precision(
        predictions, labels, weights=tf.constant([[1, 2, 3, 4], [4, 3, 2, 1]]))

    with self.test_session():
      tf.local_variables_initializer().run()
      weighted_tp = 3.0 + 4.0
      weighted_positives = (1.0 + 3.0) + (4.0 + 2.0)
      expected_precision = weighted_tp / weighted_positives
      self.assertAlmostEqual(expected_precision, update_op.eval())
      self.assertAlmostEqual(expected_precision, precision.eval())

  def testWeighted2d_placeholders(self):
    predictions = tf.placeholder(dtype=tf.float32)
    labels = tf.placeholder(dtype=tf.float32)
    feed_dict = {
        predictions: ((1, 0, 1, 0), (1, 0, 1, 0)),
        labels: ((0, 1, 1, 0), (1, 0, 0, 1))
    }
    precision, update_op = metrics.streaming_precision(
        predictions, labels, weights=tf.constant([[1, 2, 3, 4], [4, 3, 2, 1]]))

    with self.test_session():
      tf.local_variables_initializer().run()
      weighted_tp = 3.0 + 4.0
      weighted_positives = (1.0 + 3.0) + (4.0 + 2.0)
      expected_precision = weighted_tp / weighted_positives
      self.assertAlmostEqual(
          expected_precision, update_op.eval(feed_dict=feed_dict))
      self.assertAlmostEqual(
          expected_precision, precision.eval(feed_dict=feed_dict))

  def testAllIncorrect(self):
    inputs = np.random.randint(0, 2, size=(100, 1))

    predictions = tf.constant(inputs)
    labels = tf.constant(1 - inputs)
    precision, update_op = metrics.streaming_precision(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(update_op)
      self.assertAlmostEqual(0, precision.eval())

  def testZeroTrueAndFalsePositivesGivesZeroPrecision(self):
    predictions = tf.constant([0, 0, 0, 0])
    labels = tf.constant([0, 0, 0, 0])
    precision, update_op = metrics.streaming_precision(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(update_op)
      self.assertEqual(0.0, precision.eval())


class StreamingRecallTest(tf.test.TestCase):

  def setUp(self):
    np.random.seed(1)
    tf.reset_default_graph()

  def testVars(self):
    metrics.streaming_recall(
        predictions=tf.ones((10, 1)), labels=tf.ones((10, 1)))
    _assert_local_variables(self, (
        'recall/false_negatives/count:0',
        'recall/true_positives/count:0'
    ))

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = metrics.streaming_recall(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [mean])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.streaming_recall(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testValueTensorIsIdempotent(self):
    predictions = tf.random_uniform((10, 3), maxval=1, dtype=tf.int64, seed=1)
    labels = tf.random_uniform((10, 3), maxval=1, dtype=tf.int64, seed=1)
    recall, update_op = metrics.streaming_recall(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        sess.run(update_op)

      # Then verify idempotency.
      initial_recall = recall.eval()
      for _ in range(10):
        self.assertEqual(initial_recall, recall.eval())

  def testAllCorrect(self):
    np_inputs = np.random.randint(0, 2, size=(100, 1))

    predictions = tf.constant(np_inputs)
    labels = tf.constant(np_inputs)
    recall, update_op = metrics.streaming_recall(predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(update_op)
      self.assertEqual(1, recall.eval())

  def testSomeCorrect(self):
    predictions = tf.constant([1, 0, 1, 0], shape=(1, 4))
    labels = tf.constant([0, 1, 1, 0], shape=(1, 4))
    recall, update_op = metrics.streaming_recall(predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertAlmostEqual(0.5, update_op.eval())
      self.assertAlmostEqual(0.5, recall.eval())

  def testWeighted1d(self):
    predictions = tf.constant([[1, 0, 1, 0], [0, 1, 0, 1]])
    labels = tf.constant([[0, 1, 1, 0], [1, 0, 0, 1]])
    weights = tf.constant([[2], [5]])
    recall, update_op = metrics.streaming_recall(
        predictions, labels, weights=weights)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      weighted_tp = 2.0 + 5.0
      weighted_t = (2.0 + 2.0) + (5.0 + 5.0)
      expected_precision = weighted_tp / weighted_t
      self.assertAlmostEqual(expected_precision, update_op.eval())
      self.assertAlmostEqual(expected_precision, recall.eval())

  def testWeighted2d(self):
    predictions = tf.constant([[1, 0, 1, 0], [0, 1, 0, 1]])
    labels = tf.constant([[0, 1, 1, 0], [1, 0, 0, 1]])
    weights = tf.constant([[1, 2, 3, 4], [4, 3, 2, 1]])
    recall, update_op = metrics.streaming_recall(
        predictions, labels, weights=weights)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      weighted_tp = 3.0 + 1.0
      weighted_t = (2.0 + 3.0) + (4.0 + 1.0)
      expected_precision = weighted_tp / weighted_t
      self.assertAlmostEqual(expected_precision, update_op.eval())
      self.assertAlmostEqual(expected_precision, recall.eval())

  def testAllIncorrect(self):
    np_inputs = np.random.randint(0, 2, size=(100, 1))

    predictions = tf.constant(np_inputs)
    labels = tf.constant(1 - np_inputs)
    recall, update_op = metrics.streaming_recall(predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(update_op)
      self.assertEqual(0, recall.eval())

  def testZeroTruePositivesAndFalseNegativesGivesZeroRecall(self):
    predictions = tf.zeros((1, 4))
    labels = tf.zeros((1, 4))
    recall, update_op = metrics.streaming_recall(predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(update_op)
      self.assertEqual(0, recall.eval())


class StreamingAUCTest(tf.test.TestCase):

  def setUp(self):
    np.random.seed(1)
    tf.reset_default_graph()

  def testVars(self):
    metrics.streaming_auc(
        predictions=tf.ones((10, 1)), labels=tf.ones((10, 1)))
    _assert_local_variables(self, (
        'auc/true_positives:0',
        'auc/false_negatives:0',
        'auc/false_positives:0',
        'auc/true_negatives:0'
    ))

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = metrics.streaming_auc(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [mean])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.streaming_auc(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testValueTensorIsIdempotent(self):
    predictions = tf.random_uniform((10, 3), maxval=1, dtype=tf.float32, seed=1)
    labels = tf.random_uniform((10, 3), maxval=1, dtype=tf.int64, seed=1)
    auc, update_op = metrics.streaming_auc(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        sess.run(update_op)

      # Then verify idempotency.
      initial_auc = auc.eval()
      for _ in range(10):
        self.assertAlmostEqual(initial_auc, auc.eval(), 5)

  def testAllCorrect(self):
    self.allCorrectAsExpected('ROC')

  def allCorrectAsExpected(self, curve):
    inputs = np.random.randint(0, 2, size=(100, 1))

    with self.test_session() as sess:
      predictions = tf.constant(inputs, dtype=tf.float32)
      labels = tf.constant(inputs)
      auc, update_op = metrics.streaming_auc(predictions, labels, curve=curve)

      sess.run(tf.local_variables_initializer())
      self.assertEqual(1, sess.run(update_op))

      self.assertEqual(1, auc.eval())

  def testSomeCorrect(self):
    with self.test_session() as sess:
      predictions = tf.constant([1, 0, 1, 0], shape=(1, 4), dtype=tf.float32)
      labels = tf.constant([0, 1, 1, 0], shape=(1, 4))
      auc, update_op = metrics.streaming_auc(predictions, labels)

      sess.run(tf.local_variables_initializer())
      self.assertAlmostEqual(0.5, sess.run(update_op))

      self.assertAlmostEqual(0.5, auc.eval())

  def testWeighted1d(self):
    with self.test_session() as sess:
      predictions = tf.constant([1, 0, 1, 0], shape=(1, 4), dtype=tf.float32)
      labels = tf.constant([0, 1, 1, 0], shape=(1, 4))
      weights = tf.constant([2], shape=(1, 1))
      auc, update_op = metrics.streaming_auc(predictions, labels,
                                             weights=weights)

      sess.run(tf.local_variables_initializer())
      self.assertAlmostEqual(0.5, sess.run(update_op), 5)

      self.assertAlmostEqual(0.5, auc.eval(), 5)

  def testWeighted2d(self):
    with self.test_session() as sess:
      predictions = tf.constant([1, 0, 1, 0], shape=(1, 4), dtype=tf.float32)
      labels = tf.constant([0, 1, 1, 0], shape=(1, 4))
      weights = tf.constant([1, 2, 3, 4], shape=(1, 4))
      auc, update_op = metrics.streaming_auc(predictions, labels,
                                             weights=weights)

      sess.run(tf.local_variables_initializer())
      self.assertAlmostEqual(0.7, sess.run(update_op), 5)

      self.assertAlmostEqual(0.7, auc.eval(), 5)

  def testAUCPRSpecialCase(self):
    with self.test_session() as sess:
      predictions = tf.constant([0.1, 0.4, 0.35, 0.8],
                                shape=(1, 4), dtype=tf.float32)
      labels = tf.constant([0, 0, 1, 1], shape=(1, 4))
      auc, update_op = metrics.streaming_auc(predictions, labels, curve='PR')

      sess.run(tf.local_variables_initializer())
      self.assertAlmostEqual(0.79166, sess.run(update_op), delta=1e-3)

      self.assertAlmostEqual(0.79166, auc.eval(), delta=1e-3)

  def testAnotherAUCPRSpecialCase(self):
    with self.test_session() as sess:
      predictions = tf.constant([0.1, 0.4, 0.35, 0.8, 0.1, 0.135, 0.81],
                                shape=(1, 7), dtype=tf.float32)
      labels = tf.constant([0, 0, 1, 0, 1, 0, 1], shape=(1, 7))
      auc, update_op = metrics.streaming_auc(predictions, labels, curve='PR')

      sess.run(tf.local_variables_initializer())
      self.assertAlmostEqual(0.610317, sess.run(update_op), delta=1e-3)

      self.assertAlmostEqual(0.610317, auc.eval(), delta=1e-3)

  def testThirdAUCPRSpecialCase(self):
    with self.test_session() as sess:
      predictions = tf.constant([0.0, 0.1, 0.2, 0.33, 0.3, 0.4, 0.5],
                                shape=(1, 7), dtype=tf.float32)
      labels = tf.constant([0, 0, 0, 0, 1, 1, 1], shape=(1, 7))
      auc, update_op = metrics.streaming_auc(predictions, labels, curve='PR')

      sess.run(tf.local_variables_initializer())
      self.assertAlmostEqual(0.90277, sess.run(update_op), delta=1e-3)

      self.assertAlmostEqual(0.90277, auc.eval(), delta=1e-3)

  def testAllIncorrect(self):
    inputs = np.random.randint(0, 2, size=(100, 1))

    with self.test_session() as sess:
      predictions = tf.constant(inputs, dtype=tf.float32)
      labels = tf.constant(1 - inputs, dtype=tf.float32)
      auc, update_op = metrics.streaming_auc(predictions, labels)

      sess.run(tf.local_variables_initializer())
      self.assertAlmostEqual(0, sess.run(update_op))

      self.assertAlmostEqual(0, auc.eval())

  def testZeroTruePositivesAndFalseNegativesGivesOneAUC(self):
    with self.test_session() as sess:
      predictions = tf.zeros([4], dtype=tf.float32)
      labels = tf.zeros([4])
      auc, update_op = metrics.streaming_auc(predictions, labels)

      sess.run(tf.local_variables_initializer())
      self.assertAlmostEqual(1, sess.run(update_op), 6)

      self.assertAlmostEqual(1, auc.eval(), 6)

  def testRecallOneAndPrecisionOneGivesOnePRAUC(self):
    with self.test_session() as sess:
      predictions = tf.ones([4], dtype=tf.float32)
      labels = tf.ones([4])
      auc, update_op = metrics.streaming_auc(predictions,
                                             labels,
                                             curve='PR')

      sess.run(tf.local_variables_initializer())
      self.assertAlmostEqual(1, sess.run(update_op), 6)

      self.assertAlmostEqual(1, auc.eval(), 6)

  def np_auc(self, predictions, labels, weights):
    """Computes the AUC explicitely using Numpy.

    Args:
      predictions: an ndarray with shape [N].
      labels: an ndarray with shape [N].
      weights: an ndarray with shape [N].

    Returns:
      the area under the ROC curve.
    """
    if weights is None:
      weights = np.ones(np.size(predictions))
    is_positive = labels > 0
    num_positives = np.sum(weights[is_positive])
    num_negatives = np.sum(weights[~is_positive])

    # Sort descending:
    inds = np.argsort(-predictions)

    sorted_labels = labels[inds]
    sorted_weights = weights[inds]
    is_positive = sorted_labels > 0

    tp = np.cumsum(sorted_weights * is_positive) / num_positives
    return np.sum((sorted_weights * tp)[~is_positive]) / num_negatives

  def testWithMultipleUpdates(self):
    num_samples = 1000
    batch_size = 10
    num_batches = int(num_samples / batch_size)

    # Create the labels and data.
    labels = np.random.randint(0, 2, size=num_samples)
    noise = np.random.normal(0.0, scale=0.2, size=num_samples)
    predictions = 0.4 + 0.2 * labels + noise
    predictions[predictions > 1] = 1
    predictions[predictions < 0] = 0

    def _enqueue_as_batches(x, enqueue_ops):
      x_batches = x.astype(np.float32).reshape((num_batches, batch_size))
      x_queue = tf.FIFOQueue(num_batches, dtypes=tf.float32,
                             shapes=(batch_size,))
      for i in range(num_batches):
        enqueue_ops[i].append(x_queue.enqueue(x_batches[i, :]))
      return x_queue.dequeue()

    for weights in (None,
                    np.ones(num_samples),
                    np.random.exponential(scale=1.0, size=num_samples)):
      expected_auc = self.np_auc(predictions, labels, weights)

      with self.test_session() as sess:
        enqueue_ops = [[] for i in range(num_batches)]
        tf_predictions = _enqueue_as_batches(predictions, enqueue_ops)
        tf_labels = _enqueue_as_batches(labels, enqueue_ops)
        tf_weights = (_enqueue_as_batches(weights, enqueue_ops)
                      if weights is not None else None)

        for i in range(num_batches):
          sess.run(enqueue_ops[i])

        auc, update_op = metrics.streaming_auc(
            tf_predictions, tf_labels, curve='ROC', num_thresholds=500,
            weights=tf_weights)

        sess.run(tf.local_variables_initializer())
        for i in range(num_batches):
          sess.run(update_op)

        # Since this is only approximate, we can't expect a 6 digits match.
        # Although with higher number of samples/thresholds we should see the
        # accuracy improving
        self.assertAlmostEqual(expected_auc, auc.eval(), 2)


class StreamingSpecificityAtSensitivityTest(tf.test.TestCase):

  def setUp(self):
    np.random.seed(1)
    tf.reset_default_graph()

  def testVars(self):
    metrics.streaming_specificity_at_sensitivity(
        predictions=tf.ones((10, 1)), labels=tf.ones((10, 1)), sensitivity=0.7)
    _assert_local_variables(self, (
        'specificity_at_sensitivity/true_positives:0',
        'specificity_at_sensitivity/false_negatives:0',
        'specificity_at_sensitivity/false_positives:0',
        'specificity_at_sensitivity/true_negatives:0'
    ))

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = metrics.streaming_specificity_at_sensitivity(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        sensitivity=0.7,
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [mean])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.streaming_specificity_at_sensitivity(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        sensitivity=0.7,
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testValueTensorIsIdempotent(self):
    predictions = tf.random_uniform((10, 3), maxval=1, dtype=tf.float32, seed=1)
    labels = tf.random_uniform((10, 3), maxval=2, dtype=tf.int64, seed=1)
    specificity, update_op = metrics.streaming_specificity_at_sensitivity(
        predictions, labels, sensitivity=0.7)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        sess.run(update_op)

      # Then verify idempotency.
      initial_specificity = specificity.eval()
      for _ in range(10):
        self.assertAlmostEqual(initial_specificity, specificity.eval(), 5)

  def testAllCorrect(self):
    inputs = np.random.randint(0, 2, size=(100, 1))

    predictions = tf.constant(inputs, dtype=tf.float32)
    labels = tf.constant(inputs)
    specificity, update_op = metrics.streaming_specificity_at_sensitivity(
        predictions, labels, sensitivity=0.7)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertEqual(1, sess.run(update_op))
      self.assertEqual(1, specificity.eval())

  def testSomeCorrectHighSensitivity(self):
    predictions_values = [0.1, 0.2, 0.4, 0.3, 0.0,
                          0.1, 0.45, 0.5, 0.8, 0.9]
    labels_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    predictions = tf.constant(predictions_values, dtype=tf.float32)
    labels = tf.constant(labels_values)
    specificity, update_op = metrics.streaming_specificity_at_sensitivity(
        predictions, labels, sensitivity=0.8)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertAlmostEqual(1.0, sess.run(update_op))
      self.assertAlmostEqual(1.0, specificity.eval())

  def testSomeCorrectLowSensitivity(self):
    predictions_values = [0.1, 0.2, 0.4, 0.3, 0.0,
                          0.1, 0.2, 0.2, 0.26, 0.26]
    labels_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    predictions = tf.constant(predictions_values, dtype=tf.float32)
    labels = tf.constant(labels_values)
    specificity, update_op = metrics.streaming_specificity_at_sensitivity(
        predictions, labels, sensitivity=0.4)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())

      self.assertAlmostEqual(0.6, sess.run(update_op))
      self.assertAlmostEqual(0.6, specificity.eval())

  def testWeighted1d(self):
    predictions_values = [0.1, 0.2, 0.4, 0.3, 0.0,
                          0.1, 0.2, 0.2, 0.26, 0.26]
    labels_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    weights_values = [3]

    predictions = tf.constant(predictions_values, dtype=tf.float32)
    labels = tf.constant(labels_values)
    weights = tf.constant(weights_values)
    specificity, update_op = metrics.streaming_specificity_at_sensitivity(
        predictions, labels, weights=weights, sensitivity=0.4)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())

      self.assertAlmostEqual(0.6, sess.run(update_op))
      self.assertAlmostEqual(0.6, specificity.eval())

  def testWeighted2d(self):
    predictions_values = [0.1, 0.2, 0.4, 0.3, 0.0,
                          0.1, 0.2, 0.2, 0.26, 0.26]
    labels_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    weights_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    predictions = tf.constant(predictions_values, dtype=tf.float32)
    labels = tf.constant(labels_values)
    weights = tf.constant(weights_values)
    specificity, update_op = metrics.streaming_specificity_at_sensitivity(
        predictions, labels, weights=weights, sensitivity=0.4)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())

      self.assertAlmostEqual(8.0 / 15.0, sess.run(update_op))
      self.assertAlmostEqual(8.0 / 15.0, specificity.eval())


class StreamingSensitivityAtSpecificityTest(tf.test.TestCase):

  def setUp(self):
    np.random.seed(1)
    tf.reset_default_graph()

  def testVars(self):
    metrics.streaming_sensitivity_at_specificity(
        predictions=tf.ones((10, 1)), labels=tf.ones((10, 1)), specificity=0.7)
    _assert_local_variables(self, (
        'sensitivity_at_specificity/true_positives:0',
        'sensitivity_at_specificity/false_negatives:0',
        'sensitivity_at_specificity/false_positives:0',
        'sensitivity_at_specificity/true_negatives:0'
    ))

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = metrics.streaming_sensitivity_at_specificity(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        specificity=0.7,
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [mean])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.streaming_sensitivity_at_specificity(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        specificity=0.7,
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testValueTensorIsIdempotent(self):
    predictions = tf.random_uniform((10, 3), maxval=1, dtype=tf.float32, seed=1)
    labels = tf.random_uniform((10, 3), maxval=2, dtype=tf.int64, seed=1)
    sensitivity, update_op = metrics.streaming_sensitivity_at_specificity(
        predictions, labels, specificity=0.7)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        sess.run(update_op)

      # Then verify idempotency.
      initial_sensitivity = sensitivity.eval()
      for _ in range(10):
        self.assertAlmostEqual(initial_sensitivity, sensitivity.eval(), 5)

  def testAllCorrect(self):
    inputs = np.random.randint(0, 2, size=(100, 1))

    predictions = tf.constant(inputs, dtype=tf.float32)
    labels = tf.constant(inputs)
    specificity, update_op = metrics.streaming_sensitivity_at_specificity(
        predictions, labels, specificity=0.7)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertEqual(1, sess.run(update_op))
      self.assertEqual(1, specificity.eval())

  def testSomeCorrectHighSpecificity(self):
    predictions_values = [0.0, 0.1, 0.2, 0.3, 0.4,
                          0.1, 0.45, 0.5, 0.8, 0.9]
    labels_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    predictions = tf.constant(predictions_values, dtype=tf.float32)
    labels = tf.constant(labels_values)
    specificity, update_op = metrics.streaming_sensitivity_at_specificity(
        predictions, labels, specificity=0.8)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertAlmostEqual(0.8, sess.run(update_op))
      self.assertAlmostEqual(0.8, specificity.eval())

  def testSomeCorrectLowSpecificity(self):
    predictions_values = [0.0, 0.1, 0.2, 0.3, 0.4,
                          0.01, 0.02, 0.25, 0.26, 0.26]
    labels_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    predictions = tf.constant(predictions_values, dtype=tf.float32)
    labels = tf.constant(labels_values)
    specificity, update_op = metrics.streaming_sensitivity_at_specificity(
        predictions, labels, specificity=0.4)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertAlmostEqual(0.6, sess.run(update_op))
      self.assertAlmostEqual(0.6, specificity.eval())

  def testWeighted(self):
    predictions_values = [0.0, 0.1, 0.2, 0.3, 0.4,
                          0.01, 0.02, 0.25, 0.26, 0.26]
    labels_values = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    weights_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    predictions = tf.constant(predictions_values, dtype=tf.float32)
    labels = tf.constant(labels_values)
    weights = tf.constant(weights_values)
    specificity, update_op = metrics.streaming_sensitivity_at_specificity(
        predictions, labels, weights=weights, specificity=0.4)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertAlmostEqual(0.675, sess.run(update_op))
      self.assertAlmostEqual(0.675, specificity.eval())


# TODO(nsilberman): Break this up into two sets of tests.
class StreamingPrecisionRecallThresholdsTest(tf.test.TestCase):

  def setUp(self):
    np.random.seed(1)
    tf.reset_default_graph()

  def testVars(self):
    metrics.streaming_precision_at_thresholds(
        predictions=tf.ones((10, 1)), labels=tf.ones((10, 1)),
        thresholds=[0, 0.5, 1.0])
    _assert_local_variables(self, (
        'precision_at_thresholds/true_positives:0',
        'precision_at_thresholds/false_positives:0',
    ))

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    prec, _ = metrics.streaming_precision_at_thresholds(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        thresholds=[0, 0.5, 1.0],
        metrics_collections=[my_collection_name])
    rec, _ = metrics.streaming_recall_at_thresholds(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        thresholds=[0, 0.5, 1.0],
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [prec, rec])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, precision_op = metrics.streaming_precision_at_thresholds(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        thresholds=[0, 0.5, 1.0],
        updates_collections=[my_collection_name])
    _, recall_op = metrics.streaming_recall_at_thresholds(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        thresholds=[0, 0.5, 1.0],
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name),
                         [precision_op, recall_op])

  def testValueTensorIsIdempotent(self):
    predictions = tf.random_uniform((10, 3), maxval=1, dtype=tf.float32, seed=1)
    labels = tf.random_uniform((10, 3), maxval=1, dtype=tf.int64, seed=1)
    thresholds = [0, 0.5, 1.0]
    prec, prec_op = metrics.streaming_precision_at_thresholds(
        predictions, labels, thresholds)
    rec, rec_op = metrics.streaming_recall_at_thresholds(
        predictions, labels, thresholds)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())

      # Run several updates, then verify idempotency.
      sess.run([prec_op, rec_op])
      initial_prec = prec.eval()
      initial_rec = rec.eval()
      for _ in range(10):
        sess.run([prec_op, rec_op])
        self.assertAllClose(initial_prec, prec.eval())
        self.assertAllClose(initial_rec, rec.eval())

  # TODO(nsilberman): fix tests (passing but incorrect).
  def testAllCorrect(self):
    inputs = np.random.randint(0, 2, size=(100, 1))

    with self.test_session() as sess:
      predictions = tf.constant(inputs, dtype=tf.float32)
      labels = tf.constant(inputs)
      thresholds = [0.5]
      prec, prec_op = metrics.streaming_precision_at_thresholds(
          predictions, labels, thresholds)
      rec, rec_op = metrics.streaming_recall_at_thresholds(
          predictions, labels, thresholds)

      sess.run(tf.local_variables_initializer())
      sess.run([prec_op, rec_op])

      self.assertEqual(1, prec.eval())
      self.assertEqual(1, rec.eval())

  def testSomeCorrect(self):
    with self.test_session() as sess:
      predictions = tf.constant([1, 0, 1, 0], shape=(1, 4), dtype=tf.float32)
      labels = tf.constant([0, 1, 1, 0], shape=(1, 4))
      thresholds = [0.5]
      prec, prec_op = metrics.streaming_precision_at_thresholds(
          predictions, labels, thresholds)
      rec, rec_op = metrics.streaming_recall_at_thresholds(
          predictions, labels, thresholds)

      sess.run(tf.local_variables_initializer())
      sess.run([prec_op, rec_op])

      self.assertAlmostEqual(0.5, prec.eval())
      self.assertAlmostEqual(0.5, rec.eval())

  def testAllIncorrect(self):
    inputs = np.random.randint(0, 2, size=(100, 1))

    with self.test_session() as sess:
      predictions = tf.constant(inputs, dtype=tf.float32)
      labels = tf.constant(1 - inputs, dtype=tf.float32)
      thresholds = [0.5]
      prec, prec_op = metrics.streaming_precision_at_thresholds(
          predictions, labels, thresholds)
      rec, rec_op = metrics.streaming_recall_at_thresholds(
          predictions, labels, thresholds)

      sess.run(tf.local_variables_initializer())
      sess.run([prec_op, rec_op])

      self.assertAlmostEqual(0, prec.eval())
      self.assertAlmostEqual(0, rec.eval())

  def testWeights1d(self):
    with self.test_session() as sess:
      predictions = tf.constant([[1, 0], [1, 0]], shape=(2, 2),
                                dtype=tf.float32)
      labels = tf.constant([[0, 1], [1, 0]], shape=(2, 2))
      weights = tf.constant([[0], [1]], shape=(2, 1), dtype=tf.float32)
      thresholds = [0.5, 1.1]
      prec, prec_op = metrics.streaming_precision_at_thresholds(
          predictions, labels, thresholds, weights=weights)
      rec, rec_op = metrics.streaming_recall_at_thresholds(
          predictions, labels, thresholds, weights=weights)

      [prec_low, prec_high] = tf.split(0, 2, prec)
      prec_low = tf.reshape(prec_low, shape=())
      prec_high = tf.reshape(prec_high, shape=())
      [rec_low, rec_high] = tf.split(0, 2, rec)
      rec_low = tf.reshape(rec_low, shape=())
      rec_high = tf.reshape(rec_high, shape=())

      sess.run(tf.local_variables_initializer())
      sess.run([prec_op, rec_op])

      self.assertAlmostEqual(1.0, prec_low.eval(), places=5)
      self.assertAlmostEqual(0.0, prec_high.eval(), places=5)
      self.assertAlmostEqual(1.0, rec_low.eval(), places=5)
      self.assertAlmostEqual(0.0, rec_high.eval(), places=5)

  def testWeights2d(self):
    with self.test_session() as sess:
      predictions = tf.constant([[1, 0], [1, 0]], shape=(2, 2),
                                dtype=tf.float32)
      labels = tf.constant([[0, 1], [1, 0]], shape=(2, 2))
      weights = tf.constant([[0, 0], [1, 1]], shape=(2, 2), dtype=tf.float32)
      thresholds = [0.5, 1.1]
      prec, prec_op = metrics.streaming_precision_at_thresholds(
          predictions, labels, thresholds, weights=weights)
      rec, rec_op = metrics.streaming_recall_at_thresholds(
          predictions, labels, thresholds, weights=weights)

      [prec_low, prec_high] = tf.split(0, 2, prec)
      prec_low = tf.reshape(prec_low, shape=())
      prec_high = tf.reshape(prec_high, shape=())
      [rec_low, rec_high] = tf.split(0, 2, rec)
      rec_low = tf.reshape(rec_low, shape=())
      rec_high = tf.reshape(rec_high, shape=())

      sess.run(tf.local_variables_initializer())
      sess.run([prec_op, rec_op])

      self.assertAlmostEqual(1.0, prec_low.eval(), places=5)
      self.assertAlmostEqual(0.0, prec_high.eval(), places=5)
      self.assertAlmostEqual(1.0, rec_low.eval(), places=5)
      self.assertAlmostEqual(0.0, rec_high.eval(), places=5)

  def testExtremeThresholds(self):
    with self.test_session() as sess:
      predictions = tf.constant([1, 0, 1, 0], shape=(1, 4), dtype=tf.float32)
      labels = tf.constant([0, 1, 1, 1], shape=(1, 4))
      thresholds = [-1.0, 2.0]  # lower/higher than any values
      prec, prec_op = metrics.streaming_precision_at_thresholds(
          predictions, labels, thresholds)
      rec, rec_op = metrics.streaming_recall_at_thresholds(
          predictions, labels, thresholds)

      [prec_low, prec_high] = tf.split(0, 2, prec)
      [rec_low, rec_high] = tf.split(0, 2, rec)

      sess.run(tf.local_variables_initializer())
      sess.run([prec_op, rec_op])

      self.assertAlmostEqual(0.75, prec_low.eval())
      self.assertAlmostEqual(0.0, prec_high.eval())
      self.assertAlmostEqual(1.0, rec_low.eval())
      self.assertAlmostEqual(0.0, rec_high.eval())

  def testZeroLabelsPredictions(self):
    with self.test_session() as sess:
      predictions = tf.zeros([4], dtype=tf.float32)
      labels = tf.zeros([4])
      thresholds = [0.5]
      prec, prec_op = metrics.streaming_precision_at_thresholds(
          predictions, labels, thresholds)
      rec, rec_op = metrics.streaming_recall_at_thresholds(
          predictions, labels, thresholds)

      sess.run(tf.local_variables_initializer())
      sess.run([prec_op, rec_op])

      self.assertAlmostEqual(0, prec.eval(), 6)
      self.assertAlmostEqual(0, rec.eval(), 6)

  def testWithMultipleUpdates(self):
    num_samples = 1000
    batch_size = 10
    num_batches = int(num_samples / batch_size)

    # Create the labels and data.
    labels = np.random.randint(0, 2, size=(num_samples, 1))
    noise = np.random.normal(0.0, scale=0.2, size=(num_samples, 1))
    predictions = 0.4 + 0.2 * labels + noise
    predictions[predictions > 1] = 1
    predictions[predictions < 0] = 0
    thresholds = [0.3]

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(num_samples):
      if predictions[i] > thresholds[0]:
        if labels[i] == 1:
          tp += 1
        else:
          fp += 1
      else:
        if labels[i] == 1:
          fn += 1
        else:
          tn += 1
    epsilon = 1e-7
    expected_prec = tp / (epsilon + tp + fp)
    expected_rec = tp / (epsilon + tp + fn)

    labels = labels.astype(np.float32)
    predictions = predictions.astype(np.float32)

    with self.test_session() as sess:
      # Reshape the data so its easy to queue up:
      predictions_batches = predictions.reshape((batch_size, num_batches))
      labels_batches = labels.reshape((batch_size, num_batches))

      # Enqueue the data:
      predictions_queue = tf.FIFOQueue(num_batches, dtypes=tf.float32,
                                       shapes=(batch_size,))
      labels_queue = tf.FIFOQueue(num_batches, dtypes=tf.float32,
                                  shapes=(batch_size,))

      for i in range(int(num_batches)):
        tf_prediction = tf.constant(predictions_batches[:, i])
        tf_label = tf.constant(labels_batches[:, i])
        sess.run([predictions_queue.enqueue(tf_prediction),
                  labels_queue.enqueue(tf_label)])

      tf_predictions = predictions_queue.dequeue()
      tf_labels = labels_queue.dequeue()

      prec, prec_op = metrics.streaming_precision_at_thresholds(
          tf_predictions, tf_labels, thresholds)
      rec, rec_op = metrics.streaming_recall_at_thresholds(
          tf_predictions, tf_labels, thresholds)

      sess.run(tf.local_variables_initializer())
      for _ in range(int(num_samples / batch_size)):
        sess.run([prec_op, rec_op])
      # Since this is only approximate, we can't expect a 6 digits match.
      # Although with higher number of samples/thresholds we should see the
      # accuracy improving
      self.assertAlmostEqual(expected_prec, prec.eval(), 2)
      self.assertAlmostEqual(expected_rec, rec.eval(), 2)


# TODO(ptucker): Remove when we remove `streaming_recall_at_k`.
# This op will be deprecated soon in favor of `streaming_sparse_recall_at_k`.
# Until then, this test validates that both ops yield the same results.
class StreamingRecallAtKTest(tf.test.TestCase):

  def setUp(self):
    np.random.seed(1)
    tf.reset_default_graph()

    self._batch_size = 4
    self._num_classes = 3
    self._np_predictions = np.matrix(('0.1 0.2 0.7;'
                                      '0.6 0.2 0.2;'
                                      '0.0 0.9 0.1;'
                                      '0.2 0.0 0.8'))
    self._np_labels = [0, 0, 0, 0]

  def testVars(self):
    metrics.streaming_recall_at_k(
        predictions=tf.ones((self._batch_size, self._num_classes)),
        labels=tf.ones((self._batch_size,), dtype=tf.int32), k=1)
    _assert_local_variables(self, (
        'recall_at_1/count:0',
        'recall_at_1/total:0'
    ))

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = metrics.streaming_recall_at_k(
        predictions=tf.ones((self._batch_size, self._num_classes)),
        labels=tf.ones((self._batch_size,), dtype=tf.int32),
        k=1,
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [mean])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.streaming_recall_at_k(
        predictions=tf.ones((self._batch_size, self._num_classes)),
        labels=tf.ones((self._batch_size,), dtype=tf.int32),
        k=1,
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testSingleUpdateKIs1(self):
    predictions = tf.constant(self._np_predictions,
                              shape=(self._batch_size, self._num_classes),
                              dtype=tf.float32)
    labels = tf.constant(
        self._np_labels, shape=(self._batch_size,), dtype=tf.int64)
    recall, update_op = metrics.streaming_recall_at_k(
        predictions, labels, k=1)
    sp_recall, sp_update_op = metrics.streaming_sparse_recall_at_k(
        predictions, tf.reshape(labels, (self._batch_size, 1)), k=1)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertEqual(0.25, sess.run(update_op))
      self.assertEqual(0.25, recall.eval())
      self.assertEqual(0.25, sess.run(sp_update_op))
      self.assertEqual(0.25, sp_recall.eval())

  def testSingleUpdateKIs2(self):
    predictions = tf.constant(self._np_predictions,
                              shape=(self._batch_size, self._num_classes),
                              dtype=tf.float32)
    labels = tf.constant(
        self._np_labels, shape=(self._batch_size,), dtype=tf.int64)
    recall, update_op = metrics.streaming_recall_at_k(
        predictions, labels, k=2)
    sp_recall, sp_update_op = metrics.streaming_sparse_recall_at_k(
        predictions, tf.reshape(labels, (self._batch_size, 1)), k=2)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertEqual(0.5, sess.run(update_op))
      self.assertEqual(0.5, recall.eval())
      self.assertEqual(0.5, sess.run(sp_update_op))
      self.assertEqual(0.5, sp_recall.eval())

  def testSingleUpdateKIs3(self):
    predictions = tf.constant(self._np_predictions,
                              shape=(self._batch_size, self._num_classes),
                              dtype=tf.float32)
    labels = tf.constant(
        self._np_labels, shape=(self._batch_size,), dtype=tf.int64)
    recall, update_op = metrics.streaming_recall_at_k(
        predictions, labels, k=3)
    sp_recall, sp_update_op = metrics.streaming_sparse_recall_at_k(
        predictions, tf.reshape(labels, (self._batch_size, 1)), k=3)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertEqual(1.0, sess.run(update_op))
      self.assertEqual(1.0, recall.eval())
      self.assertEqual(1.0, sess.run(sp_update_op))
      self.assertEqual(1.0, sp_recall.eval())

  def testSingleUpdateSomeMissingKIs2(self):
    predictions = tf.constant(self._np_predictions,
                              shape=(self._batch_size, self._num_classes),
                              dtype=tf.float32)
    labels = tf.constant(
        self._np_labels, shape=(self._batch_size,), dtype=tf.int64)
    weights = tf.constant([0, 1, 0, 1], shape=(self._batch_size,),
                          dtype=tf.float32)
    recall, update_op = metrics.streaming_recall_at_k(
        predictions, labels, k=2, weights=weights)
    sp_recall, sp_update_op = metrics.streaming_sparse_recall_at_k(
        predictions, tf.reshape(labels, (self._batch_size, 1)), k=2,
        weights=weights)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertEqual(1.0, sess.run(update_op))
      self.assertEqual(1.0, recall.eval())
      self.assertEqual(1.0, sess.run(sp_update_op))
      self.assertEqual(1.0, sp_recall.eval())


class StreamingSparsePrecisionTest(tf.test.TestCase):

  def _test_streaming_sparse_precision_at_k(self,
                                            predictions,
                                            labels,
                                            k,
                                            expected,
                                            class_id=None,
                                            weights=None):
    with tf.Graph().as_default() as g, self.test_session(g):
      if weights is not None:
        weights = tf.constant(weights, tf.float32)
      metric, update = metrics.streaming_sparse_precision_at_k(
          predictions=tf.constant(predictions, tf.float32), labels=labels,
          k=k, class_id=class_id, weights=weights)

      # Fails without initialized vars.
      self.assertRaises(tf.OpError, metric.eval)
      self.assertRaises(tf.OpError, update.eval)
      tf.initialize_variables(tf.local_variables()).run()

      # Run per-step op and assert expected values.
      if math.isnan(expected):
        _assert_nan(self, update.eval())
        _assert_nan(self, metric.eval())
      else:
        self.assertEqual(expected, update.eval())
        self.assertEqual(expected, metric.eval())

  def _test_streaming_sparse_precision_at_top_k(self,
                                                top_k_predictions,
                                                labels,
                                                expected,
                                                class_id=None,
                                                weights=None):
    with tf.Graph().as_default() as g, self.test_session(g):
      if weights is not None:
        weights = tf.constant(weights, tf.float32)
      metric, update = metrics.streaming_sparse_precision_at_top_k(
          top_k_predictions=tf.constant(top_k_predictions, tf.int32),
          labels=labels, class_id=class_id, weights=weights)

      # Fails without initialized vars.
      self.assertRaises(tf.OpError, metric.eval)
      self.assertRaises(tf.OpError, update.eval)
      tf.initialize_variables(tf.local_variables()).run()

      # Run per-step op and assert expected values.
      if math.isnan(expected):
        self.assertTrue(math.isnan(update.eval()))
        self.assertTrue(math.isnan(metric.eval()))
      else:
        self.assertEqual(expected, update.eval())
        self.assertEqual(expected, metric.eval())

  def _test_sparse_average_precision_at_k(self,
                                          predictions,
                                          labels,
                                          k,
                                          expected):
    with tf.Graph().as_default() as g, self.test_session(g):
      predictions = tf.constant(predictions, tf.float32)
      metric = metric_ops.sparse_average_precision_at_k(
          predictions, labels, k)
      self.assertAllEqual(expected, metric.eval())

  def _test_streaming_sparse_average_precision_at_k(
      self, predictions, labels, k, expected, weights=None):
    with tf.Graph().as_default() as g, self.test_session(g):
      if weights is not None:
        weights = tf.constant(weights, tf.float32)
      predictions = tf.constant(predictions, tf.float32)
      metric, update = metrics.streaming_sparse_average_precision_at_k(
          predictions, labels, k, weights=weights)

      # Fails without initialized vars.
      self.assertRaises(tf.OpError, metric.eval)
      self.assertRaises(tf.OpError, update.eval)
      local_variables = tf.local_variables()
      tf.initialize_variables(local_variables).run()

      # Run per-step op and assert expected values.
      if math.isnan(expected):
        _assert_nan(self, update.eval())
        _assert_nan(self, metric.eval())
      else:
        self.assertAlmostEqual(expected, update.eval())
        self.assertAlmostEqual(expected, metric.eval())

  def test_top_k_rank_invalid(self):
    with self.test_session():
      # top_k_predictions has rank < 2.
      top_k_predictions = [9, 4, 6, 2, 0]
      sp_labels = tf.SparseTensorValue(
          indices=np.array([[0,], [1,], [2,]], np.int64),
          values=np.array([2, 7, 8], np.int64),
          shape=np.array([10,], np.int64))

      with self.assertRaises(ValueError):
        precision, _ = metrics.streaming_sparse_precision_at_top_k(
            top_k_predictions=tf.constant(top_k_predictions, tf.int64),
            labels=sp_labels)
        tf.initialize_variables(tf.local_variables()).run()
        precision.eval()

  def test_average_precision(self):
    # Example 1.
    # Matches example here:
    # fastml.com/what-you-wanted-to-know-about-mean-average-precision
    labels_ex1 = (0, 1, 2, 3, 4)
    labels = np.array([labels_ex1], dtype=np.int64)
    predictions_ex1 = (0.2, 0.1, 0.0, 0.4, 0.0, 0.5, 0.3)
    predictions = (predictions_ex1,)
    predictions_top_k_ex1 = (5, 3, 6, 0, 1, 2)
    precision_ex1 = (
        0.0 / 1,
        1.0 / 2,
        1.0 / 3,
        2.0 / 4
    )
    avg_precision_ex1 = (
        0.0 / 1,
        precision_ex1[1] / 2,
        precision_ex1[1] / 3,
        (precision_ex1[1] + precision_ex1[3]) / 4
    )
    for i in xrange(4):
      k = i + 1
      self._test_streaming_sparse_precision_at_k(
          predictions, labels, k, expected=precision_ex1[i])
      self._test_streaming_sparse_precision_at_top_k(
          (predictions_top_k_ex1[:k],), labels, expected=precision_ex1[i])
      self._test_sparse_average_precision_at_k(
          predictions, labels, k, expected=[avg_precision_ex1[i]])
      self._test_streaming_sparse_average_precision_at_k(
          predictions, labels, k, expected=avg_precision_ex1[i])

    # Example 2.
    labels_ex2 = (0, 2, 4, 5, 6)
    labels = np.array([labels_ex2], dtype=np.int64)
    predictions_ex2 = (0.3, 0.5, 0.0, 0.4, 0.0, 0.1, 0.2)
    predictions = (predictions_ex2,)
    predictions_top_k_ex2 = (1, 3, 0, 6, 5)
    precision_ex2 = (
        0.0 / 1,
        0.0 / 2,
        1.0 / 3,
        2.0 / 4
    )
    avg_precision_ex2 = (
        0.0 / 1,
        0.0 / 2,
        precision_ex2[2] / 3,
        (precision_ex2[2] + precision_ex2[3]) / 4
    )
    for i in xrange(4):
      k = i + 1
      self._test_streaming_sparse_precision_at_k(
          predictions, labels, k, expected=precision_ex2[i])
      self._test_streaming_sparse_precision_at_top_k(
          (predictions_top_k_ex2[:k],), labels, expected=precision_ex2[i])
      self._test_sparse_average_precision_at_k(
          predictions, labels, k, expected=[avg_precision_ex2[i]])
      self._test_streaming_sparse_average_precision_at_k(
          predictions, labels, k, expected=avg_precision_ex2[i])

    # Both examples, we expect both precision and average precision to be the
    # average of the 2 examples.
    labels = np.array([labels_ex1, labels_ex2], dtype=np.int64)
    predictions = (predictions_ex1, predictions_ex2)
    average_precision = [
        (ex1, ex2) for ex1, ex2 in zip(avg_precision_ex1, avg_precision_ex2)]
    streaming_precision = [
        (ex1 + ex2) / 2
        for ex1, ex2 in zip(precision_ex1, precision_ex2)]
    streaming_average_precision = [
        (ex1 + ex2) / 2
        for ex1, ex2 in zip(avg_precision_ex1, avg_precision_ex2)]
    for i in xrange(4):
      k = i + 1
      self._test_streaming_sparse_precision_at_k(
          predictions, labels, k, expected=streaming_precision[i])
      predictions_top_k = (predictions_top_k_ex1[:k], predictions_top_k_ex2[:k])
      self._test_streaming_sparse_precision_at_top_k(
          predictions_top_k, labels, expected=streaming_precision[i])
      self._test_sparse_average_precision_at_k(
          predictions, labels, k, expected=average_precision[i])
      self._test_streaming_sparse_average_precision_at_k(
          predictions, labels, k, expected=streaming_average_precision[i])

    # Weighted examples, we expect streaming average precision to be the
    # weighted average of the 2 examples.
    weights = (0.3, 0.6)
    streaming_average_precision = [
        (weights[0] * ex1 + weights[1] * ex2) / (weights[0] + weights[1])
        for ex1, ex2 in zip(avg_precision_ex1, avg_precision_ex2)]
    for i in xrange(4):
      k = i + 1
      self._test_streaming_sparse_average_precision_at_k(
          predictions, labels, k, expected=streaming_average_precision[i],
          weights=weights)

  def test_average_precision_some_labels_out_of_range(self):
    """Tests that labels outside the [0, n_classes) range are ignored."""
    labels_ex1 = (-1, 0, 1, 2, 3, 4, 7)
    labels = np.array([labels_ex1], dtype=np.int64)
    predictions_ex1 = (0.2, 0.1, 0.0, 0.4, 0.0, 0.5, 0.3)
    predictions = (predictions_ex1,)
    predictions_top_k_ex1 = (5, 3, 6, 0, 1, 2)
    precision_ex1 = (
        0.0 / 1,
        1.0 / 2,
        1.0 / 3,
        2.0 / 4
    )
    avg_precision_ex1 = (
        0.0 / 1,
        precision_ex1[1] / 2,
        precision_ex1[1] / 3,
        (precision_ex1[1] + precision_ex1[3]) / 4
    )
    for i in xrange(4):
      k = i + 1
      self._test_streaming_sparse_precision_at_k(
          predictions, labels, k, expected=precision_ex1[i])
      self._test_streaming_sparse_precision_at_top_k(
          (predictions_top_k_ex1[:k],), labels, expected=precision_ex1[i])
      self._test_sparse_average_precision_at_k(
          predictions, labels, k, expected=[avg_precision_ex1[i]])
      self._test_streaming_sparse_average_precision_at_k(
          predictions, labels, k, expected=avg_precision_ex1[i])

  def test_one_label_at_k1_nan(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    top_k_predictions = [[3], [3]]
    sparse_labels = _binary_2d_label_to_sparse_value(
        [[0, 0, 0, 1], [0, 0, 1, 0]])
    dense_labels = np.array([[3], [2]], dtype=np.int64)

    for labels in (sparse_labels, dense_labels):
      # Classes 0,1,2 have 0 predictions, classes -1 and 4 are out of range.
      for class_id in (-1, 0, 1, 2, 4):
        self._test_streaming_sparse_precision_at_k(
            predictions, labels, k=1, expected=NAN, class_id=class_id)
        self._test_streaming_sparse_precision_at_top_k(
            top_k_predictions, labels, expected=NAN, class_id=class_id)

  def test_one_label_at_k1(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    top_k_predictions = [[3], [3]]
    sparse_labels = _binary_2d_label_to_sparse_value(
        [[0, 0, 0, 1], [0, 0, 1, 0]])
    dense_labels = np.array([[3], [2]], dtype=np.int64)

    for labels in (sparse_labels, dense_labels):
      # Class 3: 1 label, 2 predictions, 1 correct.
      self._test_streaming_sparse_precision_at_k(
          predictions, labels, k=1, expected=1.0 / 2, class_id=3)
      self._test_streaming_sparse_precision_at_top_k(
          top_k_predictions, labels, expected=1.0 / 2, class_id=3)

      # All classes: 2 labels, 2 predictions, 1 correct.
      self._test_streaming_sparse_precision_at_k(
          predictions, labels, k=1, expected=1.0 / 2)
      self._test_streaming_sparse_precision_at_top_k(
          top_k_predictions, labels, expected=1.0 / 2)

  def test_three_labels_at_k5_no_predictions(self):
    predictions = [
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]
    ]
    top_k_predictions = [
        [9, 4, 6, 2, 0],
        [5, 7, 2, 9, 6],
    ]
    sparse_labels = _binary_2d_label_to_sparse_value([
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    ])
    dense_labels = np.array([[2, 7, 8], [1, 2, 5]], dtype=np.int64)

    for labels in (sparse_labels, dense_labels):
      # Classes 1,3,8 have 0 predictions, classes -1 and 10 are out of range.
      for class_id in (-1, 1, 3, 8, 10):
        self._test_streaming_sparse_precision_at_k(
            predictions, labels, k=5, expected=NAN, class_id=class_id)
        self._test_streaming_sparse_precision_at_top_k(
            top_k_predictions, labels, expected=NAN, class_id=class_id)

  def test_three_labels_at_k5_no_labels(self):
    predictions = [
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]
    ]
    top_k_predictions = [
        [9, 4, 6, 2, 0],
        [5, 7, 2, 9, 6],
    ]
    sparse_labels = _binary_2d_label_to_sparse_value([
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    ])
    dense_labels = np.array([[2, 7, 8], [1, 2, 5]], dtype=np.int64)

    for labels in (sparse_labels, dense_labels):
      # Classes 0,4,6,9: 0 labels, >=1 prediction.
      for class_id in (0, 4, 6, 9):
        self._test_streaming_sparse_precision_at_k(
            predictions, labels, k=5, expected=0.0, class_id=class_id)
        self._test_streaming_sparse_precision_at_top_k(
            top_k_predictions, labels, expected=0.0, class_id=class_id)

  def test_three_labels_at_k5(self):
    predictions = [
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]
    ]
    top_k_predictions = [
        [9, 4, 6, 2, 0],
        [5, 7, 2, 9, 6],
    ]
    sparse_labels = _binary_2d_label_to_sparse_value([
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    ])
    dense_labels = np.array([[2, 7, 8], [1, 2, 5]], dtype=np.int64)

    for labels in (sparse_labels, dense_labels):
      # Class 2: 2 labels, 2 correct predictions.
      self._test_streaming_sparse_precision_at_k(
          predictions, labels, k=5, expected=2.0 / 2,
          class_id=2)
      self._test_streaming_sparse_precision_at_top_k(
          top_k_predictions, labels, expected=2.0 / 2, class_id=2)

      # Class 5: 1 label, 1 correct prediction.
      self._test_streaming_sparse_precision_at_k(
          predictions, labels, k=5, expected=1.0 / 1, class_id=5)
      self._test_streaming_sparse_precision_at_top_k(
          top_k_predictions, labels, expected=1.0 / 1, class_id=5)

      # Class 7: 1 label, 1 incorrect prediction.
      self._test_streaming_sparse_precision_at_k(
          predictions, labels, k=5, expected=0.0 / 1, class_id=7)
      self._test_streaming_sparse_precision_at_top_k(
          top_k_predictions, labels, expected=0.0 / 1, class_id=7)

      # All classes: 10 predictions, 3 correct.
      self._test_streaming_sparse_precision_at_k(
          predictions, labels, k=5, expected=3.0 / 10)
      self._test_streaming_sparse_precision_at_top_k(
          top_k_predictions, labels, expected=3.0 / 10)

  def test_three_labels_at_k5_some_out_of_range(self):
    """Tests that labels outside the [0, n_classes) range are ignored."""
    predictions = [
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]
    ]
    top_k_predictions = [
        [9, 4, 6, 2, 0],
        [5, 7, 2, 9, 6],
    ]
    sp_labels = tf.SparseTensorValue(
        indices=[[0, 0], [0, 1], [0, 2], [0, 3],
                 [1, 0], [1, 1], [1, 2], [1, 3]],
        # values -1 and 10 are outside the [0, n_classes) range and are ignored.
        values=np.array([2, 7, -1, 8,
                         1, 2, 5, 10], np.int64),
        shape=[2, 4])

    # Class 2: 2 labels, 2 correct predictions.
    self._test_streaming_sparse_precision_at_k(
        predictions, sp_labels, k=5, expected=2.0 / 2, class_id=2)
    self._test_streaming_sparse_precision_at_top_k(
        top_k_predictions, sp_labels, expected=2.0 / 2, class_id=2)

    # Class 5: 1 label, 1 correct prediction.
    self._test_streaming_sparse_precision_at_k(
        predictions, sp_labels, k=5, expected=1.0 / 1, class_id=5)
    self._test_streaming_sparse_precision_at_top_k(
        top_k_predictions, sp_labels, expected=1.0 / 1, class_id=5)

    # Class 7: 1 label, 1 incorrect prediction.
    self._test_streaming_sparse_precision_at_k(
        predictions, sp_labels, k=5, expected=0.0 / 1, class_id=7)
    self._test_streaming_sparse_precision_at_top_k(
        top_k_predictions, sp_labels, expected=0.0 / 1, class_id=7)

    # All classes: 10 predictions, 3 correct.
    self._test_streaming_sparse_precision_at_k(
        predictions, sp_labels, k=5, expected=3.0 / 10)
    self._test_streaming_sparse_precision_at_top_k(
        top_k_predictions, sp_labels, expected=3.0 / 10)

  def test_3d_nan(self):
    predictions = [[
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]
    ], [
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6],
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9]
    ]]
    top_k_predictions = [[
        [9, 4, 6, 2, 0],
        [5, 7, 2, 9, 6],
    ], [
        [5, 7, 2, 9, 6],
        [9, 4, 6, 2, 0],
    ]]
    labels = _binary_3d_label_to_sparse_value([[
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    ], [
        [0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
    ]])

    # Classes 1,3,8 have 0 predictions, classes -1 and 10 are out of range.
    for class_id in (-1, 1, 3, 8, 10):
      self._test_streaming_sparse_precision_at_k(
          predictions, labels, k=5, expected=NAN, class_id=class_id)
      self._test_streaming_sparse_precision_at_top_k(
          top_k_predictions, labels, expected=NAN, class_id=class_id)

  def test_3d_no_labels(self):
    predictions = [[
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]
    ], [
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6],
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9]
    ]]
    top_k_predictions = [[
        [9, 4, 6, 2, 0],
        [5, 7, 2, 9, 6],
    ], [
        [5, 7, 2, 9, 6],
        [9, 4, 6, 2, 0],
    ]]
    labels = _binary_3d_label_to_sparse_value([[
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    ], [
        [0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
    ]])

    # Classes 0,4,6,9: 0 labels, >=1 prediction.
    for class_id in (0, 4, 6, 9):
      self._test_streaming_sparse_precision_at_k(
          predictions, labels, k=5, expected=0.0, class_id=class_id)
      self._test_streaming_sparse_precision_at_top_k(
          top_k_predictions, labels, expected=0.0, class_id=class_id)

  def test_3d(self):
    predictions = [[
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]
    ], [
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6],
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9]
    ]]
    top_k_predictions = [[
        [9, 4, 6, 2, 0],
        [5, 7, 2, 9, 6],
    ], [
        [5, 7, 2, 9, 6],
        [9, 4, 6, 2, 0],
    ]]
    labels = _binary_3d_label_to_sparse_value([[
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    ], [
        [0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
    ]])

    # Class 2: 4 predictions, all correct.
    self._test_streaming_sparse_precision_at_k(
        predictions, labels, k=5, expected=4.0 / 4, class_id=2)
    self._test_streaming_sparse_precision_at_top_k(
        top_k_predictions, labels, expected=4.0 / 4, class_id=2)

    # Class 5: 2 predictions, both correct.
    self._test_streaming_sparse_precision_at_k(
        predictions, labels, k=5, expected=2.0 / 2, class_id=5)
    self._test_streaming_sparse_precision_at_top_k(
        top_k_predictions, labels, expected=2.0 / 2, class_id=5)

    # Class 7: 2 predictions, 1 correct.
    self._test_streaming_sparse_precision_at_k(
        predictions, labels, k=5, expected=1.0 / 2, class_id=7)
    self._test_streaming_sparse_precision_at_top_k(
        top_k_predictions, labels, expected=1.0 / 2, class_id=7)

    # All classes: 20 predictions, 7 correct.
    self._test_streaming_sparse_precision_at_k(
        predictions, labels, k=5, expected=7.0 / 20)
    self._test_streaming_sparse_precision_at_top_k(
        top_k_predictions, labels, expected=7.0 / 20)

  def test_3d_ignore_all(self):
    predictions = [[
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]
    ], [
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6],
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9]
    ]]
    top_k_predictions = [[
        [9, 4, 6, 2, 0],
        [5, 7, 2, 9, 6],
    ], [
        [5, 7, 2, 9, 6],
        [9, 4, 6, 2, 0],
    ]]
    labels = _binary_3d_label_to_sparse_value([[
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    ], [
        [0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
    ]])

    for class_id in xrange(10):
      self._test_streaming_sparse_precision_at_k(
          predictions, labels, k=5, expected=NAN, class_id=class_id,
          weights=[[0], [0]])
      self._test_streaming_sparse_precision_at_top_k(
          top_k_predictions, labels, expected=NAN, class_id=class_id,
          weights=[[0], [0]])
      self._test_streaming_sparse_precision_at_k(
          predictions, labels, k=5, expected=NAN, class_id=class_id,
          weights=[[0, 0], [0, 0]])
      self._test_streaming_sparse_precision_at_top_k(
          top_k_predictions, labels, expected=NAN, class_id=class_id,
          weights=[[0, 0], [0, 0]])
    self._test_streaming_sparse_precision_at_k(
        predictions, labels, k=5, expected=NAN, weights=[[0], [0]])
    self._test_streaming_sparse_precision_at_top_k(
        top_k_predictions, labels, expected=NAN, weights=[[0], [0]])
    self._test_streaming_sparse_precision_at_k(
        predictions, labels, k=5, expected=NAN, weights=[[0, 0], [0, 0]])
    self._test_streaming_sparse_precision_at_top_k(
        top_k_predictions, labels, expected=NAN,
        weights=[[0, 0], [0, 0]])

  def test_3d_ignore_some(self):
    predictions = [[
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]
    ], [
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6],
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9]
    ]]
    top_k_predictions = [[
        [9, 4, 6, 2, 0],
        [5, 7, 2, 9, 6],
    ], [
        [5, 7, 2, 9, 6],
        [9, 4, 6, 2, 0],
    ]]
    labels = _binary_3d_label_to_sparse_value([[
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    ], [
        [0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
    ]])

    # Class 2: 2 predictions, both correct.
    self._test_streaming_sparse_precision_at_k(
        predictions, labels, k=5, expected=2.0 / 2.0, class_id=2,
        weights=[[1], [0]])
    self._test_streaming_sparse_precision_at_top_k(
        top_k_predictions, labels, expected=2.0 / 2.0, class_id=2,
        weights=[[1], [0]])

    # Class 2: 2 predictions, both correct.
    self._test_streaming_sparse_precision_at_k(
        predictions, labels, k=5, expected=2.0 / 2.0, class_id=2,
        weights=[[0], [1]])
    self._test_streaming_sparse_precision_at_top_k(
        top_k_predictions, labels, expected=2.0 / 2.0, class_id=2,
        weights=[[0], [1]])

    # Class 7: 1 incorrect prediction.
    self._test_streaming_sparse_precision_at_k(
        predictions, labels, k=5, expected=0.0 / 1.0, class_id=7,
        weights=[[1], [0]])
    self._test_streaming_sparse_precision_at_top_k(
        top_k_predictions, labels, expected=0.0 / 1.0, class_id=7,
        weights=[[1], [0]])

    # Class 7: 1 correct prediction.
    self._test_streaming_sparse_precision_at_k(
        predictions, labels, k=5, expected=1.0 / 1.0, class_id=7,
        weights=[[0], [1]])
    self._test_streaming_sparse_precision_at_top_k(
        top_k_predictions, labels, expected=1.0 / 1.0, class_id=7,
        weights=[[0], [1]])

    # Class 7: no predictions.
    self._test_streaming_sparse_precision_at_k(
        predictions, labels, k=5, expected=NAN, class_id=7,
        weights=[[1, 0], [0, 1]])
    self._test_streaming_sparse_precision_at_top_k(
        top_k_predictions, labels, expected=NAN, class_id=7,
        weights=[[1, 0], [0, 1]])

    # Class 7: 2 predictions, 1 correct.
    self._test_streaming_sparse_precision_at_k(
        predictions, labels, k=5, expected=1.0 / 2.0, class_id=7,
        weights=[[0, 1], [1, 0]])
    self._test_streaming_sparse_precision_at_top_k(
        top_k_predictions, labels, expected=1.0 / 2.0, class_id=7,
        weights=[[0, 1], [1, 0]])

  def test_sparse_tensor_value(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    labels = [[0, 0, 0, 1], [0, 0, 1, 0]]
    expected_precision = 0.5
    with self.test_session():
      _, precision = metrics.streaming_sparse_precision_at_k(
          predictions=tf.constant(predictions, tf.float32),
          labels=_binary_2d_label_to_sparse_value(labels), k=1)

      tf.initialize_variables(tf.local_variables()).run()

      self.assertEqual(expected_precision, precision.eval())


class StreamingSparseRecallTest(tf.test.TestCase):

  def _test_streaming_sparse_recall_at_k(self,
                                         predictions,
                                         labels,
                                         k,
                                         expected,
                                         class_id=None,
                                         weights=None):
    with tf.Graph().as_default() as g, self.test_session(g):
      if weights is not None:
        weights = tf.constant(weights, tf.float32)
      metric, update = metrics.streaming_sparse_recall_at_k(
          predictions=tf.constant(predictions, tf.float32),
          labels=labels, k=k, class_id=class_id, weights=weights)

      # Fails without initialized vars.
      self.assertRaises(tf.OpError, metric.eval)
      self.assertRaises(tf.OpError, update.eval)
      tf.initialize_variables(tf.local_variables()).run()

      # Run per-step op and assert expected values.
      if math.isnan(expected):
        _assert_nan(self, update.eval())
        _assert_nan(self, metric.eval())
      else:
        self.assertEqual(expected, update.eval())
        self.assertEqual(expected, metric.eval())

  def test_one_label_at_k1_nan(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    sparse_labels = _binary_2d_label_to_sparse_value(
        [[0, 0, 0, 1], [0, 0, 1, 0]])
    dense_labels = np.array([[3], [2]], dtype=np.int64)

    # Classes 0,1 have 0 labels, 0 predictions, classes -1 and 4 are out of
    # range.
    for labels in (sparse_labels, dense_labels):
      for class_id in (-1, 0, 1, 4):
        self._test_streaming_sparse_recall_at_k(
            predictions, labels, k=1, expected=NAN,
            class_id=class_id)

  def test_one_label_at_k1_no_predictions(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    sparse_labels = _binary_2d_label_to_sparse_value(
        [[0, 0, 0, 1], [0, 0, 1, 0]])
    dense_labels = np.array([[3], [2]], dtype=np.int64)

    for labels in (sparse_labels, dense_labels):
      # Class 2: 0 predictions.
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=1, expected=0.0,
          class_id=2)

  def test_one_label_at_k1(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    sparse_labels = _binary_2d_label_to_sparse_value(
        [[0, 0, 0, 1], [0, 0, 1, 0]])
    dense_labels = np.array([[3], [2]], dtype=np.int64)

    for labels in (sparse_labels, dense_labels):
      # Class 3: 1 label, 2 predictions, 1 correct.
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=1, expected=1.0 / 1,
          class_id=3)

      # All classes: 2 labels, 2 predictions, 1 correct.
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=1, expected=1.0 / 2)

  def test_one_label_at_k1_weighted(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    sparse_labels = _binary_2d_label_to_sparse_value(
        [[0, 0, 0, 1], [0, 0, 1, 0]])
    dense_labels = np.array([[3], [2]], dtype=np.int64)

    for labels in (sparse_labels, dense_labels):
      # Class 3: 1 label, 2 predictions, 1 correct.
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=1, expected=NAN, class_id=3, weights=(0.0,))
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=1, expected=1.0 / 1, class_id=3,
          weights=(1.0,))
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=1, expected=1.0 / 1, class_id=3,
          weights=(2.0,))
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=1, expected=NAN, class_id=3,
          weights=(0.0, 0.0))
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=1, expected=NAN, class_id=3,
          weights=(0.0, 1.0))
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=1, expected=1.0 / 1, class_id=3,
          weights=(1.0, 0.0))
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=1, expected=1.0 / 1, class_id=3,
          weights=(1.0, 1.0))
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=1, expected=2.0 / 2, class_id=3,
          weights=(2.0, 3.0))
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=1, expected=3.0 / 3, class_id=3,
          weights=(3.0, 2.0))
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=1, expected=0.3 / 0.3, class_id=3,
          weights=(0.3, 0.6))
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=1, expected=0.6 / 0.6, class_id=3,
          weights=(0.6, 0.3))

      # All classes: 2 labels, 2 predictions, 1 correct.
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=1, expected=NAN, weights=(0.0,))
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=1, expected=1.0 / 2, weights=(1.0,))
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=1, expected=1.0 / 2, weights=(2.0,))
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=1, expected=1.0 / 1, weights=(1.0, 0.0))
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=1, expected=0.0 / 1, weights=(0.0, 1.0))
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=1, expected=1.0 / 2, weights=(1.0, 1.0))
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=1, expected=2.0 / 5, weights=(2.0, 3.0))
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=1, expected=3.0 / 5, weights=(3.0, 2.0))
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=1, expected=0.3 / 0.9, weights=(0.3, 0.6))
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=1, expected=0.6 / 0.9, weights=(0.6, 0.3))

  def test_three_labels_at_k5_nan(self):
    predictions = [
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]]
    sparse_labels = _binary_2d_label_to_sparse_value([
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]])
    dense_labels = np.array([[2, 7, 8], [1, 2, 5]], dtype=np.int64)

    for labels in (sparse_labels, dense_labels):
      # Classes 0,3,4,6,9 have 0 labels, class 10 is out of range.
      for class_id in (0, 3, 4, 6, 9, 10):
        self._test_streaming_sparse_recall_at_k(
            predictions, labels, k=5, expected=NAN, class_id=class_id)

  def test_three_labels_at_k5_no_predictions(self):
    predictions = [
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]]
    sparse_labels = _binary_2d_label_to_sparse_value([
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]])
    dense_labels = np.array([[2, 7, 8], [1, 2, 5]], dtype=np.int64)

    for labels in (sparse_labels, dense_labels):
      # Class 8: 1 label, no predictions.
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=5, expected=0.0 / 1, class_id=8)

  def test_three_labels_at_k5(self):
    predictions = [
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]]
    sparse_labels = _binary_2d_label_to_sparse_value([
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]])
    dense_labels = np.array([[2, 7, 8], [1, 2, 5]], dtype=np.int64)

    for labels in (sparse_labels, dense_labels):
      # Class 2: 2 labels, both correct.
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=5, expected=2.0 / 2, class_id=2)

      # Class 5: 1 label, incorrect.
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=5, expected=1.0 / 1, class_id=5)

      # Class 7: 1 label, incorrect.
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=5, expected=0.0 / 1, class_id=7)

      # All classes: 6 labels, 3 correct.
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=5, expected=3.0 / 6)

  def test_three_labels_at_k5_some_out_of_range(self):
    """Tests that labels outside the [0, n_classes) count in denominator."""
    predictions = [
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]]
    sp_labels = tf.SparseTensorValue(
        indices=[[0, 0], [0, 1], [0, 2], [0, 3],
                 [1, 0], [1, 1], [1, 2], [1, 3]],
        # values -1 and 10 are outside the [0, n_classes) range.
        values=np.array([2, 7, -1, 8,
                         1, 2, 5, 10], np.int64),
        shape=[2, 4])

    # Class 2: 2 labels, both correct.
    self._test_streaming_sparse_recall_at_k(
        predictions=predictions, labels=sp_labels, k=5, expected=2.0 / 2,
        class_id=2)

    # Class 5: 1 label, incorrect.
    self._test_streaming_sparse_recall_at_k(
        predictions=predictions, labels=sp_labels, k=5, expected=1.0 / 1,
        class_id=5)

    # Class 7: 1 label, incorrect.
    self._test_streaming_sparse_recall_at_k(
        predictions=predictions, labels=sp_labels, k=5, expected=0.0 / 1,
        class_id=7)

    # All classes: 8 labels, 3 correct.
    self._test_streaming_sparse_recall_at_k(
        predictions=predictions, labels=sp_labels, k=5, expected=3.0 / 8)

  def test_3d_nan(self):
    predictions = [[
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]
    ], [
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6],
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9]
    ]]
    sparse_labels = _binary_3d_label_to_sparse_value([[
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    ], [
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0]
    ]])
    dense_labels = np.array([[
        [2, 7, 8],
        [1, 2, 5]
    ], [
        [1, 2, 5],
        [2, 7, 8],
    ]], dtype=np.int64)

    for labels in (sparse_labels, dense_labels):
      # Classes 0,3,4,6,9 have 0 labels, class 10 is out of range.
      for class_id in (0, 3, 4, 6, 9, 10):
        self._test_streaming_sparse_recall_at_k(
            predictions, labels, k=5, expected=NAN, class_id=class_id)

  def test_3d_no_predictions(self):
    predictions = [[
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]
    ], [
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6],
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9]
    ]]
    sparse_labels = _binary_3d_label_to_sparse_value([[
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    ], [
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0]
    ]])
    dense_labels = np.array([[
        [2, 7, 8],
        [1, 2, 5]
    ], [
        [1, 2, 5],
        [2, 7, 8],
    ]], dtype=np.int64)

    for labels in (sparse_labels, dense_labels):
      # Classes 1,8 have 0 predictions, >=1 label.
      for class_id in (1, 8):
        self._test_streaming_sparse_recall_at_k(
            predictions, labels, k=5, expected=0.0, class_id=class_id)

  def test_3d(self):
    predictions = [[
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]
    ], [
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6],
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9]
    ]]
    labels = _binary_3d_label_to_sparse_value([[
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    ], [
        [0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
    ]])

    # Class 2: 4 labels, all correct.
    self._test_streaming_sparse_recall_at_k(
        predictions, labels, k=5, expected=4.0 / 4, class_id=2)

    # Class 5: 2 labels, both correct.
    self._test_streaming_sparse_recall_at_k(
        predictions, labels, k=5, expected=2.0 / 2, class_id=5)

    # Class 7: 2 labels, 1 incorrect.
    self._test_streaming_sparse_recall_at_k(
        predictions, labels, k=5, expected=1.0 / 2, class_id=7)

    # All classes: 12 labels, 7 correct.
    self._test_streaming_sparse_recall_at_k(
        predictions, labels, k=5, expected=7.0 / 12)

  def test_3d_ignore_all(self):
    predictions = [[
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]
    ], [
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6],
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9]
    ]]
    labels = _binary_3d_label_to_sparse_value([[
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    ], [
        [0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
    ]])

    for class_id in xrange(10):
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=5, expected=NAN, class_id=class_id,
          weights=[[0], [0]])
      self._test_streaming_sparse_recall_at_k(
          predictions, labels, k=5, expected=NAN, class_id=class_id,
          weights=[[0, 0], [0, 0]])
    self._test_streaming_sparse_recall_at_k(
        predictions, labels, k=5, expected=NAN, weights=[[0], [0]])
    self._test_streaming_sparse_recall_at_k(
        predictions, labels, k=5, expected=NAN, weights=[[0, 0], [0, 0]])

  def test_3d_ignore_some(self):
    predictions = [[
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9],
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6]
    ], [
        [0.3, 0.0, 0.7, 0.2, 0.4, 0.9, 0.5, 0.8, 0.1, 0.6],
        [0.5, 0.1, 0.6, 0.3, 0.8, 0.0, 0.7, 0.2, 0.4, 0.9]
    ]]
    labels = _binary_3d_label_to_sparse_value([[
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    ], [
        [0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
    ]])

    # Class 2: 2 labels, both correct.
    self._test_streaming_sparse_recall_at_k(
        predictions, labels, k=5, expected=2.0 / 2.0, class_id=2,
        weights=[[1], [0]])

    # Class 2: 2 labels, both correct.
    self._test_streaming_sparse_recall_at_k(
        predictions, labels, k=5, expected=2.0 / 2.0, class_id=2,
        weights=[[0], [1]])

    # Class 7: 1 label, correct.
    self._test_streaming_sparse_recall_at_k(
        predictions, labels, k=5, expected=1.0 / 1.0, class_id=7,
        weights=[[0], [1]])

    # Class 7: 1 label, incorrect.
    self._test_streaming_sparse_recall_at_k(
        predictions, labels, k=5, expected=0.0 / 1.0, class_id=7,
        weights=[[1], [0]])

    # Class 7: 2 labels, 1 correct.
    self._test_streaming_sparse_recall_at_k(
        predictions, labels, k=5, expected=1.0 / 2.0, class_id=7,
        weights=[[1, 0], [1, 0]])

    # Class 7: No labels.
    self._test_streaming_sparse_recall_at_k(
        predictions, labels, k=5, expected=NAN, class_id=7,
        weights=[[0, 1], [0, 1]])

  def test_sparse_tensor_value(self):
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    labels = [[0, 0, 1, 0], [0, 0, 0, 1]]
    expected_recall = 0.5
    with self.test_session():
      _, recall = metrics.streaming_sparse_recall_at_k(
          predictions=tf.constant(predictions, tf.float32),
          labels=_binary_2d_label_to_sparse_value(labels), k=1)

      tf.initialize_variables(tf.local_variables()).run()

      self.assertEqual(expected_recall, recall.eval())


class StreamingMeanAbsoluteErrorTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def testVars(self):
    metrics.streaming_mean_absolute_error(
        predictions=tf.ones((10, 1)), labels=tf.ones((10, 1)))
    _assert_local_variables(self, (
        'mean_absolute_error/count:0',
        'mean_absolute_error/total:0'
    ))

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = metrics.streaming_mean_absolute_error(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [mean])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.streaming_mean_absolute_error(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testValueTensorIsIdempotent(self):
    predictions = tf.random_normal((10, 3), seed=1)
    labels = tf.random_normal((10, 3), seed=2)
    error, update_op = metrics.streaming_mean_absolute_error(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        sess.run(update_op)

      # Then verify idempotency.
      initial_error = error.eval()
      for _ in range(10):
        self.assertEqual(initial_error, error.eval())

  def testSingleUpdateWithErrorAndWeights(self):
    predictions = tf.constant([2, 4, 6, 8], shape=(1, 4), dtype=tf.float32)
    labels = tf.constant([1, 3, 2, 3], shape=(1, 4), dtype=tf.float32)
    weights = tf.constant([0, 1, 0, 1], shape=(1, 4))

    error, update_op = metrics.streaming_mean_absolute_error(
        predictions, labels, weights)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertEqual(3, sess.run(update_op))
      self.assertEqual(3, error.eval())


class StreamingMeanRelativeErrorTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def testVars(self):
    metrics.streaming_mean_relative_error(
        predictions=tf.ones((10, 1)), labels=tf.ones((10, 1)),
        normalizer=tf.ones((10, 1)))
    _assert_local_variables(self, (
        'mean_relative_error/count:0',
        'mean_relative_error/total:0'
    ))

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = metrics.streaming_mean_relative_error(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        normalizer=tf.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(
        tf.get_collection(my_collection_name), [mean])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.streaming_mean_relative_error(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        normalizer=tf.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testValueTensorIsIdempotent(self):
    predictions = tf.random_normal((10, 3), seed=1)
    labels = tf.random_normal((10, 3), seed=2)
    normalizer = tf.random_normal((10, 3), seed=3)
    error, update_op = metrics.streaming_mean_relative_error(
        predictions, labels, normalizer)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        sess.run(update_op)

      # Then verify idempotency.
      initial_error = error.eval()
      for _ in range(10):
        self.assertEqual(initial_error, error.eval())

  def testSingleUpdateNormalizedByLabels(self):
    np_predictions = np.asarray([2, 4, 6, 8], dtype=np.float32)
    np_labels = np.asarray([1, 3, 2, 3], dtype=np.float32)
    expected_error = np.mean(
        np.divide(np.absolute(np_predictions - np_labels),
                  np_labels))

    predictions = tf.constant(np_predictions, shape=(1, 4), dtype=tf.float32)
    labels = tf.constant(np_labels, shape=(1, 4))

    error, update_op = metrics.streaming_mean_relative_error(
        predictions, labels, normalizer=labels)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertEqual(expected_error, sess.run(update_op))
      self.assertEqual(expected_error, error.eval())

  def testSingleUpdateNormalizedByZeros(self):
    np_predictions = np.asarray([2, 4, 6, 8], dtype=np.float32)

    predictions = tf.constant(np_predictions, shape=(1, 4), dtype=tf.float32)
    labels = tf.constant([1, 3, 2, 3], shape=(1, 4), dtype=tf.float32)

    error, update_op = metrics.streaming_mean_relative_error(
        predictions, labels, normalizer=tf.zeros_like(labels))

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertEqual(0.0, sess.run(update_op))
      self.assertEqual(0.0, error.eval())


class StreamingMeanSquaredErrorTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def testVars(self):
    metrics.streaming_mean_squared_error(
        predictions=tf.ones((10, 1)), labels=tf.ones((10, 1)))
    _assert_local_variables(self, (
        'mean_squared_error/count:0',
        'mean_squared_error/total:0'
    ))

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = metrics.streaming_mean_squared_error(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [mean])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.streaming_mean_squared_error(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testValueTensorIsIdempotent(self):
    predictions = tf.random_normal((10, 3), seed=1)
    labels = tf.random_normal((10, 3), seed=2)
    error, update_op = metrics.streaming_mean_squared_error(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        sess.run(update_op)

      # Then verify idempotency.
      initial_error = error.eval()
      for _ in range(10):
        self.assertEqual(initial_error, error.eval())

  def testSingleUpdateZeroError(self):
    predictions = tf.zeros((1, 3), dtype=tf.float32)
    labels = tf.zeros((1, 3), dtype=tf.float32)

    error, update_op = metrics.streaming_mean_squared_error(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertEqual(0, sess.run(update_op))
      self.assertEqual(0, error.eval())

  def testSingleUpdateWithError(self):
    predictions = tf.constant([2, 4, 6], shape=(1, 3), dtype=tf.float32)
    labels = tf.constant([1, 3, 2], shape=(1, 3), dtype=tf.float32)

    error, update_op = metrics.streaming_mean_squared_error(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertEqual(6, sess.run(update_op))
      self.assertEqual(6, error.eval())

  def testSingleUpdateWithErrorAndWeights(self):
    predictions = tf.constant([2, 4, 6, 8], shape=(1, 4), dtype=tf.float32)
    labels = tf.constant([1, 3, 2, 3], shape=(1, 4), dtype=tf.float32)
    weights = tf.constant([0, 1, 0, 1], shape=(1, 4))

    error, update_op = metrics.streaming_mean_squared_error(
        predictions, labels, weights)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertEqual(13, sess.run(update_op))
      self.assertEqual(13, error.eval())

  def testMultipleBatchesOfSizeOne(self):
    with self.test_session() as sess:
      # Create the queue that populates the predictions.
      preds_queue = tf.FIFOQueue(2, dtypes=tf.float32, shapes=(1, 3))
      _enqueue_vector(sess, preds_queue, [10, 8, 6])
      _enqueue_vector(sess, preds_queue, [-4, 3, -1])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      labels_queue = tf.FIFOQueue(2, dtypes=tf.float32, shapes=(1, 3))
      _enqueue_vector(sess, labels_queue, [1, 3, 2])
      _enqueue_vector(sess, labels_queue, [2, 4, 6])
      labels = labels_queue.dequeue()

      error, update_op = metrics.streaming_mean_squared_error(
          predictions, labels)

      sess.run(tf.local_variables_initializer())
      sess.run(update_op)
      self.assertAlmostEqual(208.0 / 6, sess.run(update_op), 5)

      self.assertAlmostEqual(208.0 / 6, error.eval(), 5)

  def testMetricsComputedConcurrently(self):
    with self.test_session() as sess:
      # Create the queue that populates one set of predictions.
      preds_queue0 = tf.FIFOQueue(2, dtypes=tf.float32, shapes=(1, 3))
      _enqueue_vector(sess, preds_queue0, [10, 8, 6])
      _enqueue_vector(sess, preds_queue0, [-4, 3, -1])
      predictions0 = preds_queue0.dequeue()

      # Create the queue that populates one set of predictions.
      preds_queue1 = tf.FIFOQueue(2, dtypes=tf.float32, shapes=(1, 3))
      _enqueue_vector(sess, preds_queue1, [0, 1, 1])
      _enqueue_vector(sess, preds_queue1, [1, 1, 0])
      predictions1 = preds_queue1.dequeue()

      # Create the queue that populates one set of labels.
      labels_queue0 = tf.FIFOQueue(2, dtypes=tf.float32, shapes=(1, 3))
      _enqueue_vector(sess, labels_queue0, [1, 3, 2])
      _enqueue_vector(sess, labels_queue0, [2, 4, 6])
      labels0 = labels_queue0.dequeue()

      # Create the queue that populates another set of labels.
      labels_queue1 = tf.FIFOQueue(2, dtypes=tf.float32, shapes=(1, 3))
      _enqueue_vector(sess, labels_queue1, [-5, -3, -1])
      _enqueue_vector(sess, labels_queue1, [5, 4, 3])
      labels1 = labels_queue1.dequeue()

      mse0, update_op0 = metrics.streaming_mean_squared_error(
          predictions0, labels0, name='msd0')
      mse1, update_op1 = metrics.streaming_mean_squared_error(
          predictions1, labels1, name='msd1')

      sess.run(tf.local_variables_initializer())
      sess.run([update_op0, update_op1])
      sess.run([update_op0, update_op1])

      mse0, mse1 = sess.run([mse0, mse1])
      self.assertAlmostEqual(208.0 / 6, mse0, 5)
      self.assertAlmostEqual(79.0 / 6, mse1, 5)

  def testMultipleMetricsOnMultipleBatchesOfSizeOne(self):
    with self.test_session() as sess:
      # Create the queue that populates the predictions.
      preds_queue = tf.FIFOQueue(2, dtypes=tf.float32, shapes=(1, 3))
      _enqueue_vector(sess, preds_queue, [10, 8, 6])
      _enqueue_vector(sess, preds_queue, [-4, 3, -1])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      labels_queue = tf.FIFOQueue(2, dtypes=tf.float32, shapes=(1, 3))
      _enqueue_vector(sess, labels_queue, [1, 3, 2])
      _enqueue_vector(sess, labels_queue, [2, 4, 6])
      labels = labels_queue.dequeue()

      mae, ma_update_op = metrics.streaming_mean_absolute_error(
          predictions, labels)
      mse, ms_update_op = metrics.streaming_mean_squared_error(
          predictions, labels)

      sess.run(tf.local_variables_initializer())
      sess.run([ma_update_op, ms_update_op])
      sess.run([ma_update_op, ms_update_op])

      self.assertAlmostEqual(32.0 / 6, mae.eval(), 5)
      self.assertAlmostEqual(208.0 / 6, mse.eval(), 5)


class StreamingRootMeanSquaredErrorTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def testVars(self):
    metrics.streaming_root_mean_squared_error(
        predictions=tf.ones((10, 1)), labels=tf.ones((10, 1)))
    _assert_local_variables(self, (
        'root_mean_squared_error/count:0',
        'root_mean_squared_error/total:0'
    ))

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = metrics.streaming_root_mean_squared_error(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [mean])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.streaming_root_mean_squared_error(
        predictions=tf.ones((10, 1)),
        labels=tf.ones((10, 1)),
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testValueTensorIsIdempotent(self):
    predictions = tf.random_normal((10, 3), seed=1)
    labels = tf.random_normal((10, 3), seed=2)
    error, update_op = metrics.streaming_root_mean_squared_error(
        predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        sess.run(update_op)

      # Then verify idempotency.
      initial_error = error.eval()
      for _ in range(10):
        self.assertEqual(initial_error, error.eval())

  def testSingleUpdateZeroError(self):
    with self.test_session() as sess:
      predictions = tf.constant(0.0, shape=(1, 3), dtype=tf.float32)
      labels = tf.constant(0.0, shape=(1, 3), dtype=tf.float32)

      rmse, update_op = metrics.streaming_root_mean_squared_error(
          predictions, labels)

      sess.run(tf.local_variables_initializer())
      self.assertEqual(0, sess.run(update_op))

      self.assertEqual(0, rmse.eval())

  def testSingleUpdateWithError(self):
    with self.test_session() as sess:
      predictions = tf.constant([2, 4, 6], shape=(1, 3), dtype=tf.float32)
      labels = tf.constant([1, 3, 2], shape=(1, 3), dtype=tf.float32)

      rmse, update_op = metrics.streaming_root_mean_squared_error(
          predictions, labels)

      sess.run(tf.local_variables_initializer())
      self.assertAlmostEqual(math.sqrt(6), update_op.eval(), 5)
      self.assertAlmostEqual(math.sqrt(6), rmse.eval(), 5)

  def testSingleUpdateWithErrorAndWeights(self):
    with self.test_session() as sess:
      predictions = tf.constant([2, 4, 6, 8], shape=(1, 4), dtype=tf.float32)
      labels = tf.constant([1, 3, 2, 3], shape=(1, 4), dtype=tf.float32)
      weights = tf.constant([0, 1, 0, 1], shape=(1, 4))

      rmse, update_op = metrics.streaming_root_mean_squared_error(
          predictions, labels, weights)

      sess.run(tf.local_variables_initializer())
      self.assertAlmostEqual(math.sqrt(13), sess.run(update_op))

      self.assertAlmostEqual(math.sqrt(13), rmse.eval(), 5)


def _reweight(predictions, labels, weights):
  return (np.concatenate([[p] * int(w) for p, w in zip(predictions, weights)]),
          np.concatenate([[l] * int(w) for l, w in zip(labels, weights)]))


class StreamingCovarianceTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def testVars(self):
    metrics.streaming_covariance(
        predictions=tf.to_float(tf.range(10)) + tf.ones([10, 10]),
        labels=tf.to_float(tf.range(10)) + tf.ones([10, 10]))
    _assert_local_variables(self, (
        'covariance/comoment:0',
        'covariance/count:0',
        'covariance/mean_label:0',
        'covariance/mean_prediction:0',
    ))

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    cov, _ = metrics.streaming_covariance(
        predictions=tf.to_float(tf.range(10)) + tf.ones([10, 10]),
        labels=tf.to_float(tf.range(10)) + tf.ones([10, 10]),
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [cov])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.streaming_covariance(
        predictions=tf.to_float(tf.range(10)) + tf.ones([10, 10]),
        labels=tf.to_float(tf.range(10)) + tf.ones([10, 10]),
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testValueTensorIsIdempotent(self):
    labels = tf.random_normal((10, 3), seed=2)
    predictions = labels * 0.5 + tf.random_normal((10, 3), seed=1) * 0.5
    cov, update_op = metrics.streaming_covariance(predictions, labels)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        sess.run(update_op)

      # Then verify idempotency.
      initial_cov = cov.eval()
      for _ in range(10):
        self.assertEqual(initial_cov, cov.eval())

  def testSingleUpdateIdentical(self):
    with self.test_session() as sess:
      predictions = tf.to_float(tf.range(10))
      labels = tf.to_float(tf.range(10))

      cov, update_op = metrics.streaming_covariance(predictions, labels)

      expected_cov = np.cov(np.arange(10), np.arange(10))[0, 1]
      sess.run(tf.local_variables_initializer())
      self.assertAlmostEqual(expected_cov, sess.run(update_op), 5)
      self.assertAlmostEqual(expected_cov, cov.eval(), 5)

  def testSingleUpdateNonIdentical(self):
    with self.test_session() as sess:
      predictions = tf.constant([2, 4, 6], shape=(1, 3), dtype=tf.float32)
      labels = tf.constant([1, 3, 2], shape=(1, 3), dtype=tf.float32)

      cov, update_op = metrics.streaming_covariance(predictions, labels)

      expected_cov = np.cov([2, 4, 6], [1, 3, 2])[0, 1]
      sess.run(tf.local_variables_initializer())
      self.assertAlmostEqual(expected_cov, update_op.eval())
      self.assertAlmostEqual(expected_cov, cov.eval())

  def testSingleUpdateWithErrorAndWeights(self):
    with self.test_session() as sess:
      predictions = tf.constant([2, 4, 6, 8], shape=(1, 4), dtype=tf.float32)
      labels = tf.constant([1, 3, 2, 7], shape=(1, 4), dtype=tf.float32)
      weights = tf.constant([0, 1, 3, 1], shape=(1, 4), dtype=tf.float32)

      cov, update_op = metrics.streaming_covariance(
          predictions, labels, weights=weights)

      p, l = _reweight([2, 4, 6, 8], [1, 3, 2, 7], [0, 1, 3, 1])
      expected_cov = np.cov(p, l)[0, 1]
      sess.run(tf.local_variables_initializer())
      self.assertAlmostEqual(expected_cov, sess.run(update_op))
      self.assertAlmostEqual(expected_cov, cov.eval())

  def testMultiUpdateWithErrorNoWeights(self):
    with self.test_session() as sess:
      np.random.seed(123)
      n = 100
      predictions = np.random.randn(n)
      labels = 0.5 * predictions + np.random.randn(n)

      stride = 10
      predictions_t = tf.placeholder(tf.float32, [stride])
      labels_t = tf.placeholder(tf.float32, [stride])

      cov, update_op = metrics.streaming_covariance(predictions_t, labels_t)

      sess.run(tf.local_variables_initializer())
      prev_expected_cov = 0.
      for i in range(n // stride):
        feed_dict = {
            predictions_t: predictions[stride * i:stride * (i + 1)],
            labels_t: labels[stride * i:stride * (i + 1)]
        }
        self.assertAlmostEqual(
            prev_expected_cov, sess.run(cov, feed_dict=feed_dict), 5)
        expected_cov = np.cov(predictions[:stride * (i + 1)],
                              labels[:stride * (i + 1)])[0, 1]
        self.assertAlmostEqual(
            expected_cov, sess.run(update_op, feed_dict=feed_dict), 5)
        self.assertAlmostEqual(
            expected_cov, sess.run(cov, feed_dict=feed_dict), 5)
        prev_expected_cov = expected_cov

  def testMultiUpdateWithErrorAndWeights(self):
    with self.test_session() as sess:
      np.random.seed(123)
      n = 100
      predictions = np.random.randn(n)
      labels = 0.5 * predictions + np.random.randn(n)
      weights = np.tile(np.arange(n // 10), n // 10)
      np.random.shuffle(weights)

      stride = 10
      predictions_t = tf.placeholder(tf.float32, [stride])
      labels_t = tf.placeholder(tf.float32, [stride])
      weights_t = tf.placeholder(tf.float32, [stride])

      cov, update_op = metrics.streaming_covariance(
          predictions_t, labels_t, weights=weights_t)

      sess.run(tf.local_variables_initializer())
      prev_expected_cov = 0.
      for i in range(n // stride):
        feed_dict = {
            predictions_t: predictions[stride * i:stride * (i + 1)],
            labels_t: labels[stride * i:stride * (i + 1)],
            weights_t: weights[stride * i:stride * (i + 1)]
        }
        self.assertAlmostEqual(
            prev_expected_cov, sess.run(cov, feed_dict=feed_dict), 5)
        p, l = _reweight(predictions[:stride * (i + 1)],
                         labels[:stride * (i + 1)], weights[:stride * (i + 1)])
        expected_cov = np.cov(p, l)[0, 1]
        self.assertAlmostEqual(
            expected_cov, sess.run(update_op, feed_dict=feed_dict), 5)
        self.assertAlmostEqual(
            expected_cov, sess.run(cov, feed_dict=feed_dict), 5)
        prev_expected_cov = expected_cov


class StreamingPearsonRTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def testVars(self):
    metrics.streaming_pearson_correlation(
        predictions=tf.to_float(tf.range(10)) + tf.ones([10, 10]),
        labels=tf.to_float(tf.range(10)) + tf.ones([10, 10]))
    _assert_local_variables(self, (
        'pearson_r/covariance/comoment:0',
        'pearson_r/covariance/count:0',
        'pearson_r/covariance/mean_label:0',
        'pearson_r/covariance/mean_prediction:0',
        'pearson_r/variance_labels/count:0',
        'pearson_r/variance_labels/comoment:0',
        'pearson_r/variance_labels/mean_label:0',
        'pearson_r/variance_labels/mean_prediction:0',
        'pearson_r/variance_predictions/comoment:0',
        'pearson_r/variance_predictions/count:0',
        'pearson_r/variance_predictions/mean_label:0',
        'pearson_r/variance_predictions/mean_prediction:0',
    ))

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    pearson_r, _ = metrics.streaming_pearson_correlation(
        predictions=tf.to_float(tf.range(10)) + tf.ones([10, 10]),
        labels=tf.to_float(tf.range(10)) + tf.ones([10, 10]),
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [pearson_r])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.streaming_pearson_correlation(
        predictions=tf.to_float(tf.range(10)) + tf.ones([10, 10]),
        labels=tf.to_float(tf.range(10)) + tf.ones([10, 10]),
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testValueTensorIsIdempotent(self):
    labels = tf.random_normal((10, 3), seed=2)
    predictions = labels * 0.5 + tf.random_normal((10, 3), seed=1) * 0.5
    pearson_r, update_op = metrics.streaming_pearson_correlation(predictions,
                                                                 labels)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        sess.run(update_op)

      # Then verify idempotency.
      initial_r = pearson_r.eval()
      for _ in range(10):
        self.assertEqual(initial_r, pearson_r.eval())

  def testSingleUpdateIdentical(self):
    with self.test_session() as sess:
      predictions = tf.to_float(tf.range(10))
      labels = tf.to_float(tf.range(10))

      pearson_r, update_op = metrics.streaming_pearson_correlation(predictions,
                                                                   labels)

      expected_r = np.corrcoef(np.arange(10), np.arange(10))[0, 1]
      sess.run(tf.local_variables_initializer())
      self.assertAlmostEqual(expected_r, sess.run(update_op), 5)
      self.assertAlmostEqual(expected_r, pearson_r.eval(), 5)

  def testSingleUpdateNonIdentical(self):
    with self.test_session() as sess:
      predictions = tf.constant([2, 4, 6], shape=(1, 3), dtype=tf.float32)
      labels = tf.constant([1, 3, 2], shape=(1, 3), dtype=tf.float32)

      pearson_r, update_op = metrics.streaming_pearson_correlation(predictions,
                                                                   labels)

      expected_r = np.corrcoef([2, 4, 6], [1, 3, 2])[0, 1]
      sess.run(tf.local_variables_initializer())
      self.assertAlmostEqual(expected_r, update_op.eval())
      self.assertAlmostEqual(expected_r, pearson_r.eval())

  def testSingleUpdateWithErrorAndWeights(self):
    with self.test_session() as sess:
      predictions = np.array([2, 4, 6, 8])
      labels = np.array([1, 3, 2, 7])
      weights = np.array([0, 1, 3, 1])
      predictions_t = tf.constant(predictions, shape=(1, 4), dtype=tf.float32)
      labels_t = tf.constant(labels, shape=(1, 4), dtype=tf.float32)
      weights_t = tf.constant(weights, shape=(1, 4), dtype=tf.float32)

      pearson_r, update_op = metrics.streaming_pearson_correlation(
          predictions_t, labels_t, weights=weights_t)

      p, l = _reweight(predictions, labels, weights)
      cmat = np.cov(p, l)
      expected_r = cmat[0, 1] / np.sqrt(cmat[0, 0] * cmat[1, 1])
      sess.run(tf.local_variables_initializer())
      self.assertAlmostEqual(expected_r, sess.run(update_op))
      self.assertAlmostEqual(expected_r, pearson_r.eval())

  def testMultiUpdateWithErrorNoWeights(self):
    with self.test_session() as sess:
      np.random.seed(123)
      n = 100
      predictions = np.random.randn(n)
      labels = 0.5 * predictions + np.random.randn(n)

      stride = 10
      predictions_t = tf.placeholder(tf.float32, [stride])
      labels_t = tf.placeholder(tf.float32, [stride])

      pearson_r, update_op = metrics.streaming_pearson_correlation(
          predictions_t, labels_t)

      sess.run(tf.local_variables_initializer())
      prev_expected_r = 0.
      for i in range(n // stride):
        feed_dict = {
            predictions_t: predictions[stride * i:stride * (i + 1)],
            labels_t: labels[stride * i:stride * (i + 1)]
        }
        self.assertAlmostEqual(
            prev_expected_r, sess.run(pearson_r, feed_dict=feed_dict), 5)
        expected_r = np.corrcoef(predictions[:stride * (i + 1)],
                                 labels[:stride * (i + 1)])[0, 1]
        self.assertAlmostEqual(
            expected_r, sess.run(update_op, feed_dict=feed_dict), 5)
        self.assertAlmostEqual(
            expected_r, sess.run(pearson_r, feed_dict=feed_dict), 5)
        prev_expected_r = expected_r

  def testMultiUpdateWithErrorAndWeights(self):
    with self.test_session() as sess:
      np.random.seed(123)
      n = 100
      predictions = np.random.randn(n)
      labels = 0.5 * predictions + np.random.randn(n)
      weights = np.tile(np.arange(n // 10), n // 10)
      np.random.shuffle(weights)

      stride = 10
      predictions_t = tf.placeholder(tf.float32, [stride])
      labels_t = tf.placeholder(tf.float32, [stride])
      weights_t = tf.placeholder(tf.float32, [stride])

      pearson_r, update_op = metrics.streaming_pearson_correlation(
          predictions_t, labels_t, weights=weights_t)

      sess.run(tf.local_variables_initializer())
      prev_expected_r = 0.
      for i in range(n // stride):
        feed_dict = {
            predictions_t: predictions[stride * i:stride * (i + 1)],
            labels_t: labels[stride * i:stride * (i + 1)],
            weights_t: weights[stride * i:stride * (i + 1)]
        }
        self.assertAlmostEqual(
            prev_expected_r, sess.run(pearson_r, feed_dict=feed_dict), 5)
        p, l = _reweight(predictions[:stride * (i + 1)],
                         labels[:stride * (i + 1)], weights[:stride * (i + 1)])
        cmat = np.cov(p, l)
        expected_r = cmat[0, 1] / np.sqrt(cmat[0, 0] * cmat[1, 1])
        self.assertAlmostEqual(
            expected_r, sess.run(update_op, feed_dict=feed_dict), 5)
        self.assertAlmostEqual(
            expected_r, sess.run(pearson_r, feed_dict=feed_dict), 5)
        prev_expected_r = expected_r


class StreamingMeanCosineDistanceTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def testVars(self):
    metrics.streaming_mean_cosine_distance(
        predictions=tf.ones((10, 3)), labels=tf.ones((10, 3)), dim=1)
    _assert_local_variables(self, (
        'mean_cosine_distance/count:0',
        'mean_cosine_distance/total:0',
    ))

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = metrics.streaming_mean_cosine_distance(
        predictions=tf.ones((10, 3)),
        labels=tf.ones((10, 3)),
        dim=1,
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [mean])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.streaming_mean_cosine_distance(
        predictions=tf.ones((10, 3)),
        labels=tf.ones((10, 3)),
        dim=1,
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testValueTensorIsIdempotent(self):
    predictions = tf.random_normal((10, 3), seed=1)
    labels = tf.random_normal((10, 3), seed=2)
    error, update_op = metrics.streaming_mean_cosine_distance(
        predictions, labels, dim=1)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        sess.run(update_op)

      # Then verify idempotency.
      initial_error = error.eval()
      for _ in range(10):
        self.assertEqual(initial_error, error.eval())

  def testSingleUpdateZeroError(self):
    np_labels = np.matrix(('1 0 0;'
                           '0 0 1;'
                           '0 1 0'))

    predictions = tf.constant(np_labels, shape=(1, 3, 3), dtype=tf.float32)
    labels = tf.constant(np_labels, shape=(1, 3, 3), dtype=tf.float32)

    error, update_op = metrics.streaming_mean_cosine_distance(
        predictions, labels, dim=2)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertEqual(0, sess.run(update_op))
      self.assertEqual(0, error.eval())

  def testSingleUpdateWithError1(self):
    np_labels = np.matrix(('1 0 0;'
                           '0 0 1;'
                           '0 1 0'))
    np_predictions = np.matrix(('1 0 0;'
                                '0 0 -1;'
                                '1 0 0'))

    predictions = tf.constant(np_predictions, shape=(3, 1, 3), dtype=tf.float32)
    labels = tf.constant(np_labels, shape=(3, 1, 3), dtype=tf.float32)

    error, update_op = metrics.streaming_mean_cosine_distance(
        predictions, labels, dim=2)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertAlmostEqual(1, sess.run(update_op), 5)
      self.assertAlmostEqual(1, error.eval(), 5)

  def testSingleUpdateWithError2(self):
    np_predictions = np.matrix((
        '0.819031913261206 0.567041924552012 0.087465312324590;'
        '-0.665139432070255 -0.739487441769973 -0.103671883216994;'
        '0.707106781186548 -0.707106781186548 0'))
    np_labels = np.matrix((
        '0.819031913261206 0.567041924552012 0.087465312324590;'
        '0.665139432070255 0.739487441769973 0.103671883216994;'
        '0.707106781186548 0.707106781186548 0'))

    predictions = tf.constant(np_predictions, shape=(3, 1, 3), dtype=tf.float32)
    labels = tf.constant(np_labels, shape=(3, 1, 3), dtype=tf.float32)
    error, update_op = metrics.streaming_mean_cosine_distance(
        predictions, labels, dim=2)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertAlmostEqual(1.0, sess.run(update_op), 5)
      self.assertAlmostEqual(1.0, error.eval(), 5)

  def testSingleUpdateWithErrorAndWeights1(self):
    np_predictions = np.matrix(('1 0 0;'
                                '0 0 -1;'
                                '1 0 0'))
    np_labels = np.matrix(('1 0 0;'
                           '0 0 1;'
                           '0 1 0'))

    predictions = tf.constant(np_predictions, shape=(3, 1, 3), dtype=tf.float32)
    labels = tf.constant(np_labels, shape=(3, 1, 3), dtype=tf.float32)
    weights = tf.constant([1, 0, 0], shape=(3, 1, 1), dtype=tf.float32)

    error, update_op = metrics.streaming_mean_cosine_distance(
        predictions, labels, dim=2, weights=weights)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertEqual(0, sess.run(update_op))
      self.assertEqual(0, error.eval())

  def testSingleUpdateWithErrorAndWeights2(self):
    np_predictions = np.matrix(('1 0 0;'
                                '0 0 -1;'
                                '1 0 0'))
    np_labels = np.matrix(('1 0 0;'
                           '0 0 1;'
                           '0 1 0'))

    predictions = tf.constant(np_predictions, shape=(3, 1, 3), dtype=tf.float32)
    labels = tf.constant(np_labels, shape=(3, 1, 3), dtype=tf.float32)
    weights = tf.constant([0, 1, 1], shape=(3, 1, 1), dtype=tf.float32)

    error, update_op = metrics.streaming_mean_cosine_distance(
        predictions, labels, dim=2, weights=weights)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertEqual(1.5, update_op.eval())
      self.assertEqual(1.5, error.eval())


class PcntBelowThreshTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def testVars(self):
    metrics.streaming_percentage_less(values=tf.ones((10,)), threshold=2)
    _assert_local_variables(self, (
        'percentage_below_threshold/count:0',
        'percentage_below_threshold/total:0',
    ))

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    mean, _ = metrics.streaming_percentage_less(
        values=tf.ones((10,)),
        threshold=2,
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [mean])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.streaming_percentage_less(
        values=tf.ones((10,)),
        threshold=2,
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testOneUpdate(self):
    with self.test_session() as sess:
      values = tf.constant([2, 4, 6, 8], shape=(1, 4), dtype=tf.float32)

      pcnt0, update_op0 = metrics.streaming_percentage_less(
          values, 100, name='high')
      pcnt1, update_op1 = metrics.streaming_percentage_less(
          values, 7, name='medium')
      pcnt2, update_op2 = metrics.streaming_percentage_less(
          values, 1, name='low')

      sess.run(tf.local_variables_initializer())
      sess.run([update_op0, update_op1, update_op2])

      pcnt0, pcnt1, pcnt2 = sess.run([pcnt0, pcnt1, pcnt2])
      self.assertAlmostEqual(1.0, pcnt0, 5)
      self.assertAlmostEqual(0.75, pcnt1, 5)
      self.assertAlmostEqual(0.0, pcnt2, 5)

  def testSomePresentOneUpdate(self):
    with self.test_session() as sess:
      values = tf.constant([2, 4, 6, 8], shape=(1, 4), dtype=tf.float32)
      weights = tf.constant([1, 0, 0, 1], shape=(1, 4), dtype=tf.float32)

      pcnt0, update_op0 = metrics.streaming_percentage_less(
          values, 100, weights=weights, name='high')
      pcnt1, update_op1 = metrics.streaming_percentage_less(
          values, 7, weights=weights, name='medium')
      pcnt2, update_op2 = metrics.streaming_percentage_less(
          values, 1, weights=weights, name='low')

      sess.run(tf.local_variables_initializer())
      self.assertListEqual([1.0, 0.5, 0.0],
                           sess.run([update_op0, update_op1, update_op2]))

      pcnt0, pcnt1, pcnt2 = sess.run([pcnt0, pcnt1, pcnt2])
      self.assertAlmostEqual(1.0, pcnt0, 5)
      self.assertAlmostEqual(0.5, pcnt1, 5)
      self.assertAlmostEqual(0.0, pcnt2, 5)


class StreamingMeanIOUTest(tf.test.TestCase):

  def setUp(self):
    np.random.seed(1)
    tf.reset_default_graph()

  def testVars(self):
    metrics.streaming_mean_iou(
        predictions=tf.ones([10, 1]), labels=tf.ones([10, 1]), num_classes=2)
    _assert_local_variables(self, ('mean_iou/total_confusion_matrix:0',))

  def testMetricsCollections(self):
    my_collection_name = '__metrics__'
    mean_iou, _ = metrics.streaming_mean_iou(
        predictions=tf.ones([10, 1]),
        labels=tf.ones([10, 1]),
        num_classes=2,
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [mean_iou])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.streaming_mean_iou(
        predictions=tf.ones([10, 1]),
        labels=tf.ones([10, 1]),
        num_classes=2,
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testPredictionsAndLabelsOfDifferentSizeRaisesValueError(self):
    predictions = tf.ones([10, 3])
    labels = tf.ones([10, 4])
    with self.assertRaises(ValueError):
      metrics.streaming_mean_iou(
          predictions, labels, num_classes=2)

  def testLabelsAndWeightsOfDifferentSizeRaisesValueError(self):
    predictions = tf.ones([10])
    labels = tf.ones([10])
    weights = tf.zeros([9])
    with self.assertRaises(ValueError):
      metrics.streaming_mean_iou(
          predictions, labels, num_classes=2, weights=weights)

  def testValueTensorIsIdempotent(self):
    num_classes = 3
    predictions = tf.random_uniform([10], maxval=num_classes,
                                    dtype=tf.int64, seed=1)
    labels = tf.random_uniform([10], maxval=num_classes,
                               dtype=tf.int64, seed=1)
    miou, update_op = metrics.streaming_mean_iou(
        predictions, labels, num_classes=num_classes)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())

      # Run several updates.
      for _ in range(10):
        sess.run(update_op)

      # Then verify idempotency.
      initial_miou = miou.eval()
      for _ in range(10):
        self.assertEqual(initial_miou, miou.eval())

  def testMultipleUpdates(self):
    num_classes = 3
    with self.test_session() as sess:
      # Create the queue that populates the predictions.
      preds_queue = tf.FIFOQueue(5, dtypes=tf.int32, shapes=(1, 1))
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [2])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [0])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      labels_queue = tf.FIFOQueue(5, dtypes=tf.int32, shapes=(1, 1))
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [2])
      _enqueue_vector(sess, labels_queue, [1])
      labels = labels_queue.dequeue()

      miou, update_op = metrics.streaming_mean_iou(
          predictions, labels, num_classes)

      sess.run(tf.local_variables_initializer())
      for _ in range(5):
        sess.run(update_op)
      desired_output = np.mean([1.0/2.0, 1.0/4.0, 0.])
      self.assertEqual(desired_output, miou.eval())

  def testMultipleUpdatesWithWeights(self):
    num_classes = 2
    with self.test_session() as sess:
      # Create the queue that populates the predictions.
      preds_queue = tf.FIFOQueue(6, dtypes=tf.int32, shapes=(1, 1))
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      labels_queue = tf.FIFOQueue(6, dtypes=tf.int32, shapes=(1, 1))
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      labels = labels_queue.dequeue()

      # Create the queue that populates the weights.
      weights_queue = tf.FIFOQueue(6, dtypes=tf.float32, shapes=(1, 1))
      _enqueue_vector(sess, weights_queue, [1.0])
      _enqueue_vector(sess, weights_queue, [1.0])
      _enqueue_vector(sess, weights_queue, [1.0])
      _enqueue_vector(sess, weights_queue, [0.0])
      _enqueue_vector(sess, weights_queue, [1.0])
      _enqueue_vector(sess, weights_queue, [0.0])
      weights = weights_queue.dequeue()

      miou, update_op = metrics.streaming_mean_iou(
          predictions, labels, num_classes, weights=weights)

      sess.run(tf.local_variables_initializer())
      for _ in range(6):
        sess.run(update_op)
      desired_output = np.mean([2.0/3.0, 1.0/2.0])
      self.assertAlmostEqual(desired_output, miou.eval())

  def testMultipleUpdatesWithMissingClass(self):
    # Test the case where there are no predicions and labels for
    # one class, and thus there is one row and one column with
    # zero entries in the confusion matrix.
    num_classes = 3
    with self.test_session() as sess:
      # Create the queue that populates the predictions.
      # There is no prediction for class 2.
      preds_queue = tf.FIFOQueue(5, dtypes=tf.int32, shapes=(1, 1))
      _enqueue_vector(sess, preds_queue, [0])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [1])
      _enqueue_vector(sess, preds_queue, [0])
      predictions = preds_queue.dequeue()

      # Create the queue that populates the labels.
      # There is label for class 2.
      labels_queue = tf.FIFOQueue(5, dtypes=tf.int32, shapes=(1, 1))
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [1])
      _enqueue_vector(sess, labels_queue, [0])
      _enqueue_vector(sess, labels_queue, [1])
      labels = labels_queue.dequeue()

      miou, update_op = metrics.streaming_mean_iou(
          predictions, labels, num_classes)

      sess.run(tf.local_variables_initializer())
      for _ in range(5):
        sess.run(update_op)
      desired_output = np.mean([1.0/3.0, 2.0/4.0, 0.])
      self.assertAlmostEqual(desired_output, miou.eval())

  def testUpdateOpEvalIsAccumulatedConfusionMatrix(self):
    predictions = tf.concat(0,
                            [tf.constant(0, shape=[5]),
                             tf.constant(1, shape=[5])])
    labels = tf.concat(0,
                       [tf.constant(0, shape=[3]),
                        tf.constant(1, shape=[7])])
    num_classes = 2
    with self.test_session() as sess:
      miou, update_op = metrics.streaming_mean_iou(
          predictions, labels, num_classes)
      sess.run(tf.local_variables_initializer())
      confusion_matrix = update_op.eval()
      self.assertAllEqual([[3, 2], [0, 5]], confusion_matrix)
      desired_miou = np.mean([3./5., 5./7.])
      self.assertAlmostEqual(desired_miou, miou.eval())

  def testAllCorrect(self):
    predictions = tf.zeros([40])
    labels = tf.zeros([40])
    num_classes = 1
    with self.test_session() as sess:
      miou, update_op = metrics.streaming_mean_iou(
          predictions, labels, num_classes)
      sess.run(tf.local_variables_initializer())
      self.assertEqual(40, update_op.eval()[0])
      self.assertEqual(1.0, miou.eval())

  def testAllWrong(self):
    predictions = tf.zeros([40])
    labels = tf.ones([40])
    num_classes = 2
    with self.test_session() as sess:
      miou, update_op = metrics.streaming_mean_iou(
          predictions, labels, num_classes)
      sess.run(tf.local_variables_initializer())
      self.assertAllEqual([[0, 40], [0, 0]], update_op.eval())
      self.assertEqual(0., miou.eval())

  def testResultsWithSomeMissing(self):
    predictions = tf.concat(0, [tf.constant(0, shape=[5]),
                                tf.constant(1, shape=[5])])
    labels = tf.concat(0, [tf.constant(0, shape=[3]),
                           tf.constant(1, shape=[7])])
    num_classes = 2
    weights = tf.concat(0, [tf.constant(0, shape=[1]),
                            tf.constant(1, shape=[8]),
                            tf.constant(0, shape=[1])])
    with self.test_session() as sess:
      miou, update_op = metrics.streaming_mean_iou(
          predictions, labels, num_classes, weights=weights)
      sess.run(tf.local_variables_initializer())
      self.assertAllEqual([[2, 2], [0, 4]], update_op.eval())
      desired_miou = np.mean([2./4., 4./6.])
      self.assertAlmostEqual(desired_miou, miou.eval())


class StreamingConcatTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def testVars(self):
    metrics.streaming_concat(values=tf.ones((10,)))
    _assert_local_variables(self, (
        'streaming_concat/array:0',
        'streaming_concat/size:0',
    ))

  def testMetricsCollection(self):
    my_collection_name = '__metrics__'
    value, _ = metrics.streaming_concat(
        values=tf.ones((10,)),
        metrics_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [value])

  def testUpdatesCollection(self):
    my_collection_name = '__updates__'
    _, update_op = metrics.streaming_concat(
        values=tf.ones((10,)),
        updates_collections=[my_collection_name])
    self.assertListEqual(tf.get_collection(my_collection_name), [update_op])

  def testNextArraySize(self):
    next_array_size = metrics.python.ops.metric_ops._next_array_size
    with self.test_session():
      self.assertEqual(next_array_size(2, growth_factor=2).eval(), 2)
      self.assertEqual(next_array_size(3, growth_factor=2).eval(), 4)
      self.assertEqual(next_array_size(4, growth_factor=2).eval(), 4)
      self.assertEqual(next_array_size(5, growth_factor=2).eval(), 8)
      self.assertEqual(next_array_size(6, growth_factor=2).eval(), 8)

  def testStreamingConcat(self):
    with self.test_session() as sess:
      values = tf.placeholder(tf.int32, [None])
      concatenated, update_op = metrics.streaming_concat(values)
      sess.run(tf.local_variables_initializer())

      self.assertAllEqual([], concatenated.eval())

      sess.run([update_op], feed_dict={values: [0, 1, 2]})
      self.assertAllEqual([0, 1, 2], concatenated.eval())

      sess.run([update_op], feed_dict={values: [3, 4]})
      self.assertAllEqual([0, 1, 2, 3, 4], concatenated.eval())

      sess.run([update_op], feed_dict={values: [5, 6, 7, 8, 9]})
      self.assertAllEqual(np.arange(10), concatenated.eval())

  def testStreamingConcatMaxSize(self):
    with self.test_session() as sess:
      values = tf.range(3)
      concatenated, update_op = metrics.streaming_concat(values, max_size=5)
      sess.run(tf.local_variables_initializer())

      self.assertAllEqual([], concatenated.eval())

      sess.run([update_op])
      self.assertAllEqual([0, 1, 2], concatenated.eval())

      sess.run([update_op])
      self.assertAllEqual([0, 1, 2, 0, 1], concatenated.eval())

      sess.run([update_op])
      self.assertAllEqual([0, 1, 2, 0, 1], concatenated.eval())

  def testStreamingConcat2D(self):
    with self.test_session() as sess:
      values = tf.reshape(tf.range(3), (3, 1))
      concatenated, update_op = metrics.streaming_concat(values, axis=-1)
      sess.run(tf.local_variables_initializer())
      for _ in range(10):
        sess.run([update_op])
      self.assertAllEqual([[0] * 10, [1] * 10, [2] * 10],
                          concatenated.eval())

  def testStreamingConcatErrors(self):
    with self.assertRaises(ValueError):
      metrics.streaming_concat(tf.placeholder(tf.float32))

    values = tf.zeros((2, 3))
    with self.assertRaises(ValueError):
      metrics.streaming_concat(values, axis=-3, max_size=3)
    with self.assertRaises(ValueError):
      metrics.streaming_concat(values, axis=2, max_size=3)

    with self.assertRaises(ValueError):
      metrics.streaming_concat(tf.placeholder(tf.float32, [None, None]))

  def testStreamingConcatReset(self):
    with self.test_session() as sess:
      values = tf.placeholder(tf.int32, [None])
      concatenated, update_op = metrics.streaming_concat(values)
      sess.run(tf.local_variables_initializer())

      self.assertAllEqual([], concatenated.eval())

      sess.run([update_op], feed_dict={values: [0, 1, 2]})
      self.assertAllEqual([0, 1, 2], concatenated.eval())

      sess.run(tf.local_variables_initializer())

      sess.run([update_op], feed_dict={values: [3, 4]})
      self.assertAllEqual([3, 4], concatenated.eval())


class AggregateMetricsTest(tf.test.TestCase):

  def testAggregateNoMetricsRaisesValueError(self):
    with self.assertRaises(ValueError):
      metrics.aggregate_metrics()

  def testAggregateSingleMetricReturnsOneItemLists(self):
    values = tf.ones((10, 4))
    value_tensors, update_ops = metrics.aggregate_metrics(
        metrics.streaming_mean(values))
    self.assertEqual(len(value_tensors), 1)
    self.assertEqual(len(update_ops), 1)
    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertEqual(1, update_ops[0].eval())
      self.assertEqual(1, value_tensors[0].eval())

  def testAggregateMultipleMetricsReturnsListsInOrder(self):
    predictions = tf.ones((10, 4))
    labels = tf.ones((10, 4)) * 3
    value_tensors, update_ops = metrics.aggregate_metrics(
        metrics.streaming_mean_absolute_error(
            predictions, labels),
        metrics.streaming_mean_squared_error(
            predictions, labels))
    self.assertEqual(len(value_tensors), 2)
    self.assertEqual(len(update_ops), 2)
    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertEqual(2, update_ops[0].eval())
      self.assertEqual(4, update_ops[1].eval())
      self.assertEqual(2, value_tensors[0].eval())
      self.assertEqual(4, value_tensors[1].eval())


class AggregateMetricMapTest(tf.test.TestCase):

  def testAggregateMultipleMetricsReturnsListsInOrder(self):
    predictions = tf.ones((10, 4))
    labels = tf.ones((10, 4)) * 3
    names_to_values, names_to_updates = metrics.aggregate_metric_map(
        {
            'm1': metrics.streaming_mean_absolute_error(
                predictions, labels),
            'm2': metrics.streaming_mean_squared_error(
                predictions, labels),
        })

    self.assertEqual(2, len(names_to_values))
    self.assertEqual(2, len(names_to_updates))

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      self.assertEqual(2, names_to_updates['m1'].eval())
      self.assertEqual(4, names_to_updates['m2'].eval())
      self.assertEqual(2, names_to_values['m1'].eval())
      self.assertEqual(4, names_to_values['m2'].eval())


class NumRelevantTest(tf.test.TestCase):

  def testNumRelevantInvalidArgs(self):
    labels = tf.random_uniform(
        shape=(3, 3, 3), minval=0, maxval=100, dtype=tf.int32)
    with self.assertRaisesRegexp(ValueError, 'nvalid k'):
      metric_ops.num_relevant(labels, k=0)
    with self.assertRaisesRegexp(ValueError, 'nvalid k'):
      metric_ops.num_relevant(labels, k=-1)

  def testNumRelevantDense(self):
    with self.test_session():
      labels = tf.random_uniform(
          shape=(3, 3, 3), minval=0, maxval=100, dtype=tf.int32)
      ones = np.ones(shape=(3, 3))
      self.assertAllEqual(ones, metric_ops.num_relevant(labels, k=1).eval())
      twos = ones * 2
      self.assertAllEqual(twos, metric_ops.num_relevant(labels, k=2).eval())
      threes = ones * 3
      self.assertAllEqual(threes, metric_ops.num_relevant(labels, k=3).eval())
      self.assertAllEqual(threes, metric_ops.num_relevant(labels, k=4).eval())
      self.assertAllEqual(threes, metric_ops.num_relevant(labels, k=999).eval())

  def testNumRelevantSparse(self):
    with self.test_session():
      labels = tf.SparseTensorValue(
          indices=(
              (0, 0, 0), (0, 0, 1),
              (0, 1, 0), (0, 1, 1), (0, 1, 2),
              # (0, 2) missing
              (1, 0, 0), (1, 0, 1), (1, 0, 2),
              (1, 1, 0),
              (1, 2, 0),
              # (2, 0) missing
              (2, 1, 0), (2, 1, 1),
              (2, 2, 0)),
          values=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
          shape=(3, 3, 3))
      self.assertAllEqual(
          ((1, 1, 0), (1, 1, 1), (0, 1, 1)),
          metric_ops.num_relevant(labels, k=1).eval())
      self.assertAllEqual(
          ((2, 2, 0), (2, 1, 1), (0, 2, 1)),
          metric_ops.num_relevant(labels, k=2).eval())
      label_lengths = ((2, 3, 0), (3, 1, 1), (0, 2, 1))
      self.assertAllEqual(
          label_lengths, metric_ops.num_relevant(labels, k=3).eval())
      self.assertAllEqual(
          label_lengths, metric_ops.num_relevant(labels, k=999).eval())


class ExpandAndTileTest(tf.test.TestCase):

  def testExpandAndTileInvalidArgs(self):
    x = tf.ones(shape=(3, 3, 3))
    with self.assertRaisesRegexp(ValueError, 'nvalid multiple'):
      metric_ops.expand_and_tile(x, multiple=0)
    with self.test_session():
      with self.assertRaises(ValueError):
        metric_ops.expand_and_tile(x, multiple=1, dim=-4).eval()
      with self.assertRaises(ValueError):
        metric_ops.expand_and_tile(x, multiple=1, dim=4).eval()

  def testSparseExpandAndTileInvalidArgs(self):
    x = tf.SparseTensorValue(
        indices=[
            (i, j, k) for i in range(3) for j in range(3) for k in range(3)],
        values=[1] * 27,
        shape=[3, 3, 3])
    with self.assertRaisesRegexp(ValueError, 'nvalid multiple'):
      metric_ops.expand_and_tile(x, multiple=0)
    with self.test_session():
      with self.assertRaises(tf.OpError):
        metric_ops.expand_and_tile(x, multiple=1, dim=-4).eval()
      with self.assertRaises(ValueError):
        metric_ops.expand_and_tile(x, multiple=1, dim=4).eval()

  def _test_expand_and_tile(
      self, expected_shape, expected_value, tensor, multiple, dim=None):
    with tf.Graph().as_default() as g, self.test_session(g):
      if dim is None:
        op = metric_ops.expand_and_tile(tensor=tensor, multiple=multiple)
      else:
        op = metric_ops.expand_and_tile(
            tensor=tensor, multiple=multiple, dim=dim)
      self.assertAllEqual(expected_shape, tf.shape(op).eval())
      self.assertAllEqual(expected_value, op.eval())

  # TODO(ptucker): Use @parameterized when it's available in tf.
  def testExpandAndTile1x(self):
    # Shape (3,3,3).
    x = ((
        (1, 2, 3),
        (4, 5, 6),
        (7, 8, 9)
    ), (
        (10, 11, 12),
        (13, 14, 15),
        (16, 17, 18)
    ), (
        (19, 20, 21),
        (22, 23, 24),
        (25, 26, 26)
    ))
    for dim in (None, -3, 0):
      self._test_expand_and_tile(
          expected_shape=(1, 3, 3, 3),
          expected_value=[x],
          tensor=x, multiple=1, dim=dim)

    for dim in (-2, 1):
      self._test_expand_and_tile(
          expected_shape=(3, 1, 3, 3),
          expected_value=[[x1] for x1 in x],
          tensor=x, multiple=1, dim=dim)

    for dim in (-1, 2):
      self._test_expand_and_tile(
          expected_shape=(3, 3, 1, 3),
          expected_value=[[[x2] for x2 in x1] for x1 in x],
          tensor=x, multiple=1, dim=dim)

    self._test_expand_and_tile(
        expected_shape=(3, 3, 3, 1),
        expected_value=[[[[x3] for x3 in x2] for x2 in x1] for x1 in x],
        tensor=x, multiple=1, dim=3)

  # TODO(ptucker): Use @parameterized when it's available in tf.
  def testExpandAndTile5x(self):
    # Shape (3,3,3).
    x = ((
        (1, 2, 3),
        (4, 5, 6),
        (7, 8, 9)
    ), (
        (10, 11, 12),
        (13, 14, 15),
        (16, 17, 18)
    ), (
        (19, 20, 21),
        (22, 23, 24),
        (25, 26, 26)
    ))
    with self.test_session():
      for dim in (None, -3, 0):
        self._test_expand_and_tile(
            expected_shape=(5, 3, 3, 3),
            expected_value=[x] * 5,
            tensor=x, multiple=5, dim=dim)

      for dim in (-2, 1):
        self._test_expand_and_tile(
            expected_shape=(3, 5, 3, 3),
            expected_value=[[x1] * 5 for x1 in x],
            tensor=x, multiple=5, dim=dim)

      for dim in (-1, 2):
        self._test_expand_and_tile(
            expected_shape=(3, 3, 5, 3),
            expected_value=[[[x2] * 5 for x2 in x1] for x1 in x],
            tensor=x, multiple=5, dim=dim)

    self._test_expand_and_tile(
        expected_shape=(3, 3, 3, 5),
        expected_value=[[[[x3] * 5 for x3 in x2] for x2 in x1] for x1 in x],
        tensor=x, multiple=5, dim=3)

  def _assert_sparse_tensors_equal(self, expected, actual):
    self.assertAllEqual(expected.indices, actual.indices)
    self.assertAllEqual(expected.values, actual.values)
    self.assertAllEqual(expected.shape, actual.shape)

  # TODO(ptucker): Use @parameterized when it's available in tf.
  def testSparseExpandAndTile1x(self):
    # Shape (3,3).
    x = tf.SparseTensorValue(
        indices=[
            [0, 0], [0, 1],
            [1, 0], [1, 1], [1, 2],
            [2, 0]],
        values=[
            1, 2,
            3, 4, 5,
            6],
        shape=[3, 3])
    with self.test_session():
      expected_result_dim0 = tf.SparseTensorValue(
          indices=[[0, i[0], i[1]] for i in x.indices], values=x.values,
          shape=[1, 3, 3])
      self._assert_sparse_tensors_equal(
          expected_result_dim0,
          metric_ops.expand_and_tile(x, multiple=1).eval())
      for dim in (-2, 0):
        self._assert_sparse_tensors_equal(
            expected_result_dim0,
            metric_ops.expand_and_tile(x, multiple=1, dim=dim).eval())

      expected_result_dim1 = tf.SparseTensorValue(
          indices=[[i[0], 0, i[1]] for i in x.indices], values=x.values,
          shape=[3, 1, 3])
      for dim in (-1, 1):
        self._assert_sparse_tensors_equal(
            expected_result_dim1,
            metric_ops.expand_and_tile(x, multiple=1, dim=dim).eval())

      expected_result_dim2 = tf.SparseTensorValue(
          indices=[[i[0], i[1], 0] for i in x.indices], values=x.values,
          shape=[3, 3, 1])
      self._assert_sparse_tensors_equal(
          expected_result_dim2,
          metric_ops.expand_and_tile(x, multiple=1, dim=2).eval())

  # TODO(ptucker): Use @parameterized when it's available in tf.
  def testSparseExpandAndTile5x(self):
    # Shape (3,3).
    x = tf.SparseTensorValue(
        indices=(
            (0, 0), (0, 1),
            (1, 0), (1, 1), (1, 2),
            (2, 0)),
        values=(
            1, 2,
            3, 4, 5,
            6),
        shape=(3, 3))
    with self.test_session():
      expected_result_dim0 = tf.SparseTensorValue(
          indices=[(d0, i[0], i[1]) for d0 in range(5) for i in x.indices],
          values=[v for _ in range(5) for v in x.values],
          shape=(5, 3, 3))
      self._assert_sparse_tensors_equal(
          expected_result_dim0,
          metric_ops.expand_and_tile(x, multiple=5).eval())
      for dim in (-2, 0):
        self._assert_sparse_tensors_equal(
            expected_result_dim0,
            metric_ops.expand_and_tile(x, multiple=5, dim=dim).eval())

      expected_result_dim1 = tf.SparseTensorValue(
          indices=[
              (d0, d1, i[1])
              for d0 in range(3)
              for d1 in range(5)
              for i in x.indices if i[0] == d0],
          values=x.values[0:2] * 5 + x.values[2:5] * 5 + x.values[5:] * 5,
          shape=(3, 5, 3))
      for dim in (-1, 1):
        self._assert_sparse_tensors_equal(
            expected_result_dim1,
            metric_ops.expand_and_tile(x, multiple=5, dim=dim).eval())

      expected_result_dim2 = tf.SparseTensorValue(
          indices=[(i[0], i[1], d2) for i in x.indices for d2 in range(5)],
          values=[v for v in x.values for _ in range(5)],
          shape=(3, 3, 5))
      self._assert_sparse_tensors_equal(
          expected_result_dim2,
          metric_ops.expand_and_tile(x, multiple=5, dim=2).eval())


if __name__ == '__main__':
  tf.test.main()
