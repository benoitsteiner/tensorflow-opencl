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

"""Tests for Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import itertools
import tempfile

import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.learn.python.learn.estimators import _sklearn
from tensorflow.contrib.learn.python.learn.estimators import estimator


_BOSTON_INPUT_DIM = 13
_IRIS_INPUT_DIM = 4


def boston_input_fn(num_epochs=None):
  boston = tf.contrib.learn.datasets.load_boston()
  features = tf.train.limit_epochs(
      tf.reshape(tf.constant(boston.data), [-1, _BOSTON_INPUT_DIM]),
      num_epochs=num_epochs)
  labels = tf.reshape(tf.constant(boston.target), [-1, 1])
  return features, labels


def iris_input_fn():
  iris = tf.contrib.learn.datasets.load_iris()
  features = tf.reshape(tf.constant(iris.data), [-1, _IRIS_INPUT_DIM])
  labels = tf.reshape(tf.constant(iris.target), [-1])
  return features, labels


def iris_input_fn_labels_dict():
  iris = tf.contrib.learn.datasets.load_iris()
  features = tf.reshape(tf.constant(iris.data), [-1, _IRIS_INPUT_DIM])
  labels = {
      'labels': tf.reshape(tf.constant(iris.target), [-1])
  }
  return features, labels


def boston_eval_fn():
  boston = tf.contrib.learn.datasets.load_boston()
  n_examples = len(boston.target)
  features = tf.reshape(
      tf.constant(boston.data), [n_examples, _BOSTON_INPUT_DIM])
  labels = tf.reshape(tf.constant(boston.target), [n_examples, 1])
  return tf.concat(0, [features, features]), tf.concat(0, [labels, labels])


def linear_model_params_fn(features, labels, mode, params):
  assert mode in (
      tf.contrib.learn.ModeKeys.TRAIN,
      tf.contrib.learn.ModeKeys.EVAL,
      tf.contrib.learn.ModeKeys.INFER)
  prediction, loss = (
      tf.contrib.learn.models.linear_regression_zero_init(features, labels)
  )
  train_op = tf.contrib.layers.optimize_loss(
      loss, tf.contrib.framework.get_global_step(), optimizer='Adagrad',
      learning_rate=params['learning_rate'])
  return prediction, loss, train_op


def linear_model_fn(features, labels, mode):
  assert mode in (
      tf.contrib.learn.ModeKeys.TRAIN,
      tf.contrib.learn.ModeKeys.EVAL,
      tf.contrib.learn.ModeKeys.INFER)
  prediction, loss = (
      tf.contrib.learn.models.linear_regression_zero_init(features, labels)
  )
  train_op = tf.contrib.layers.optimize_loss(
      loss, tf.contrib.framework.get_global_step(), optimizer='Adagrad',
      learning_rate=0.1)
  return prediction, loss, train_op


def logistic_model_no_mode_fn(features, labels):
  if isinstance(labels, dict):
    labels = labels['labels']
  labels = tf.one_hot(labels, 3, 1, 0)
  prediction, loss = (
      tf.contrib.learn.models.logistic_regression_zero_init(features, labels)
  )
  train_op = tf.contrib.layers.optimize_loss(
      loss, tf.contrib.framework.get_global_step(), optimizer='Adagrad',
      learning_rate=0.1)
  return {'class': tf.argmax(prediction, 1), 'prob': prediction}, loss, train_op


class CheckCallsMonitor(tf.contrib.learn.monitors.BaseMonitor):

  def __init__(self, expect_calls):
    super(CheckCallsMonitor, self).__init__()
    self.begin_calls = None
    self.end_calls = None
    self.expect_calls = expect_calls

  def begin(self, max_steps):
    self.begin_calls = 0
    self.end_calls = 0

  def step_begin(self, step):
    self.begin_calls += 1
    return {}

  def step_end(self, step, outputs):
    self.end_calls += 1
    return False

  def end(self):
    assert (self.end_calls == self.expect_calls and
            self.begin_calls == self.expect_calls)


class EstimatorTest(tf.test.TestCase):

  def testInvalidModelFn_no_train_op(self):
    def _invalid_model_fn(features, labels):
      # pylint: disable=unused-argument
      tf.Variable(42.0, 'weight')
      return None, None, None
    est = tf.contrib.learn.Estimator(model_fn=_invalid_model_fn)
    with self.assertRaisesRegexp(ValueError, 'Missing training_op'):
      est.fit(input_fn=boston_input_fn, steps=1)

  def testInvalidModelFn_no_loss(self):
    def _invalid_model_fn(features, labels, mode):
      # pylint: disable=unused-argument
      w = tf.Variable(42.0, 'weight')
      loss = 100.0 - w
      train_op = w.assign_add(loss / 100.0)
      if mode == tf.contrib.learn.ModeKeys.EVAL:
        loss = None
      return None, loss, train_op
    est = tf.contrib.learn.Estimator(model_fn=_invalid_model_fn)
    est.fit(input_fn=boston_input_fn, steps=1)
    with self.assertRaisesRegexp(ValueError, 'Missing loss'):
      est.evaluate(input_fn=boston_eval_fn, steps=1)

  def testInvalidModelFn_no_prediction(self):
    def _invalid_model_fn(features, labels):
      # pylint: disable=unused-argument
      w = tf.Variable(42.0, 'weight')
      loss = 100.0 - w
      train_op = w.assign_add(loss / 100.0)
      return None, loss, train_op
    est = tf.contrib.learn.Estimator(model_fn=_invalid_model_fn)
    est.fit(input_fn=boston_input_fn, steps=1)
    with self.assertRaisesRegexp(ValueError, 'Missing prediction'):
      est.evaluate(input_fn=boston_eval_fn, steps=1)
    with self.assertRaisesRegexp(ValueError, 'Missing prediction'):
      est.predict(input_fn=boston_input_fn)
    with self.assertRaisesRegexp(ValueError, 'Missing prediction'):
      est.predict(
          input_fn=functools.partial(boston_input_fn, num_epochs=1),
          as_iterable=True)

  def testCustomConfig(self):
    test_random_seed = 5783452

    class TestInput(object):

      def __init__(self):
        self.random_seed = 0

      def config_test_input_fn(self):
        self.random_seed = tf.get_default_graph().seed
        return tf.constant([[1.]]), tf.constant([1.])

    config = tf.contrib.learn.RunConfig(tf_random_seed=test_random_seed)
    test_input = TestInput()
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn, config=config)
    est.fit(input_fn=test_input.config_test_input_fn, steps=1)
    # If input_fn ran, it will have given us the random seed set on the graph.
    self.assertEquals(test_random_seed, test_input.random_seed)

  def testCheckInputs(self):
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn)
    # Lambdas so we have to different objects to compare
    right_features = lambda: np.ones(shape=[7, 8], dtype=np.float32)
    right_labels = lambda: np.ones(shape=[7, 10], dtype=np.int32)
    est.fit(right_features(), right_labels(), steps=1)
    # TODO(wicke): This does not fail for np.int32 because of data_feeder magic.
    wrong_type_features = np.ones(shape=[7., 8.], dtype=np.int64)
    wrong_size_features = np.ones(shape=[7, 10])
    wrong_type_labels = np.ones(shape=[7., 10.], dtype=np.float32)
    wrong_size_labels = np.ones(shape=[7, 11])
    est.fit(x=right_features(), y=right_labels(), steps=1)
    with self.assertRaises(ValueError):
      est.fit(x=wrong_type_features, y=right_labels(), steps=1)
    with self.assertRaises(ValueError):
      est.fit(x=wrong_size_features, y=right_labels(), steps=1)
    with self.assertRaises(ValueError):
      est.fit(x=right_features(), y=wrong_type_labels, steps=1)
    with self.assertRaises(ValueError):
      est.fit(x=right_features(), y=wrong_size_labels, steps=1)

  def testBadInput(self):
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn)
    self.assertRaisesRegexp(ValueError,
                            'Either x or input_fn must be provided.',
                            est.fit, x=None, input_fn=None)
    self.assertRaisesRegexp(ValueError,
                            'Can not provide both input_fn and x or y',
                            est.fit, x='X', input_fn=iris_input_fn)
    self.assertRaisesRegexp(ValueError,
                            'Can not provide both input_fn and x or y',
                            est.fit, y='Y', input_fn=iris_input_fn)
    self.assertRaisesRegexp(ValueError,
                            'Can not provide both input_fn and batch_size',
                            est.fit, input_fn=iris_input_fn, batch_size=100)
    self.assertRaisesRegexp(
        ValueError, 'Inputs cannot be tensors. Please provide input_fn.',
        est.fit, x=tf.constant(1.))

  def testUntrained(self):
    boston = tf.contrib.learn.datasets.load_boston()
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn)
    with self.assertRaises(tf.contrib.learn.NotFittedError):
      _ = est.evaluate(
          x=boston.data,
          y=boston.target.astype(np.float64))
    with self.assertRaises(tf.contrib.learn.NotFittedError):
      est.predict(x=boston.data)

  def testContinueTraining(self):
    boston = tf.contrib.learn.datasets.load_boston()
    output_dir = tempfile.mkdtemp()
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn,
                                     model_dir=output_dir)
    float64_labels = boston.target.astype(np.float64)
    est.fit(x=boston.data, y=float64_labels, steps=50)
    scores = est.evaluate(
        x=boston.data,
        y=float64_labels,
        metrics={'MSE': tf.contrib.metrics.streaming_mean_squared_error})
    del est
    # Create another estimator object with the same output dir.
    est2 = tf.contrib.learn.Estimator(model_fn=linear_model_fn,
                                      model_dir=output_dir)

    # Check we can evaluate and predict.
    scores2 = est2.evaluate(
        x=boston.data,
        y=float64_labels,
        metrics={'MSE': tf.contrib.metrics.streaming_mean_squared_error})
    self.assertAllClose(scores['MSE'], scores2['MSE'])
    predictions = np.array(list(est2.predict(x=boston.data)))
    other_score = _sklearn.mean_squared_error(predictions, float64_labels)
    self.assertAllClose(scores['MSE'], other_score)

    # Check we can keep training.
    est2.fit(x=boston.data, y=float64_labels, steps=100)
    scores3 = est2.evaluate(
        x=boston.data,
        y=float64_labels,
        metrics={'MSE': tf.contrib.metrics.streaming_mean_squared_error})
    self.assertLess(scores3['MSE'], scores['MSE'])

  def testEstimatorParams(self):
    boston = tf.contrib.learn.datasets.load_boston()
    est = tf.contrib.learn.Estimator(model_fn=linear_model_params_fn,
                                     params={'learning_rate': 0.01})
    est.fit(x=boston.data, y=boston.target, steps=100)

  def testBostonAll(self):
    boston = tf.contrib.learn.datasets.load_boston()
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn)
    float64_labels = boston.target.astype(np.float64)
    est.fit(x=boston.data, y=float64_labels, steps=100)
    scores = est.evaluate(
        x=boston.data,
        y=float64_labels,
        metrics={'MSE': tf.contrib.metrics.streaming_mean_squared_error})
    predictions = np.array(list(est.predict(x=boston.data)))
    other_score = _sklearn.mean_squared_error(predictions, boston.target)
    self.assertAllClose(scores['MSE'], other_score)
    self.assertTrue('global_step' in scores)
    self.assertEqual(100, scores['global_step'])

  def testIrisAll(self):
    iris = tf.contrib.learn.datasets.load_iris()
    est = tf.contrib.learn.Estimator(model_fn=logistic_model_no_mode_fn)
    est.fit(iris.data, iris.target, steps=100)
    scores = est.evaluate(
        x=iris.data,
        y=iris.target,
        metrics={('accuracy', 'class'): tf.contrib.metrics.streaming_accuracy})
    predictions = list(est.predict(x=iris.data))
    predictions_class = list(est.predict(x=iris.data, outputs=['class']))
    self.assertEqual(len(predictions), iris.target.shape[0])
    classes_batch = np.array([p['class'] for p in predictions])
    self.assertAllClose(
        classes_batch,
        np.array([p['class'] for p in predictions_class]))
    self.assertAllClose(
        classes_batch,
        np.argmax(np.array([p['prob'] for p in predictions]), axis=1))
    other_score = _sklearn.accuracy_score(iris.target, classes_batch)
    self.assertAllClose(scores['accuracy'], other_score)
    self.assertTrue('global_step' in scores)
    self.assertEqual(100, scores['global_step'])

  def testIrisInputFn(self):
    iris = tf.contrib.learn.datasets.load_iris()
    est = tf.contrib.learn.Estimator(model_fn=logistic_model_no_mode_fn)
    est.fit(input_fn=iris_input_fn, steps=100)
    _ = est.evaluate(input_fn=iris_input_fn, steps=1)
    predictions = list(est.predict(x=iris.data))
    self.assertEqual(len(predictions), iris.target.shape[0])

  def testIrisInputFnLabelsDict(self):
    iris = tf.contrib.learn.datasets.load_iris()
    est = tf.contrib.learn.Estimator(model_fn=logistic_model_no_mode_fn)
    est.fit(input_fn=iris_input_fn_labels_dict, steps=100)
    _ = est.evaluate(
        input_fn=iris_input_fn_labels_dict,
        steps=1,
        metrics={
            'accuracy':
                metric_spec.MetricSpec(
                    metric_fn=tf.contrib.metrics.streaming_accuracy,
                    prediction_key='class',
                    label_key='labels')
        })
    predictions = list(est.predict(x=iris.data))
    self.assertEqual(len(predictions), iris.target.shape[0])

  def testIrisIterator(self):
    iris = tf.contrib.learn.datasets.load_iris()
    est = tf.contrib.learn.Estimator(model_fn=logistic_model_no_mode_fn)
    x_iter = itertools.islice(iris.data, 100)
    y_iter = itertools.islice(iris.target, 100)
    est.fit(x_iter, y_iter, steps=100)
    _ = est.evaluate(input_fn=iris_input_fn, steps=1)
    predictions = list(est.predict(x=iris.data))
    self.assertEqual(len(predictions), iris.target.shape[0])

  def testIrisIteratorArray(self):
    iris = tf.contrib.learn.datasets.load_iris()
    est = tf.contrib.learn.Estimator(model_fn=logistic_model_no_mode_fn)
    x_iter = itertools.islice(iris.data, 100)
    y_iter = (np.array(x) for x in iris.target)
    est.fit(x_iter, y_iter, steps=100)
    _ = est.evaluate(input_fn=iris_input_fn, steps=1)
    _ = six.next(est.predict(x=iris.data))['class']

  def testIrisIteratorPlainInt(self):
    iris = tf.contrib.learn.datasets.load_iris()
    est = tf.contrib.learn.Estimator(model_fn=logistic_model_no_mode_fn)
    x_iter = itertools.islice(iris.data, 100)
    y_iter = (v for v in iris.target)
    est.fit(x_iter, y_iter, steps=100)
    _ = est.evaluate(input_fn=iris_input_fn, steps=1)
    _ = six.next(est.predict(x=iris.data))['class']

  def testIrisTruncatedIterator(self):
    iris = tf.contrib.learn.datasets.load_iris()
    est = tf.contrib.learn.Estimator(model_fn=logistic_model_no_mode_fn)
    x_iter = itertools.islice(iris.data, 50)
    y_iter = ([np.int32(v)] for v in iris.target)
    est.fit(x_iter, y_iter, steps=100)

  def testTrainInputFn(self):
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn)
    est.fit(input_fn=boston_input_fn, steps=1)
    _ = est.evaluate(input_fn=boston_eval_fn, steps=1)

  def testTrainStepsIsIncremental(self):
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn)
    est.fit(input_fn=boston_input_fn, steps=10)
    self.assertEqual(10, est.get_variable_value('global_step'))
    est.fit(input_fn=boston_input_fn, steps=15)
    self.assertEqual(25, est.get_variable_value('global_step'))

  def testTrainMaxStepsIsNotIncremental(self):
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn)
    est.fit(input_fn=boston_input_fn, max_steps=10)
    self.assertEqual(10, est.get_variable_value('global_step'))
    est.fit(input_fn=boston_input_fn, max_steps=15)
    self.assertEqual(15, est.get_variable_value('global_step'))

  def testPredict(self):
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn)
    boston = tf.contrib.learn.datasets.load_boston()
    est.fit(input_fn=boston_input_fn, steps=1)
    output = list(est.predict(x=boston.data, batch_size=10))
    self.assertEqual(len(output), boston.target.shape[0])

  def testPredictInputFn(self):
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn)
    boston = tf.contrib.learn.datasets.load_boston()
    est.fit(input_fn=boston_input_fn, steps=1)
    input_fn = functools.partial(boston_input_fn, num_epochs=1)
    output = list(est.predict(input_fn=input_fn))
    self.assertEqual(len(output), boston.target.shape[0])

  def testWrongInput(self):
    def other_input_fn():
      return {'other': tf.constant([0, 0, 0])}, tf.constant([0, 0, 0])
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn)
    est.fit(input_fn=boston_input_fn, steps=1)
    with self.assertRaises(ValueError):
      est.fit(input_fn=other_input_fn, steps=1)

  def testMonitors(self):
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn)
    est.fit(input_fn=boston_input_fn,
            steps=21,
            monitors=[CheckCallsMonitor(expect_calls=21)])

  def testSummaryWriting(self):
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn)
    est.fit(input_fn=boston_input_fn, steps=200)
    est.evaluate(input_fn=boston_input_fn, steps=200)
    loss_summary = tf.contrib.testing.simple_values_from_events(
        tf.contrib.testing.latest_events(est.model_dir), ['loss'])
    self.assertEqual(1, len(loss_summary))

  def testLossInGraphCollection(self):

    class _LossCheckerHook(tf.train.SessionRunHook):

      def begin(self):
        self.loss_collection = tf.get_collection(tf.GraphKeys.LOSSES)

    hook = _LossCheckerHook()
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn)
    est.fit(input_fn=boston_input_fn, steps=200, monitors=[hook])
    self.assertTrue(hook.loss_collection)

  def test_export_returns_exported_dirname(self):
    expected = '/path/to/some_dir'
    with tf.test.mock.patch.object(estimator, 'export') as mock_export_module:
      mock_export_module._export_estimator.return_value = expected

      est = tf.contrib.learn.Estimator(model_fn=linear_model_fn)
      actual = est.export('/path/to')

    self.assertEquals(expected, actual)


class InferRealValuedColumnsTest(tf.test.TestCase):

  def testInvalidArgs(self):
    with self.assertRaisesRegexp(ValueError, 'x or input_fn must be provided'):
      tf.contrib.learn.infer_real_valued_columns_from_input(None)

    with self.assertRaisesRegexp(ValueError, 'cannot be tensors'):
      tf.contrib.learn.infer_real_valued_columns_from_input(tf.constant(1.0))

  def _assert_single_feature_column(
      self, expected_shape, expected_dtype, feature_columns):
    self.assertEqual(1, len(feature_columns))
    feature_column = feature_columns[0]
    self.assertEqual('', feature_column.name)
    self.assertEqual({
        '': tf.FixedLenFeature(shape=expected_shape, dtype=expected_dtype)
    }, feature_column.config)

  def testInt32Input(self):
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(
        np.ones(shape=[7, 8], dtype=np.int32))
    self._assert_single_feature_column([8], tf.int32, feature_columns)

  def testInt32InputFn(self):
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input_fn(
        lambda: (tf.ones(shape=[7, 8], dtype=tf.int32), None))
    self._assert_single_feature_column([8], tf.int32, feature_columns)

  def testInt64Input(self):
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(
        np.ones(shape=[7, 8], dtype=np.int64))
    self._assert_single_feature_column([8], tf.int64, feature_columns)

  def testInt64InputFn(self):
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input_fn(
        lambda: (tf.ones(shape=[7, 8], dtype=tf.int64), None))
    self._assert_single_feature_column([8], tf.int64, feature_columns)

  def testFloat32Input(self):
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(
        np.ones(shape=[7, 8], dtype=np.float32))
    self._assert_single_feature_column([8], tf.float32, feature_columns)

  def testFloat32InputFn(self):
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input_fn(
        lambda: (tf.ones(shape=[7, 8], dtype=tf.float32), None))
    self._assert_single_feature_column([8], tf.float32, feature_columns)

  def testFloat64Input(self):
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(
        np.ones(shape=[7, 8], dtype=np.float64))
    self._assert_single_feature_column([8], tf.float64, feature_columns)

  def testFloat64InputFn(self):
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input_fn(
        lambda: (tf.ones(shape=[7, 8], dtype=tf.float64), None))
    self._assert_single_feature_column([8], tf.float64, feature_columns)

  def testBoolInput(self):
    with self.assertRaisesRegexp(
        ValueError, 'on integer or non floating types are not supported'):
      tf.contrib.learn.infer_real_valued_columns_from_input(
          np.array([[False for _ in xrange(8)] for _ in xrange(7)]))

  def testBoolInputFn(self):
    with self.assertRaisesRegexp(
        ValueError, 'on integer or non floating types are not supported'):
      # pylint: disable=g-long-lambda
      tf.contrib.learn.infer_real_valued_columns_from_input_fn(
          lambda: (tf.constant(False, shape=[7, 8], dtype=tf.bool), None))

  def testStringInput(self):
    with self.assertRaisesRegexp(
        ValueError, 'on integer or non floating types are not supported'):
      # pylint: disable=g-long-lambda
      tf.contrib.learn.infer_real_valued_columns_from_input(
          np.array([['%d.0' % i for i in xrange(8)] for _ in xrange(7)]))

  def testStringInputFn(self):
    with self.assertRaisesRegexp(
        ValueError, 'on integer or non floating types are not supported'):
      # pylint: disable=g-long-lambda
      tf.contrib.learn.infer_real_valued_columns_from_input_fn(
          lambda: (
              tf.constant([['%d.0' % i for i in xrange(8)] for _ in xrange(7)]),
              None))

  def testBostonInputFn(self):
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input_fn(
        boston_input_fn)
    self._assert_single_feature_column(
        [_BOSTON_INPUT_DIM], tf.float64, feature_columns)

  def testIrisInputFn(self):
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input_fn(
        iris_input_fn)
    self._assert_single_feature_column(
        [_IRIS_INPUT_DIM], tf.float64, feature_columns)


class ReplicaDeviceSetterTest(tf.test.TestCase):

  def testVariablesAreOnPs(self):
    with tf.device(estimator._get_replica_device_setter(
        tf.contrib.learn.RunConfig(num_ps_replicas=1))):
      v = tf.Variable([1, 2])
      w = tf.Variable([2, 1])
      a = v + w
    self.assertDeviceEqual('/job:ps/task:0', v.device)
    self.assertDeviceEqual('/job:ps/task:0', v.initializer.device)
    self.assertDeviceEqual('/job:ps/task:0', w.device)
    self.assertDeviceEqual('/job:ps/task:0', w.initializer.device)
    self.assertDeviceEqual('/job:worker', a.device)

  def testVariablesAreLocal(self):
    with tf.device(estimator._get_replica_device_setter(
        tf.contrib.learn.RunConfig(num_ps_replicas=0))):
      v = tf.Variable([1, 2])
      w = tf.Variable([2, 1])
      a = v + w
    self.assertDeviceEqual('', v.device)
    self.assertDeviceEqual('', v.initializer.device)
    self.assertDeviceEqual('', w.device)
    self.assertDeviceEqual('', w.initializer.device)
    self.assertDeviceEqual('', a.device)

  def testMutableHashTableIsOnPs(self):
    with tf.device(estimator._get_replica_device_setter(
        tf.contrib.learn.RunConfig(num_ps_replicas=1))):
      default_val = tf.constant([-1, -1], tf.int64)
      table = tf.contrib.lookup.MutableHashTable(tf.string,
                                                 tf.int64,
                                                 default_val)
      input_string = tf.constant(['brain', 'salad', 'tank'])
      output = table.lookup(input_string)
    self.assertDeviceEqual('/job:ps/task:0', table._table_ref.device)
    self.assertDeviceEqual('/job:ps/task:0', output.device)

  def testMutableHashTableIsLocal(self):
    with tf.device(estimator._get_replica_device_setter(
        tf.contrib.learn.RunConfig(num_ps_replicas=0))):
      default_val = tf.constant([-1, -1], tf.int64)
      table = tf.contrib.lookup.MutableHashTable(tf.string,
                                                 tf.int64,
                                                 default_val)
      input_string = tf.constant(['brain', 'salad', 'tank'])
      output = table.lookup(input_string)
    self.assertDeviceEqual('', table._table_ref.device)
    self.assertDeviceEqual('', output.device)

  def testTaskIsSetOnWorkerWhenJobNameIsSet(self):
    with tf.device(
        estimator._get_replica_device_setter(
            tf.contrib.learn.RunConfig(
                num_ps_replicas=1, job_name='worker', task=3))):
      v = tf.Variable([1, 2])
      w = tf.Variable([2, 1])
      a = v + w
    self.assertDeviceEqual('/job:ps/task:0', v.device)
    self.assertDeviceEqual('/job:ps/task:0', v.initializer.device)
    self.assertDeviceEqual('/job:ps/task:0', w.device)
    self.assertDeviceEqual('/job:ps/task:0', w.initializer.device)
    self.assertDeviceEqual('/job:worker/task:3', a.device)


if __name__ == '__main__':
  tf.test.main()
