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
"""Logistic regression (aka binary classifier) class.

This defines some useful basic metrics for using logistic regression to classify
a binary event (0 vs 1).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import metrics as metrics_lib
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.python.ops import math_ops


def _labels_streaming_mean(unused_predictions, labels):
  return metrics_lib.streaming_mean(labels)


def _predictions_streaming_mean(predictions, unused_labels):
  return metrics_lib.streaming_mean(predictions)


def _make_streaming_with_threshold(streaming_metrics_fn, threshold):

  def _streaming_metrics(predictions, labels):
    return streaming_metrics_fn(predictions=math_ops.to_float(
        math_ops.greater_equal(predictions, threshold)),
                                labels=labels)

  return _streaming_metrics


class LogisticRegressor(estimator.Estimator):
  """Logistic regression Estimator for binary classification.

  This class provides a basic Estimator with some additional metrics for custom
  binary classification models, including AUC, precision/recall and accuracy.

  Example:

  ```python
    # See tf.contrib.learn.Estimator(...) for details on model_fn structure
    def my_model_fn(...):
      pass

    estimator = LogisticRegressor(model_fn=my_model_fn)

    # Input builders
    def input_fn_train:
      pass

    estimator.fit(input_fn=input_fn_train)
    estimator.predict(x=x)
  ```
  """

  def __init__(self, model_fn, thresholds=None, model_dir=None, config=None,
               feature_engineering_fn=None):
    """Initializes a LogisticRegressor.

    Args:
      model_fn: Model function. See superclass Estimator for more details. This
        expects the returned predictions to be probabilities in [0.0, 1.0].
      thresholds: List of floating point thresholds to use for accuracy,
        precision, and recall metrics. If `None`, defaults to `[0.5]`.
      model_dir: Directory to save model parameters, graphs, etc. This can also
        be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      config: A RunConfig configuration object.
      feature_engineering_fn: Feature engineering function. Takes features and
                        labels which are the output of `input_fn` and
                        returns features and labels which will be fed
                        into the model.
    """
    if thresholds is None:
      thresholds = [0.5]
    self._thresholds = thresholds
    super(LogisticRegressor, self).__init__(
        model_fn=model_fn,
        model_dir=model_dir,
        config=config,
        feature_engineering_fn=feature_engineering_fn)

  # TODO(zakaria): use target column.

  # Metrics string keys.
  AUC = "auc"
  PREDICTION_MEAN = "labels/prediction_mean"
  TARGET_MEAN = "labels/actual_label_mean"
  ACCURACY_BASELINE = "accuracy/baseline_label_mean"
  ACCURACY_MEAN = "accuracy/threshold_%f_mean"
  PRECISION_MEAN = "precision/positive_threshold_%f_mean"
  RECALL_MEAN = "recall/positive_threshold_%f_mean"

  @classmethod
  def get_default_metrics(cls, thresholds=None):
    """Returns a dictionary of basic metrics for logistic regression.

    Args:
      thresholds: List of floating point thresholds to use for accuracy,
        precision, and recall metrics. If None, defaults to [0.5].

    Returns:
      Dictionary mapping metrics string names to metrics functions.
    """
    if thresholds is None:
      thresholds = [0.5]

    metrics = {}
    metrics[cls.PREDICTION_MEAN] = _predictions_streaming_mean
    metrics[cls.TARGET_MEAN] = _labels_streaming_mean
    # Also include the streaming mean of the label as an accuracy baseline, as
    # a reminder to users.
    metrics[cls.ACCURACY_BASELINE] = _labels_streaming_mean

    metrics[cls.AUC] = metrics_lib.streaming_auc

    for threshold in thresholds:
      metrics[cls.ACCURACY_MEAN % threshold] = _make_streaming_with_threshold(
          metrics_lib.streaming_accuracy, threshold)
      # Precision for positive examples.
      metrics[cls.PRECISION_MEAN % threshold] = _make_streaming_with_threshold(
          metrics_lib.streaming_precision, threshold)
      # Recall for positive examples.
      metrics[cls.RECALL_MEAN % threshold] = _make_streaming_with_threshold(
          metrics_lib.streaming_recall, threshold)

    return metrics

  def evaluate(self,
               x=None,
               y=None,
               input_fn=None,
               feed_fn=None,
               batch_size=None,
               steps=None,
               metrics=None,
               name=None,
               checkpoint_path=None):
    """Evaluates given model with provided evaluation data.

    See superclass Estimator for more details.

    Args:
      x: features.
      y: labels (must be 0 or 1).
      input_fn: Input function.
      feed_fn: Function creating a feed dict every time it is called.
      batch_size: minibatch size to use on the input.
      steps: Number of steps for which to evaluate model.
      metrics: Dict of metric ops to run. If None, the default metrics are used.
      name: Name of the evaluation.
      checkpoint_path: A specific checkpoint to use. By default, use the latest
        checkpoint in the `model_dir`.

    Returns:
      Returns `dict` with evaluation results.
    """
    metrics = metrics or self.get_default_metrics(thresholds=self._thresholds)
    return super(LogisticRegressor, self).evaluate(
        x=x,
        y=y,
        input_fn=input_fn,
        batch_size=batch_size,
        steps=steps,
        metrics=metrics,
        name=name,
        checkpoint_path=checkpoint_path)
