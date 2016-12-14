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
"""The metric spec class to flexibly connect models and metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import tf_logging as logging


class MetricSpec(object):
  """MetricSpec connects a model to metric functions.

  The MetricSpec class contains all information necessary to connect the
  output of a `model_fn` to the metrics (usually, streaming metrics) that are
  used in evaluation.

  It is passed in the `metrics` argument of `Estimator.evaluate`. The
  `Estimator` then knows which predictions, labels, and weight to use to call a
  given metric function.

  When building the ops to run in evaluation, `Estimator` will call
  `create_metric_ops`, which will connect the given `metric_fn` to the model
  as detailed in the docstring for `create_metric_ops`, and return the metric.

  Example:

  Assuming a model has an input function which returns inputs containing
  (among other things) a tensor with key "input_key", and a labels dictionary
  containing "label_key". Let's assume that the `model_fn` for this model
  returns a prediction with key "prediction_key".

  In order to compute the accuracy of the "prediction_key" prediction, we
  would add

  ```
  "prediction accuracy": MetricSpec(metric_fn=prediction_accuracy_fn,
                                    prediction_key="prediction_key",
                                    label_key="label_key")
  ```

  to the metrics argument to `evaluate`. `prediction_accuracy_fn` can be either
  a predefined function in metric_ops (e.g., `streaming_accuracy`) or a custom
  function you define.

  If we would like the accuracy to be weighted by "input_key", we can add that
  as the `weight_key` argument.

  ```
  "prediction accuracy": MetricSpec(metric_fn=prediction_accuracy_fn,
                                    prediction_key="prediction_key",
                                    label_key="label_key",
                                    weight_key="input_key")
  ```

  An end-to-end example is as follows:

  ```
  estimator = tf.contrib.learn.Estimator(...)
  estimator.fit(...)
  _ = estimator.evaluate(
      input_fn=input_fn,
      steps=1,
      metrics={
          'prediction accuracy':
              metric_spec.MetricSpec(
                  metric_fn=prediction_accuracy_fn,
                  prediction_key="prediction_key",
                  label_key="label_key")
      })
  ```

  """

  def __init__(self,
               metric_fn,
               prediction_key=None,
               label_key=None,
               weight_key=None):
    """Constructor.

    Creates a MetricSpec.

    Args:
      metric_fn: A function to use as a metric. Must accept `predictions`,
        `labels` and optionally, `weights` tensors as inputs, and must return
        either a single tensor which is interpreted as a value of this metric,
        or a pair `(value_op, update_op)`, where value_op is the op to call to
        obtain the value of the metric, and update_op should be evaluated for
        each batch in order to update internal state.
      prediction_key: The key for a tensor in the `predictions` dict (output
        from the `model_fn`) to use as the `predictions` input to the
        `metric_fn`. Optional. If `None`, the `model_fn` must return a single
        tensor or a dict with only a single entry as `predictions`.
      label_key: The key for a tensor in the `labels` dict (output from the
        `input_fn`) to use as the `labels` input to the `metric_fn`.
        Optional. If `None`, the `input_fn` must return a single tensor or a
        dict with only a single entry as `labels`.
      weight_key: The key for a tensor in the `inputs` dict (output from the
        `input_fn`) to use as the `weights` input to the `metric_fn`.
        Optional. If `None`, no weights will be passed to the `metric_fn`.
    """
    self._metric_fn = metric_fn
    self._prediction_key = prediction_key
    self._label_key = label_key
    self._weight_key = weight_key

  @property
  def prediction_key(self):
    return self._prediction_key

  @property
  def label_key(self):
    return self._label_key

  @property
  def weight_key(self):
    return self._weight_key

  @property
  def metric_fn(self):
    return self._metric_fn

  def __str__(self):
    if hasattr(self.metric_fn, '__name__'):
      fn_name = self.metric_fn.__name__
    elif (hasattr(self.metric_fn, 'func') and
          hasattr(self.metric_fn.func, '__name__')):
      fn_name = self.metric_fn.func.__name__  # If it's a functools.partial.
    else:
      fn_name = '%s' % self.metric_fn

    return ('MetricSpec(metric_fn=%s, ' % fn_name +
            'prediction_key=%s, ' % self.prediction_key +
            'label_key=%s, ' % self.label_key +
            'weight_key=%s)' % self.weight_key
           )

  def create_metric_ops(self, inputs, labels, predictions):
    """Connect our `metric_fn` to the specified members of the given dicts.

    This function will call the `metric_fn` given in our constructor as follows:

    ```
      metric_fn(predictions[self.prediction_key],
                labels[self.label_key],
                weights=weights[self.weight_key])
    ```

    And returns the result. The `weights` argument is only passed if
    `self.weight_key` is not `None`.

    `predictions` and `labels` may be single tensors as well as dicts. If
    `predictions` is a single tensor, `self.prediction_key` must be `None`. If
    `predictions` is a single element dict, `self.prediction_key` is allowed to
    be `None`. Conversely, if `labels` is a single tensor, `self.label_key` must
    be `None`. If `labels` is a single element dict, `self.label_key` is allowed
    to be `None`.

    Args:
      inputs: A dict of inputs produced by the `input_fn`
      labels: A dict of labels or a single label tensor produced by the
        `input_fn`.
      predictions: A dict of predictions or a single tensor produced by the
        `model_fn`.

    Returns:
      The result of calling `metric_fn`.

    Raises:
      ValueError: If `predictions` or `labels` is a single `Tensor` and
        `self.prediction_key` or `self.label_key` is not `None`; or if
        `self.label_key` is `None` but `labels` is a dict with more than one
        element, or if `self.prediction_key` is `None but `predictions` is a
        dict with more than one element.
    """
    def _get_dict(name, dict_or_tensor, key):
      """Get a single tensor or an element of a dict or raise ValueError."""
      if key:
        if not isinstance(dict_or_tensor, dict):
          raise ValueError('MetricSpec with ' + name + '_key specified'
                           ' requires ' +
                           name + 's dict, got %s' % dict_or_tensor)
        if key not in dict_or_tensor:
          raise KeyError(
              'Key \'%s\' missing from %s.' % (key, dict_or_tensor.keys()))
        return dict_or_tensor[key]
      else:
        if isinstance(dict_or_tensor, dict):
          if len(dict_or_tensor) != 1:
            raise ValueError('MetricSpec without specified ' + name + '_key'
                             ' requires ' + name + 's tensor or single element'
                             ' dict, got %s' % dict_or_tensor)
          return dict_or_tensor.values()[0]
        else:
          return dict_or_tensor

    # Get the predictions
    prediction = _get_dict('prediction', predictions, self.prediction_key)

    # Get the labels
    label = _get_dict('label', labels, self.label_key)

    try:
      if self.weight_key:
        return self.metric_fn(prediction, label,
                              weights=inputs[self.weight_key])
      else:
        return self.metric_fn(prediction, label)
    except:  # pylint: disable=bare-except
      logging.error('Could not create metric ops for %s.' % self)
      raise
