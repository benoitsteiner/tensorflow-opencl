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
"""Tests for MetricSpec."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags"):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
from tensorflow.python.platform import test


def test_metric(predictions, labels, weights=None):
  return predictions, labels, weights


class MetricSpecTest(test.TestCase):

  def test_create_metric_ops(self):
    features = {"feature1": "feature1_tensor", "feature2": "feature2_tensor"}
    labels = {"label1": "label1_tensor", "label2": "label2_tensor"}
    predictions = {"pred1": "pred1_tensor", "pred2": "pred2_tensor"}

    passed = MetricSpec(
        metric_fn=test_metric,
        prediction_key="pred1",
        label_key="label1",
        weight_key="feature2").create_metric_ops(features, labels, predictions)

    self.assertEqual(passed[0], "pred1_tensor")
    self.assertEqual(passed[1], "label1_tensor")
    self.assertEqual(passed[2], "feature2_tensor")

  def test_no_weight(self):
    features = {"feature1": "feature1_tensor", "feature2": "feature2_tensor"}
    labels = {"label1": "label1_tensor", "label2": "label2_tensor"}
    predictions = {"pred1": "pred1_tensor", "pred2": "pred2_tensor"}

    passed = MetricSpec(
        metric_fn=test_metric, prediction_key="pred1",
        label_key="label1").create_metric_ops(features, labels, predictions)

    self.assertEqual(passed[0], "pred1_tensor")
    self.assertEqual(passed[1], "label1_tensor")
    self.assertEqual(passed[2], None)

  def test_fail_no_prediction(self):
    features = {"feature1": "feature1_tensor", "feature2": "feature2_tensor"}
    labels = {"label1": "label1_tensor", "label2": "label2_tensor"}
    predictions = {"pred1": "pred1_tensor", "pred2": "pred2_tensor"}

    self.assertRaisesRegexp(
        ValueError,
        "MetricSpec without specified prediction_key "
        "requires predictions tensor or single element "
        "dict, got",
        MetricSpec(
            metric_fn=test_metric, label_key="label1",
            weight_key="feature2").create_metric_ops,
        features,
        labels,
        predictions)

  def test_fail_no_label(self):
    features = {"feature1": "feature1_tensor", "feature2": "feature2_tensor"}
    labels = {"label1": "label1_tensor", "label2": "label2_tensor"}
    predictions = {"pred1": "pred1_tensor", "pred2": "pred2_tensor"}

    self.assertRaisesRegexp(
        ValueError,
        "MetricSpec without specified label_key requires "
        "labels tensor or single element dict, got",
        MetricSpec(
            metric_fn=test_metric,
            prediction_key="pred1",
            weight_key="feature2").create_metric_ops,
        features,
        labels,
        predictions)

  def test_single_prediction(self):
    features = {"feature1": "feature1_tensor", "feature2": "feature2_tensor"}
    labels = {"label1": "label1_tensor", "label2": "label2_tensor"}
    predictions = "pred1_tensor"

    passed = MetricSpec(
        metric_fn=test_metric, label_key="label1",
        weight_key="feature2").create_metric_ops(features, labels, predictions)

    self.assertEqual(passed[0], "pred1_tensor")
    self.assertEqual(passed[1], "label1_tensor")
    self.assertEqual(passed[2], "feature2_tensor")

  def test_single_label(self):
    features = {"feature1": "feature1_tensor", "feature2": "feature2_tensor"}
    labels = "label1_tensor"
    predictions = {"pred1": "pred1_tensor", "pred2": "pred2_tensor"}

    passed = MetricSpec(
        metric_fn=test_metric, prediction_key="pred1",
        weight_key="feature2").create_metric_ops(features, labels, predictions)

    self.assertEqual(passed[0], "pred1_tensor")
    self.assertEqual(passed[1], "label1_tensor")
    self.assertEqual(passed[2], "feature2_tensor")

  def test_fail_single_prediction(self):
    features = {"feature1": "feature1_tensor", "feature2": "feature2_tensor"}
    labels = {"label1": "label1_tensor", "label2": "label2_tensor"}
    predictions = "pred1_tensor"

    self.assertRaisesRegexp(
        ValueError,
        "MetricSpec with prediction_key specified requires "
        "predictions dict, got",
        MetricSpec(
            metric_fn=test_metric,
            prediction_key="pred1",
            label_key="label1",
            weight_key="feature2").create_metric_ops,
        features,
        labels,
        predictions)

  def test_fail_single_label(self):
    features = {"feature1": "feature1_tensor", "feature2": "feature2_tensor"}
    labels = "label1_tensor"
    predictions = {"pred1": "pred1_tensor", "pred2": "pred2_tensor"}

    self.assertRaisesRegexp(
        ValueError,
        "MetricSpec with label_key specified requires "
        "labels dict, got",
        MetricSpec(
            metric_fn=test_metric,
            prediction_key="pred1",
            label_key="label1",
            weight_key="feature2").create_metric_ops,
        features,
        labels,
        predictions)

  def test_str(self):
    metric_spec = MetricSpec(
        metric_fn=test_metric,
        label_key="label1",
        prediction_key="pred1",
        weight_key="feature2")
    string = str(metric_spec)
    self.assertIn("test_metric", string)
    self.assertIn("label1", string)
    self.assertIn("pred1", string)
    self.assertIn("feature2", string)

  def test_partial_str(self):

    def custom_metric(predictions, labels, stuff, weights=None):
      return predictions, labels, weights, stuff

    partial_metric = functools.partial(custom_metric, stuff=5)
    metric_spec = MetricSpec(
        metric_fn=partial_metric,
        label_key="label1",
        prediction_key="pred1",
        weight_key="feature2")
    self.assertIn("custom_metric", str(metric_spec))

  def test_partial(self):
    features = {"feature1": "feature1_tensor", "feature2": "feature2_tensor"}
    labels = {"label1": "label1_tensor"}
    predictions = {"pred1": "pred1_tensor", "pred2": "pred2_tensor"}

    def custom_metric(predictions, labels, stuff, weights=None):
      if stuff:
        return predictions, labels, weights
      else:
        raise ValueError("Nooooo")

    partial_metric = functools.partial(custom_metric, stuff=5)
    passed = MetricSpec(
        metric_fn=partial_metric,
        label_key="label1",
        prediction_key="pred1",
        weight_key="feature2").create_metric_ops(features, labels, predictions)
    self.assertEqual(passed[0], "pred1_tensor")
    self.assertEqual(passed[1], "label1_tensor")
    self.assertEqual(passed[2], "feature2_tensor")

    broken_partial_metric = functools.partial(custom_metric, stuff=0)
    self.assertRaisesRegexp(
        ValueError,
        "Nooooo",
        MetricSpec(
            metric_fn=broken_partial_metric,
            prediction_key="pred1",
            label_key="label1",
            weight_key="feature2").create_metric_ops,
        features,
        labels,
        predictions)


if __name__ == "__main__":
  test.main()
