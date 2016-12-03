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
"""Tests for tensorflow.ops.parsing_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import tensorflow as tf

from google.protobuf import json_format
from tensorflow.python.framework import tensor_util

# Helpers for creating Example objects
example = tf.train.Example
feature = tf.train.Feature
features = lambda d: tf.train.Features(feature=d)
bytes_feature = lambda v: feature(bytes_list=tf.train.BytesList(value=v))
int64_feature = lambda v: feature(int64_list=tf.train.Int64List(value=v))
float_feature = lambda v: feature(float_list=tf.train.FloatList(value=v))
# Helpers for creating SequenceExample objects
feature_list = lambda l: tf.train.FeatureList(feature=l)
feature_lists = lambda d: tf.train.FeatureLists(feature_list=d)
sequence_example = tf.train.SequenceExample


def flatten(list_of_lists):
  """Flatten one level of nesting."""
  return itertools.chain.from_iterable(list_of_lists)


def flatten_values_tensors_or_sparse(tensors_list):
  """Flatten each SparseTensor object into 3 Tensors for session.run()."""
  return list(
      flatten([[v.indices, v.values, v.shape] if isinstance(v, tf.SparseTensor)
               else [v] for v in tensors_list]))


def _compare_output_to_expected(tester, dict_tensors, expected_tensors,
                                flat_output):
  tester.assertEqual(set(dict_tensors.keys()), set(expected_tensors.keys()))

  i = 0  # Index into the flattened output of session.run()
  for k, v in dict_tensors.items():
    expected_v = expected_tensors[k]
    tf.logging.info("Comparing key: %s", k)
    if isinstance(v, tf.SparseTensor):
      # Three outputs for SparseTensor : indices, values, shape.
      tester.assertEqual([k, len(expected_v)], [k, 3])
      tester.assertAllEqual(expected_v[0], flat_output[i])
      tester.assertAllEqual(expected_v[1], flat_output[i + 1])
      tester.assertAllEqual(expected_v[2], flat_output[i + 2])
      i += 3
    else:
      # One output for standard Tensor.
      tester.assertAllEqual(expected_v, flat_output[i])
      i += 1


class ParseExampleTest(tf.test.TestCase):

  def _test(self, kwargs, expected_values=None, expected_err=None):
    with self.test_session() as sess:
      if expected_err:
        with self.assertRaisesWithPredicateMatch(
            expected_err[0], expected_err[1]):
          out = tf.parse_example(**kwargs)
          sess.run(flatten_values_tensors_or_sparse(out.values()))
      else:
        # Returns dict w/ Tensors and SparseTensors.
        out = tf.parse_example(**kwargs)
        result = flatten_values_tensors_or_sparse(out.values())
        # Check values.
        tf_result = sess.run(result)
        _compare_output_to_expected(self, out, expected_values, tf_result)

      # Check shapes; if serialized is a Tensor we need its size to
      # properly check.
      serialized = kwargs["serialized"]
      batch_size = (serialized.eval().size if isinstance(serialized, tf.Tensor)
                    else np.asarray(serialized).size)
      for k, f in kwargs["features"].items():
        if isinstance(f, tf.FixedLenFeature) and f.shape is not None:
          self.assertEqual(
              tuple(out[k].get_shape().as_list()), (batch_size,) + f.shape)
        elif isinstance(f, tf.VarLenFeature):
          self.assertEqual(
              tuple(out[k].indices.get_shape().as_list()), (None, 2))
          self.assertEqual(tuple(out[k].values.get_shape().as_list()), (None,))
          self.assertEqual(tuple(out[k].shape.get_shape().as_list()), (2,))

  def testEmptySerializedWithAllDefaults(self):
    sparse_name = "st_a"
    a_name = "a"
    b_name = "b"
    c_name = "c:has_a_tricky_name"
    a_default = [0, 42, 0]
    b_default = np.random.rand(3, 3).astype(bytes)
    c_default = np.random.rand(2).astype(np.float32)

    expected_st_a = (  # indices, values, shape
        np.empty(
            (0, 2), dtype=np.int64),  # indices
        np.empty(
            (0,), dtype=np.int64),  # sp_a is DT_INT64
        np.array(
            [2, 0], dtype=np.int64))  # batch == 2, max_elems = 0

    expected_output = {
        sparse_name: expected_st_a,
        a_name: np.array(2 * [[a_default]]),
        b_name: np.array(2 * [b_default]),
        c_name: np.array(2 * [c_default]),
    }

    self._test(
        {
            "example_names": np.empty(
                (0,), dtype=bytes),
            "serialized": tf.convert_to_tensor(["", ""]),
            "features": {
                sparse_name: tf.VarLenFeature(tf.int64),
                a_name: tf.FixedLenFeature(
                    (1, 3), tf.int64, default_value=a_default),
                b_name: tf.FixedLenFeature(
                    (3, 3), tf.string, default_value=b_default),
                c_name: tf.FixedLenFeature(
                    (2,), tf.float32, default_value=c_default),
            }
        },
        expected_output)

  def testEmptySerializedWithoutDefaultsShouldFail(self):
    input_features = {
        "st_a": tf.VarLenFeature(tf.int64),
        "a": tf.FixedLenFeature(
            (1, 3), tf.int64, default_value=[0, 42, 0]),
        "b": tf.FixedLenFeature(
            (3, 3),
            tf.string,
            default_value=np.random.rand(3, 3).astype(bytes)),
        # Feature "c" is missing a default, this gap will cause failure.
        "c": tf.FixedLenFeature(
            (2,), dtype=tf.float32),
    }

    # Edge case where the key is there but the feature value is empty
    original = example(features=features({
        "c": feature()
    }))
    self._test(
        {
            "example_names": ["in1"],
            "serialized": [original.SerializeToString()],
            "features": input_features,
        },
        expected_err=(tf.OpError, "Name: in1, Feature: c is required"))

    # Standard case of missing key and value.
    self._test(
        {
            "example_names": ["in1", "in2"],
            "serialized": ["", ""],
            "features": input_features,
        },
        expected_err=(tf.OpError, "Name: in1, Feature: c is required"))

  def testDenseNotMatchingShapeShouldFail(self):
    original = [
        example(features=features({
            "a": float_feature([1, 1, 3]),
        })), example(features=features({
            "a": float_feature([-1, -1]),
        }))
    ]

    names = ["passing", "failing"]
    serialized = [m.SerializeToString() for m in original]

    self._test(
        {
            "example_names": names,
            "serialized": tf.convert_to_tensor(serialized),
            "features": {"a": tf.FixedLenFeature((1, 3), tf.float32)}
        },
        expected_err=(tf.OpError,
                      "Name: failing, Key: a, Index: 1.  Number of float val"))

  def testDenseDefaultNoShapeShouldFail(self):
    original = [example(features=features({"a": float_feature([1, 1, 3]),})),]

    serialized = [m.SerializeToString() for m in original]

    self._test(
        {
            "example_names": ["failing"],
            "serialized": tf.convert_to_tensor(serialized),
            "features": {"a": tf.FixedLenFeature(None, tf.float32)}
        },
        expected_err=(ValueError, "Missing shape for feature a"))

  def testSerializedContainingSparse(self):
    original = [
        example(features=features({
            "st_c": float_feature([3, 4])
        })),
        example(features=features({
            "st_c": float_feature([]),  # empty float list
        })),
        example(features=features({
            "st_d": feature(),  # feature with nothing in it
        })),
        example(features=features({
            "st_c": float_feature([1, 2, -1]),
            "st_d": bytes_feature([b"hi"])
        }))
    ]

    serialized = [m.SerializeToString() for m in original]

    expected_st_c = (  # indices, values, shape
        np.array(
            [[0, 0], [0, 1], [3, 0], [3, 1], [3, 2]], dtype=np.int64), np.array(
                [3.0, 4.0, 1.0, 2.0, -1.0], dtype=np.float32), np.array(
                    [4, 3], dtype=np.int64))  # batch == 2, max_elems = 3

    expected_st_d = (  # indices, values, shape
        np.array(
            [[3, 0]], dtype=np.int64), np.array(
                ["hi"], dtype=bytes), np.array(
                    [4, 1], dtype=np.int64))  # batch == 2, max_elems = 1

    expected_output = {
        "st_c": expected_st_c,
        "st_d": expected_st_d,
    }

    self._test({
        "serialized": tf.convert_to_tensor(serialized),
        "features": {
            "st_c": tf.VarLenFeature(tf.float32),
            "st_d": tf.VarLenFeature(tf.string)
        }
    }, expected_output)

  def testSerializedContainingSparseFeature(self):
    original = [
        example(features=features({
            "val": float_feature([3, 4]),
            "idx": int64_feature([5, 10])
        })),
        example(features=features({
            "val": float_feature([]),  # empty float list
            "idx": int64_feature([])
        })),
        example(features=features({
            "val": feature(),  # feature with nothing in it
                               # missing idx feature
        })),
        example(features=features({
            "val": float_feature([1, 2, -1]),
            "idx": int64_feature([0, 9, 3])  # unsorted
        }))
    ]

    serialized = [m.SerializeToString() for m in original]

    expected_sp = (  # indices, values, shape
        np.array([[0, 5], [0, 10], [3, 0], [3, 3], [3, 9]], dtype=np.int64),
        np.array([3.0, 4.0, 1.0, -1.0, 2.0], dtype=np.float32),
        np.array([4, 13], dtype=np.int64))  # batch == 4, max_elems = 13

    expected_output = {
        "sp": expected_sp,
    }

    self._test({
        "serialized": tf.convert_to_tensor(serialized),
        "features": {
            "sp": tf.SparseFeature("idx", "val", tf.float32, 13)
        }
    }, expected_output)

  def testSerializedContainingSparseFeatureReuse(self):
    original = [
        example(features=features({
            "val1": float_feature([3, 4]),
            "val2": float_feature([5, 6]),
            "idx": int64_feature([5, 10])
        })),
        example(features=features({
            "val1": float_feature([]),  # empty float list
            "idx": int64_feature([])
        })),
    ]

    serialized = [m.SerializeToString() for m in original]

    expected_sp1 = (  # indices, values, shape
        np.array([[0, 5], [0, 10]], dtype=np.int64),
        np.array([3.0, 4.0], dtype=np.float32),
        np.array([2, 13], dtype=np.int64))  # batch == 2, max_elems = 13

    expected_sp2 = (  # indices, values, shape
        np.array([[0, 5], [0, 10]], dtype=np.int64),
        np.array([5.0, 6.0], dtype=np.float32),
        np.array([2, 7], dtype=np.int64))  # batch == 2, max_elems = 13

    expected_output = {
        "sp1": expected_sp1,
        "sp2": expected_sp2,
    }

    self._test({
        "serialized": tf.convert_to_tensor(serialized),
        "features": {
            "sp1": tf.SparseFeature("idx", "val1", tf.float32, 13),
            "sp2": tf.SparseFeature("idx", "val2", tf.float32, 7)
        }
    }, expected_output)

  def testSerializedContainingDense(self):
    aname = "a"
    bname = "b*has+a:tricky_name"
    original = [
        example(features=features({
            aname: float_feature([1, 1]),
            bname: bytes_feature([b"b0_str"]),
        })), example(features=features({
            aname: float_feature([-1, -1]),
            bname: bytes_feature([b"b1"]),
        }))
    ]

    serialized = [m.SerializeToString() for m in original]

    expected_output = {
        aname: np.array(
            [[1, 1], [-1, -1]], dtype=np.float32).reshape(2, 1, 2, 1),
        bname: np.array(
            ["b0_str", "b1"], dtype=bytes).reshape(2, 1, 1, 1, 1),
    }

    # No defaults, values required
    self._test(
        {
            "serialized": tf.convert_to_tensor(serialized),
            "features": {
                aname: tf.FixedLenFeature(
                    (1, 2, 1), dtype=tf.float32),
                bname: tf.FixedLenFeature(
                    (1, 1, 1, 1), dtype=tf.string),
            }
        },
        expected_output)

  # This test is identical as the previous one except
  # for the creation of 'serialized'.
  def testSerializedContainingDenseWithConcat(self):
    aname = "a"
    bname = "b*has+a:tricky_name"
    # TODO(lew): Feature appearing twice should be an error in future.
    original = [
        (
            example(features=features({
                aname: float_feature([10, 10]),
            })),
            example(features=features({
                aname: float_feature([1, 1]),
                bname: bytes_feature([b"b0_str"]),
            }))
        ),
        (
            example(features=features({
                bname: bytes_feature([b"b100"]),
            })),
            example(features=features({
                aname: float_feature([-1, -1]),
                bname: bytes_feature([b"b1"]),
            })),
        ),
    ]

    serialized = [
        m.SerializeToString() + n.SerializeToString() for (m, n) in original]

    expected_output = {
        aname: np.array(
            [[1, 1], [-1, -1]], dtype=np.float32).reshape(2, 1, 2, 1),
        bname: np.array(
            ["b0_str", "b1"], dtype=bytes).reshape(2, 1, 1, 1, 1),
    }

    # No defaults, values required
    self._test(
        {
            "serialized": tf.convert_to_tensor(serialized),
            "features": {
                aname: tf.FixedLenFeature(
                    (1, 2, 1), dtype=tf.float32),
                bname: tf.FixedLenFeature(
                    (1, 1, 1, 1), dtype=tf.string),
            }
        },
        expected_output)

  def testSerializedContainingDenseScalar(self):
    original = [
        example(features=features({
            "a": float_feature([1]),
        })), example(features=features({}))
    ]

    serialized = [m.SerializeToString() for m in original]

    expected_output = {
        "a": np.array([[1], [-1]], dtype=np.float32)  # 2x1 (column vector)
    }

    self._test(
        {
            "serialized": tf.convert_to_tensor(serialized),
            "features": {
                "a": tf.FixedLenFeature(
                    (1,), dtype=tf.float32, default_value=-1),
            }
        },
        expected_output)

  def testSerializedContainingDenseWithDefaults(self):
    original = [
        example(features=features({
            "a": float_feature([1, 1]),
        })),
        example(features=features({
            "b": bytes_feature([b"b1"]),
        })),
        example(features=features({
            "b": feature()
        })),
    ]

    serialized = [m.SerializeToString() for m in original]

    expected_output = {
        "a": np.array(
            [[1, 1], [3, -3], [3, -3]], dtype=np.float32).reshape(3, 1, 2, 1),
        "b": np.array(
            ["tmp_str", "b1", "tmp_str"], dtype=bytes).reshape(3, 1, 1, 1, 1),
    }

    self._test(
        {
            "serialized": tf.convert_to_tensor(serialized),
            "features": {
                "a": tf.FixedLenFeature(
                    (1, 2, 1), dtype=tf.float32, default_value=[3.0, -3.0]),
                "b": tf.FixedLenFeature(
                    (1, 1, 1, 1), dtype=tf.string, default_value="tmp_str"),
            }
        },
        expected_output)

  def testSerializedContainingSparseAndSparseFeatureAndDenseWithNoDefault(self):
    expected_st_a = (  # indices, values, shape
        np.empty(
            (0, 2), dtype=np.int64),  # indices
        np.empty(
            (0,), dtype=np.int64),  # sp_a is DT_INT64
        np.array(
            [2, 0], dtype=np.int64))  # batch == 2, max_elems = 0
    expected_sp = (  # indices, values, shape
        np.array([[0, 0], [0, 3], [1, 7]], dtype=np.int64),
        np.array(["a", "b", "c"], dtype="|S"),
        np.array([2, 13], dtype=np.int64))  # batch == 4, max_elems = 13

    original = [
        example(features=features({
            "c": float_feature([3, 4]),
            "val": bytes_feature([b"a", b"b"]),
            "idx": int64_feature([0, 3])
        })), example(features=features({
            "c": float_feature([1, 2]),
            "val": bytes_feature([b"c"]),
            "idx": int64_feature([7])
        }))
    ]

    names = ["in1", "in2"]
    serialized = [m.SerializeToString() for m in original]

    a_default = [1, 2, 3]
    b_default = np.random.rand(3, 3).astype(bytes)
    expected_output = {
        "st_a": expected_st_a,
        "sp": expected_sp,
        "a": np.array(2 * [[a_default]]),
        "b": np.array(2 * [b_default]),
        "c": np.array(
            [[3, 4], [1, 2]], dtype=np.float32),
    }

    self._test(
        {
            "example_names": names,
            "serialized": tf.convert_to_tensor(serialized),
            "features": {
                "st_a": tf.VarLenFeature(tf.int64),
                "sp": tf.SparseFeature("idx", "val", tf.string, 13),
                "a": tf.FixedLenFeature(
                    (1, 3), tf.int64, default_value=a_default),
                "b": tf.FixedLenFeature(
                    (3, 3), tf.string, default_value=b_default),
                # Feature "c" must be provided, since it has no default_value.
                "c": tf.FixedLenFeature((2,), tf.float32),
            }
        },
        expected_output)

  def testSerializedContainingSparseAndSparseFeatureWithReuse(self):
    expected_idx = (  # indices, values, shape
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64),
        np.array([0, 3, 7, 1]),
        np.array([2, 2], dtype=np.int64))  # batch == 4, max_elems = 2

    expected_sp = (  # indices, values, shape
        np.array([[0, 0], [0, 3], [1, 1], [1, 7]], dtype=np.int64),
        np.array(["a", "b", "d", "c"], dtype="|S"),
        np.array([2, 13], dtype=np.int64))  # batch == 4, max_elems = 13

    original = [
        example(features=features({
            "val": bytes_feature([b"a", b"b"]),
            "idx": int64_feature([0, 3])
        })), example(features=features({
            "val": bytes_feature([b"c", b"d"]),
            "idx": int64_feature([7, 1])
        }))
    ]

    names = ["in1", "in2"]
    serialized = [m.SerializeToString() for m in original]

    expected_output = {
        "idx": expected_idx,
        "sp": expected_sp,
    }

    self._test(
        {
            "example_names": names,
            "serialized": tf.convert_to_tensor(serialized),
            "features": {
                "idx": tf.VarLenFeature(tf.int64),
                "sp": tf.SparseFeature("idx", "val", tf.string, 13),
            }
        },
        expected_output)


class ParseSingleExampleTest(tf.test.TestCase):

  def _test(self, kwargs, expected_values=None, expected_err=None):
    with self.test_session() as sess:
      if expected_err:
        with self.assertRaisesWithPredicateMatch(
            expected_err[0], expected_err[1]):
          out = tf.parse_single_example(**kwargs)
          sess.run(flatten_values_tensors_or_sparse(out.values()))
      else:
        # Returns dict w/ Tensors and SparseTensors.
        out = tf.parse_single_example(**kwargs)
        # Check values.
        tf_result = sess.run(flatten_values_tensors_or_sparse(out.values()))
        _compare_output_to_expected(self, out, expected_values, tf_result)

      # Check shapes.
      for k, f in kwargs["features"].items():
        if isinstance(f, tf.FixedLenFeature) and f.shape is not None:
          self.assertEqual(tuple(out[k].get_shape()), f.shape)
        elif isinstance(f, tf.VarLenFeature):
          self.assertEqual(
              tuple(out[k].indices.get_shape().as_list()), (None, 1))
          self.assertEqual(tuple(out[k].values.get_shape().as_list()), (None,))
          self.assertEqual(tuple(out[k].shape.get_shape().as_list()), (1,))

  def testSingleExampleWithSparseAndSparseFeatureAndDense(self):
    original = example(features=features({"c": float_feature([3, 4]),
                                          "val": bytes_feature([b"a", b"b"]),
                                          "idx": int64_feature([0, 3]),
                                          "st_a": float_feature([3.0, 4.0])}))

    serialized = original.SerializeToString()

    expected_st_a = (np.array(
        [[0], [1]], dtype=np.int64),  # indices
                     np.array(
                         [3.0, 4.0], dtype=np.float32),  # values
                     np.array(
                         [2], dtype=np.int64))  # shape: max_values = 2

    expected_sp = (  # indices, values, shape
        np.array([[0], [3]], dtype=np.int64),
        np.array(["a", "b"], dtype="|S"),
        np.array([13], dtype=np.int64))  # max_values = 13

    a_default = [1, 2, 3]
    b_default = np.random.rand(3, 3).astype(bytes)
    expected_output = {
        "st_a": expected_st_a,
        "sp": expected_sp,
        "a": [a_default],
        "b": b_default,
        "c": np.array(
            [3, 4], dtype=np.float32),
    }

    self._test(
        {
            "example_names": tf.convert_to_tensor("in1"),
            "serialized": tf.convert_to_tensor(serialized),
            "features": {
                "st_a": tf.VarLenFeature(tf.float32),
                "sp": tf.SparseFeature("idx", "val", tf.string, 13),
                "a": tf.FixedLenFeature(
                    (1, 3), tf.int64, default_value=a_default),
                "b": tf.FixedLenFeature(
                    (3, 3), tf.string, default_value=b_default),
                # Feature "c" must be provided, since it has no default_value.
                "c": tf.FixedLenFeature((2,), tf.float32),
            }
        },
        expected_output)


class ParseSequenceExampleTest(tf.test.TestCase):

  def testCreateSequenceExample(self):
    value = sequence_example(
        context=features({
            "global_feature": float_feature([1, 2, 3]),
        }),
        feature_lists=feature_lists({
            "repeated_feature_2_frames": feature_list([
                bytes_feature([b"a", b"b", b"c"]),
                bytes_feature([b"a", b"d", b"e"])
            ]),
            "repeated_feature_3_frames": feature_list([
                int64_feature([3, 4, 5, 6, 7]), int64_feature([-1, 0, 0, 0, 0]),
                int64_feature([1, 2, 3, 4, 5])
            ])
        }))
    value.SerializeToString()  # Smoke test

  def _test(self,
            kwargs,
            expected_context_values=None,
            expected_feat_list_values=None,
            expected_err=None):
    expected_context_values = expected_context_values or {}
    expected_feat_list_values = expected_feat_list_values or {}

    with self.test_session() as sess:
      if expected_err:
        with self.assertRaisesWithPredicateMatch(
            expected_err[0], expected_err[1]):
          c_out, fl_out = tf.parse_single_sequence_example(**kwargs)
          if c_out:
            sess.run(flatten_values_tensors_or_sparse(c_out.values()))
          if fl_out:
            sess.run(flatten_values_tensors_or_sparse(fl_out.values()))
      else:
        # Returns dicts w/ Tensors and SparseTensors.
        context_out, feat_list_out = tf.parse_single_sequence_example(**kwargs)
        context_result = sess.run(
            flatten_values_tensors_or_sparse(context_out.values(
            ))) if context_out else []
        feat_list_result = sess.run(
            flatten_values_tensors_or_sparse(feat_list_out.values(
            ))) if feat_list_out else []
        # Check values.
        _compare_output_to_expected(self, context_out, expected_context_values,
                                    context_result)
        _compare_output_to_expected(self, feat_list_out,
                                    expected_feat_list_values, feat_list_result)

      # Check shapes; if serialized is a Tensor we need its size to
      # properly check.
      if "context_features" in kwargs:
        for k, f in kwargs["context_features"].items():
          if isinstance(f, tf.FixedLenFeature) and f.shape is not None:
            self.assertEqual(
                tuple(context_out[k].get_shape().as_list()), f.shape)
          elif isinstance(f, tf.VarLenFeature):
            self.assertEqual(
                tuple(context_out[k].indices.get_shape().as_list()), (None, 1))
            self.assertEqual(
                tuple(context_out[k].values.get_shape().as_list()), (None,))
            self.assertEqual(
                tuple(context_out[k].shape.get_shape().as_list()), (1,))

  def testSequenceExampleWithSparseAndDenseContext(self):
    original = sequence_example(context=features({"c": float_feature([3, 4]),
                                                  "st_a": float_feature(
                                                      [3.0, 4.0])}))

    serialized = original.SerializeToString()

    expected_st_a = (np.array(
        [[0], [1]], dtype=np.int64),  # indices
                     np.array(
                         [3.0, 4.0], dtype=np.float32),  # values
                     np.array(
                         [2], dtype=np.int64))  # shape: num_features = 2

    a_default = [1, 2, 3]
    b_default = np.random.rand(3, 3).astype(bytes)
    expected_context_output = {
        "st_a": expected_st_a,
        "a": [a_default],
        "b": b_default,
        "c": np.array(
            [3, 4], dtype=np.float32),
    }

    self._test(
        {
            "example_name": "in1",
            "serialized": tf.convert_to_tensor(serialized),
            "context_features": {
                "st_a": tf.VarLenFeature(tf.float32),
                "a": tf.FixedLenFeature(
                    (1, 3), tf.int64, default_value=a_default),
                "b": tf.FixedLenFeature(
                    (3, 3), tf.string, default_value=b_default),
                # Feature "c" must be provided, since it has no default_value.
                "c": tf.FixedLenFeature((2,), tf.float32),
            }
        },
        expected_context_values=expected_context_output)

  def testSequenceExampleWithMultipleSizeFeatureLists(self):
    original = sequence_example(feature_lists=feature_lists({
        "a": feature_list([
            int64_feature([-1, 0, 1]),
            int64_feature([2, 3, 4]),
            int64_feature([5, 6, 7]),
            int64_feature([8, 9, 10]),
        ]),
        "b": feature_list([
            bytes_feature([b"r00", b"r01", b"r10", b"r11"])
        ]),
        "c": feature_list([
            float_feature([3, 4]), float_feature([-1, 2])
        ]),
    }))

    serialized = original.SerializeToString()

    expected_feature_list_output = {
        "a": np.array(
            [  # outer dimension is time.
                [[-1, 0, 1]],  # inside are 1x3 matrices
                [[2, 3, 4]],
                [[5, 6, 7]],
                [[8, 9, 10]]
            ],
            dtype=np.int64),
        "b": np.array(
            [  # outer dimension is time, inside are 2x2 matrices
                [[b"r00", b"r01"], [b"r10", b"r11"]]
            ],
            dtype=bytes),
        "c": np.array(
            [  # outer dimension is time, inside are 2-vectors
                [3, 4], [-1, 2]
            ],
            dtype=np.float32),
        "d": np.empty(
            shape=(0, 5), dtype=np.float32),  # empty_allowed_missing
    }

    self._test(
        {
            "example_name": "in1",
            "serialized": tf.convert_to_tensor(serialized),
            "sequence_features": {
                "a": tf.FixedLenSequenceFeature((1, 3), tf.int64),
                "b": tf.FixedLenSequenceFeature((2, 2), tf.string),
                "c": tf.FixedLenSequenceFeature((2,), tf.float32),
                "d": tf.FixedLenSequenceFeature(
                    (5,), tf.float32, allow_missing=True),
            }
        },
        expected_feat_list_values=expected_feature_list_output)

  def testSequenceExampleWithoutDebugName(self):
    original = sequence_example(feature_lists=feature_lists({
        "a": feature_list([
            int64_feature([3, 4]), int64_feature([1, 0])
        ]),
        "st_a": feature_list([
            float_feature([3.0, 4.0]), float_feature([5.0]), float_feature([])
        ]),
        "st_b": feature_list([
            bytes_feature([b"a"]), bytes_feature([]), bytes_feature([]),
            bytes_feature([b"b", b"c"])
        ])
    }))

    serialized = original.SerializeToString()

    expected_st_a = (
        np.array(
            [[0, 0], [0, 1], [1, 0]], dtype=np.int64),  # indices
        np.array(
            [3.0, 4.0, 5.0], dtype=np.float32),  # values
        np.array(
            [3, 2], dtype=np.int64))  # shape: num_time = 3, max_feat = 2

    expected_st_b = (
        np.array(
            [[0, 0], [3, 0], [3, 1]], dtype=np.int64),  # indices
        np.array(
            ["a", "b", "c"], dtype="|S"),  # values
        np.array(
            [4, 2], dtype=np.int64))  # shape: num_time = 4, max_feat = 2

    expected_st_c = (
        np.empty(
            (0, 2), dtype=np.int64),  # indices
        np.empty(
            (0,), dtype=np.int64),  # values
        np.array(
            [0, 0], dtype=np.int64))  # shape: num_time = 0, max_feat = 0

    expected_feature_list_output = {
        "a": np.array(
            [[3, 4], [1, 0]], dtype=np.int64),
        "st_a": expected_st_a,
        "st_b": expected_st_b,
        "st_c": expected_st_c,
    }

    self._test(
        {
            "serialized": tf.convert_to_tensor(serialized),
            "sequence_features": {
                "st_a": tf.VarLenFeature(tf.float32),
                "st_b": tf.VarLenFeature(tf.string),
                "st_c": tf.VarLenFeature(tf.int64),
                "a": tf.FixedLenSequenceFeature((2,), tf.int64),
            }
        },
        expected_feat_list_values=expected_feature_list_output)

  def testSequenceExampleWithSparseAndDenseFeatureLists(self):
    original = sequence_example(feature_lists=feature_lists({
        "a": feature_list([
            int64_feature([3, 4]), int64_feature([1, 0])
        ]),
        "st_a": feature_list([
            float_feature([3.0, 4.0]), float_feature([5.0]), float_feature([])
        ]),
        "st_b": feature_list([
            bytes_feature([b"a"]), bytes_feature([]), bytes_feature([]),
            bytes_feature([b"b", b"c"])
        ])
    }))

    serialized = original.SerializeToString()

    expected_st_a = (
        np.array(
            [[0, 0], [0, 1], [1, 0]], dtype=np.int64),  # indices
        np.array(
            [3.0, 4.0, 5.0], dtype=np.float32),  # values
        np.array(
            [3, 2], dtype=np.int64))  # shape: num_time = 3, max_feat = 2

    expected_st_b = (
        np.array(
            [[0, 0], [3, 0], [3, 1]], dtype=np.int64),  # indices
        np.array(
            ["a", "b", "c"], dtype="|S"),  # values
        np.array(
            [4, 2], dtype=np.int64))  # shape: num_time = 4, max_feat = 2

    expected_st_c = (
        np.empty(
            (0, 2), dtype=np.int64),  # indices
        np.empty(
            (0,), dtype=np.int64),  # values
        np.array(
            [0, 0], dtype=np.int64))  # shape: num_time = 0, max_feat = 0

    expected_feature_list_output = {
        "a": np.array(
            [[3, 4], [1, 0]], dtype=np.int64),
        "st_a": expected_st_a,
        "st_b": expected_st_b,
        "st_c": expected_st_c,
    }

    self._test(
        {
            "example_name": "in1",
            "serialized": tf.convert_to_tensor(serialized),
            "sequence_features": {
                "st_a": tf.VarLenFeature(tf.float32),
                "st_b": tf.VarLenFeature(tf.string),
                "st_c": tf.VarLenFeature(tf.int64),
                "a": tf.FixedLenSequenceFeature((2,), tf.int64),
            }
        },
        expected_feat_list_values=expected_feature_list_output)

  def testSequenceExampleListWithInconsistentDataFails(self):
    original = sequence_example(feature_lists=feature_lists({
        "a": feature_list([
            int64_feature([-1, 0]), float_feature([2, 3])
        ])
    }))

    serialized = original.SerializeToString()

    self._test(
        {
            "example_name": "in1",
            "serialized": tf.convert_to_tensor(serialized),
            "sequence_features": {"a": tf.FixedLenSequenceFeature(
                (2,), tf.int64)}
        },
        expected_err=(tf.OpError, "Feature list: a, Index: 1."
                      "  Data types don't match. Expected type: int64"))

  def testSequenceExampleListWithWrongDataTypeFails(self):
    original = sequence_example(feature_lists=feature_lists({
        "a": feature_list([
            float_feature([2, 3])
        ])
    }))

    serialized = original.SerializeToString()

    self._test(
        {
            "example_name": "in1",
            "serialized": tf.convert_to_tensor(serialized),
            "sequence_features": {"a": tf.FixedLenSequenceFeature(
                (2,), tf.int64)}
        },
        expected_err=(tf.OpError,
                      "Feature list: a, Index: 0.  Data types don't match."
                      " Expected type: int64"))

  def testSequenceExampleListWithWrongSparseDataTypeFails(self):
    original = sequence_example(feature_lists=feature_lists({
        "a": feature_list([
            int64_feature([3, 4]), int64_feature([1, 2]),
            float_feature([2.0, 3.0])
        ])
    }))

    serialized = original.SerializeToString()

    self._test(
        {
            "example_name": "in1",
            "serialized": tf.convert_to_tensor(serialized),
            "sequence_features": {"a": tf.FixedLenSequenceFeature(
                (2,), tf.int64)}
        },
        expected_err=(tf.OpError, "Name: in1, Feature list: a, Index: 2."
                      "  Data types don't match. Expected type: int64"
                      "  Feature is: float_list"))

  def testSequenceExampleListWithWrongShapeFails(self):
    original = sequence_example(feature_lists=feature_lists({
        "a": feature_list([
            int64_feature([2, 3]), int64_feature([2, 3, 4])
        ]),
    }))

    serialized = original.SerializeToString()

    self._test(
        {
            "example_name": "in1",
            "serialized": tf.convert_to_tensor(serialized),
            "sequence_features": {"a": tf.FixedLenSequenceFeature(
                (2,), tf.int64)}
        },
        expected_err=(tf.OpError, r"Name: in1, Key: a, Index: 1."
                      r"  Number of int64 values != expected."
                      r"  values size: 3 but output shape: \[2\]"))

  def testSequenceExampleWithMissingFeatureListFails(self):
    original = sequence_example(feature_lists=feature_lists({}))

    # Test fails because we didn't add:
    #  feature_list_dense_defaults = {"a": None}
    self._test(
        {
            "example_name": "in1",
            "serialized": tf.convert_to_tensor(original.SerializeToString()),
            "sequence_features": {"a": tf.FixedLenSequenceFeature(
                (2,), tf.int64)}
        },
        expected_err=(
            tf.OpError,
            "Name: in1, Feature list 'a' is required but could not be found."
            "  Did you mean to include it in"
            " feature_list_dense_missing_assumed_empty or"
            " feature_list_dense_defaults?"))


class DecodeJSONExampleTest(tf.test.TestCase):

  def _testRoundTrip(self, examples):
    with self.test_session() as sess:
      examples = np.array(examples, dtype=np.object)

      json_tensor = tf.constant(
          [json_format.MessageToJson(m) for m in examples.flatten()],
          shape=examples.shape,
          dtype=tf.string)
      binary_tensor = tf.decode_json_example(json_tensor)
      binary_val = sess.run(binary_tensor)

      if examples.shape:
        self.assertShapeEqual(binary_val, json_tensor)
        for input_example, output_binary in zip(
            np.array(examples).flatten(), binary_val.flatten()):
          output_example = tf.train.Example()
          output_example.ParseFromString(output_binary)
          self.assertProtoEquals(input_example, output_example)
      else:
        output_example = tf.train.Example()
        output_example.ParseFromString(binary_val)
        self.assertProtoEquals(examples.item(), output_example)

  def testEmptyTensor(self):
    self._testRoundTrip([])
    self._testRoundTrip([[], [], []])

  def testEmptyExamples(self):
    self._testRoundTrip([example(), example(), example()])

  def testDenseFeaturesScalar(self):
    self._testRoundTrip(
        example(features=features({"a": float_feature([1, 1, 3])})))

  def testDenseFeaturesVector(self):
    self._testRoundTrip([
        example(features=features({"a": float_feature([1, 1, 3])})),
        example(features=features({"a": float_feature([-1, -1, 2])})),
    ])

  def testDenseFeaturesMatrix(self):
    self._testRoundTrip([
        [example(features=features({"a": float_feature([1, 1, 3])}))],
        [example(features=features({"a": float_feature([-1, -1, 2])}))],
    ])

  def testSparseFeatures(self):
    self._testRoundTrip([
        example(features=features({"st_c": float_feature([3, 4])})),
        example(features=features({"st_c": float_feature([])})),
        example(features=features({"st_d": feature()})),
        example(features=features({"st_c": float_feature([1, 2, -1]),
                                   "st_d": bytes_feature([b"hi"])})),
    ])

  def testSerializedContainingBytes(self):
    aname = "a"
    bname = "b*has+a:tricky_name"
    self._testRoundTrip([
        example(features=features({aname: float_feature([1, 1]),
                                   bname: bytes_feature([b"b0_str"])})),
        example(features=features({aname: float_feature([-1, -1]),
                                   bname: bytes_feature([b"b1"])})),
    ])

  def testInvalidSyntax(self):
    with self.test_session() as sess:
      json_tensor = tf.constant(["{]"])
      binary_tensor = tf.decode_json_example(json_tensor)
      with self.assertRaisesOpError("Error while parsing JSON"):
        sess.run(binary_tensor)


class ParseTensorOpTest(tf.test.TestCase):

  def testToFloat32(self):
    with self.test_session():
      expected = np.random.rand(3, 4, 5).astype(np.float32)
      tensor_proto = tensor_util.make_tensor_proto(expected)

      serialized = tf.placeholder(tf.string)
      tensor = tf.parse_tensor(serialized, tf.float32)

      result = tensor.eval(
          feed_dict={serialized: tensor_proto.SerializeToString()})

      self.assertAllEqual(expected, result)

  def testToUint8(self):
    with self.test_session():
      expected = np.random.rand(3, 4, 5).astype(np.uint8)
      tensor_proto = tensor_util.make_tensor_proto(expected)

      serialized = tf.placeholder(tf.string)
      tensor = tf.parse_tensor(serialized, tf.uint8)

      result = tensor.eval(
          feed_dict={serialized: tensor_proto.SerializeToString()})

      self.assertAllEqual(expected, result)

  def testTypeMismatch(self):
    with self.test_session():
      expected = np.random.rand(3, 4, 5).astype(np.uint8)
      tensor_proto = tensor_util.make_tensor_proto(expected)

      serialized = tf.placeholder(tf.string)
      tensor = tf.parse_tensor(serialized, tf.uint16)

      with self.assertRaisesOpError(
          r"Type mismatch between parsed tensor \(uint8\) and dtype "
          r"\(uint16\)"):
        tensor.eval(feed_dict={serialized: tensor_proto.SerializeToString()})

  def testInvalidInput(self):
    with self.test_session():
      serialized = tf.placeholder(tf.string)
      tensor = tf.parse_tensor(serialized, tf.uint16)

      with self.assertRaisesOpError(
          "Could not parse `serialized` as TensorProto: 'bogus'"):
        tensor.eval(feed_dict={serialized: "bogus"})

      with self.assertRaisesOpError(
          r"Expected `serialized` to be a scalar, got shape: \[1\]"):
        tensor.eval(feed_dict={serialized: ["bogus"]})


if __name__ == "__main__":
  tf.test.main()
