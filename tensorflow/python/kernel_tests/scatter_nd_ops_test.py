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
"""Tests for tensorflow.ops.tf.scatter_nd."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np
import tensorflow as tf


def _AsType(v, vtype):
  return v.astype(vtype) if isinstance(v, np.ndarray) else vtype(v)


def _FlatInnerDims(tensor, ndims=2):
  shape = list(tensor.shape)
  return tensor.reshape([functools.reduce(lambda x, y: x * y,
                                          shape[:-ndims + 1], 1)] +
                        shape[-ndims + 1:])


def _FlatOuterDims(tensor, ndims=2):
  shape = list(tensor.shape)
  return tensor.reshape(shape[:ndims - 1] +
                        [functools.reduce(lambda x, y: x * y,
                                          shape[ndims - 1:], 1)])


def _NumpyScatterNd(ref, indices, updates, op):
  ixdim = indices.shape[-1]
  num_updates = indices.size / ixdim
  total_nd = len(ref.shape)
  slice_size = 1
  for i in range(ixdim, total_nd):
    slice_size *= ref.shape[i]
  flat_indices = _FlatInnerDims(indices)
  flat_updates = updates.reshape((num_updates, slice_size))
  output_flat = _FlatOuterDims(ref, ixdim + 1)
  for ix_updates, ix_output in enumerate(flat_indices):
    ix_output = tuple(ix_output)
    output_flat[ix_output] = op(output_flat[ix_output],
                                flat_updates[ix_updates])
  return output_flat.reshape(ref.shape)


def _NumpyUpdate(ref, indices, updates):
  return _NumpyScatterNd(ref, indices, updates, lambda p, u: u)


def _NumpyAdd(ref, indices, updates):
  return _NumpyScatterNd(ref, indices, updates, lambda p, u: p + u)


def _NumpySub(ref, indices, updates):
  return _NumpyScatterNd(ref, indices, updates, lambda p, u: p - u)


def _NumpyMul(ref, indices, updates):
  return _NumpyScatterNd(ref, indices, updates, lambda p, u: p * u)


def _NumpyDiv(ref, indices, updates):
  return _NumpyScatterNd(ref, indices, updates, lambda p, u: p / u)


class ScatterNdTest(tf.test.TestCase):

  def _VariableRankTest(self,
                        np_scatter,
                        tf_scatter,
                        vtype,
                        itype,
                        use_gpu,
                        repeat_indices=False):
    np.random.seed(8)
    ref_shapes = [(3, 6), (3, 6), (3, 6, 9), (3, 6, 9), (3, 6, 9), (3, 6, 9)]
    indices_shapes = [(2,), (2, 2), (2,), (2, 2), (2, 3), (2, 3, 3)]
    with self.test_session(use_gpu=use_gpu):
      for ref_shape, indices_shape in zip(ref_shapes, indices_shapes):
        num_updates = indices_shape[0]
        ixdim = indices_shape[-1]

        indexable_area_shape = ()
        for i in range(ixdim):
          indexable_area_shape += (ref_shape[i],)
        all_indices = [
            list(coord)
            for coord, _ in np.ndenumerate(
                np.empty(indexable_area_shape, vtype))
        ]
        np.random.shuffle(all_indices)
        indices = np.array(all_indices[:num_updates])

        if num_updates > 1 and repeat_indices:
          indices = indices[:num_updates // 2]
          for _ in range(num_updates - num_updates // 2):
            indices = np.append(
                indices, [indices[np.random.randint(num_updates // 2)]], axis=0)
          np.random.shuffle(indices)
        indices = _AsType(indices[:num_updates], itype)

        updates_shape = (num_updates,)
        for i in range(ixdim, len(ref_shape)):
          updates_shape += (ref_shape[i],)
        updates = _AsType(np.random.randn(*(updates_shape)), vtype)
        ref = _AsType(np.random.randn(*(ref_shape)), vtype)

        # Scatter via numpy
        new = ref.copy()
        np_scatter(new, indices, updates)
        # Scatter via tensorflow
        ref_var = tf.Variable(ref)
        ref_var.initializer.run()
        tf_scatter(ref_var, indices, updates).eval()
        # Compare
        self.assertAllClose(new, ref_var.eval())

  def _VariableRankTests(self, np_scatter, tf_scatter):
    for vtype in (np.float32, np.float64):
      for itype in (np.int32, np.int64):
        for use_gpu in (False, True):
          self._VariableRankTest(np_scatter, tf_scatter, vtype, itype, use_gpu)

  def testVariableRankUpdate(self):
    self._VariableRankTests(_NumpyUpdate, tf.scatter_nd_update)

  def testVariableRankAdd(self):
    self._VariableRankTests(_NumpyAdd, tf.scatter_nd_add)

  def testVariableRankSub(self):
    self._VariableRankTests(_NumpySub, tf.scatter_nd_sub)

  # TODO(simister): Re-enable once binary size increase due to
  # scatter_nd ops is under control.
  # def testVariableRankMul(self):
  #   self._VariableRankTests(_NumpyMul, tf.scatter_nd_mul)

  # def testVariableRankDiv(self):
  #   self._VariableRankTests(_NumpyDiv, tf.scatter_nd_div)

  def _ScatterRepeatIndicesTest(self, np_scatter, tf_scatter):
    for vtype in (np.float32, np.float64):
      for itype in (np.int32, np.int64):
        for use_gpu in (False, True):
          self._VariableRankTest(
              np_scatter,
              tf_scatter,
              vtype,
              itype,
              use_gpu,
              repeat_indices=True)

  def testScatterRepeatIndices(self):
    """This tests scatter_add using indices that repeat."""
    self._ScatterRepeatIndicesTest(_NumpyAdd, tf.scatter_nd_add)
    self._ScatterRepeatIndicesTest(_NumpySub, tf.scatter_nd_sub)
    # TODO(simister): Re-enable once binary size increase due to
    # extra templating is back under control.
    # self._ScatterRepeatIndicesTest(_NumpyMul, tf.scatter_nd_mul)
    # self._ScatterRepeatIndicesTest(_NumpyDiv, tf.scatter_nd_div)

  # TODO(simister): Re-enable once binary size increase due to
  # extra templating is back under control and this op is re-enabled
  # def testBooleanScatterUpdate(self):
  #   with self.test_session(use_gpu=False) as session:
  #     var = tf.Variable([True, False])
  #     update0 = tf.scatter_nd_update(var, [[1]], [True])
  #     update1 = tf.scatter_nd_update(
  #         var, tf.constant(
  #             [[0]], dtype=tf.int64), [False])
  #     var.initializer.run()
  #     session.run([update0, update1])
  #     self.assertAllEqual([False, True], var.eval())

  def testScatterOutOfRangeCpu(self):
    # TODO(simister): Re-enable once binary size increase due to
    # scatter_nd ops is under control.
    #  tf.scatter_nd_mul, tf.scatter_nd_div,
    for op in (tf.scatter_nd_add, tf.scatter_nd_sub, tf.scatter_nd_update):
      params = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32)
      updates = np.array([-3, -4, -5]).astype(np.float32)
      with self.test_session(use_gpu=False):
        ref = tf.Variable(params)
        ref.initializer.run()

        # Indices all in range, no problem.
        indices = np.array([[2], [0], [5]])
        op(ref, indices, updates).eval()

        # Test some out of range errors.
        indices = np.array([[-1], [0], [5]])
        with self.assertRaisesOpError(
            r"Invalid indices: \[0,0\] = \[-1\] is not in \[0, 6\)"):
          op(ref, indices, updates).eval()

        indices = np.array([[2], [0], [6]])
        with self.assertRaisesOpError(
            r"Invalid indices: \[2,0\] = \[6\] is not in \[0, 6\)"):
          op(ref, indices, updates).eval()

  def testRank3ValidShape(self):
    indices = tf.zeros([2, 2, 2], tf.int32)
    updates = tf.zeros([2, 2, 2], tf.int32)
    shape = np.array([2, 2, 2])
    self.assertAllEqual(
        tf.scatter_nd(indices, updates, shape).get_shape().as_list(), shape)

    ref = tf.Variable(tf.zeros(shape, tf.int32))
    self.assertAllEqual(
        tf.scatter_nd_update(ref, indices, updates).get_shape().as_list(),
        shape)

  def testUndefinedIndicesShape(self):
    indices = tf.placeholder(tf.int32, shape=None)
    updates = tf.placeholder(tf.int32, shape=[2, 2, 2])
    shape = tf.constant([2, 2, 2], tf.int32)
    tf.scatter_nd(indices, updates, shape)

  def testUndefinedUpdatesShape(self):
    indices = tf.placeholder(tf.int32, shape=[2, 2, 2])
    updates = tf.placeholder(tf.int32, shape=None)
    shape = tf.constant([2, 2, 2], tf.int32)
    tf.scatter_nd(indices, updates, shape)

  def testUndefinedOutputShape(self):
    indices = tf.placeholder(tf.int32, shape=[2, 2, 2])
    updates = tf.placeholder(tf.int32, shape=[2, 2, 2])
    shape = tf.placeholder(tf.int32, shape=[None])
    tf.scatter_nd(indices, updates, shape)

  def testEmptyoutputShape1(self):
    indices = tf.zeros([2, 2, 2], tf.int32)
    updates = tf.zeros([2, 2, 2], tf.int32)
    shape = tf.constant([0, 3, 2], tf.int32)

    with self.assertRaisesWithPredicateMatch(
        ValueError, "Indices and updates specified for empty output shape"):
      tf.scatter_nd(indices, updates, shape)

  def testEmptyoutputShape2(self):
    indices = tf.placeholder(tf.int32, shape=None)
    updates = tf.placeholder(tf.int32, shape=None)
    shape = tf.constant([0, 3, 2], tf.int32)

    with self.test_session():
      tf.scatter_nd(indices, updates, shape).eval(feed_dict={
          indices: np.zeros(
              [2, 2, 2], dtype=np.int32),
          updates: np.zeros(
              [2, 2, 2], dtype=np.int32)
      })

  def testEmptyoutputShape3(self):
    indices = tf.zeros([0], tf.int32)
    updates = tf.zeros([0], tf.int32)
    shape = tf.constant([0], tf.int32)
    scatter = tf.scatter_nd(indices, updates, shape)

    with self.test_session():
      self.assertEqual(scatter.eval().size, 0)

  def testRank3InvalidShape1(self):
    indices = tf.zeros([3, 2, 2], tf.int32)
    updates = tf.zeros([2, 2, 2], tf.int32)
    shape = np.array([2, 2, 2])
    with self.assertRaisesWithPredicateMatch(
        ValueError, "The outer \\d+ dimensions of indices\\.shape="):
      tf.scatter_nd(indices, updates, shape)

    ref = tf.Variable(tf.zeros(shape, tf.int32))
    with self.assertRaisesWithPredicateMatch(
        ValueError, "The outer \\d+ dimensions of indices\\.shape="):
      tf.scatter_nd_update(ref, indices, updates)

  def testRank3InvalidShape2(self):
    indices = tf.zeros([2, 2, 1], tf.int32)
    updates = tf.zeros([2, 2], tf.int32)
    shape = np.array([2, 2, 2])
    with self.assertRaisesWithPredicateMatch(
        ValueError, "The inner \\d+ dimensions of output\\.shape="):
      tf.scatter_nd(indices, updates, shape)

    ref = tf.Variable(tf.zeros(shape, tf.int32))
    with self.assertRaisesWithPredicateMatch(
        ValueError, "The inner \\d+ dimensions of ref\\.shape="):
      tf.scatter_nd_update(ref, indices, updates)

  def testGradientsRank2ElementUpdate(self):
    indices = tf.constant([[0, 0], [1, 1]], dtype=tf.int32)
    updates = tf.constant([1, 4], dtype=tf.float64)
    shape = tf.constant([2, 2], dtype=tf.int32)
    outputs = tf.scatter_nd(indices, updates, shape)

    grad_vals = tf.constant([[1, 2], [3, 4]], dtype=tf.float64)
    grads = tf.gradients([outputs], [updates], [grad_vals])[0]
    expected_grads = np.array([1, 4], dtype=np.float64)
    with self.test_session():
      self.assertAllEqual(expected_grads, grads.eval())

  def testGradientsRank2SliceUpdate(self):
    indices = tf.constant([[1], [0]], dtype=tf.int32)
    updates = tf.constant([[3, 4], [1, 2]], dtype=tf.float64)
    shape = tf.constant([2, 2], dtype=tf.int32)
    outputs = tf.scatter_nd(indices, updates, shape)

    grad_vals = tf.constant([[3, 4], [1, 2]], dtype=tf.float64)
    grads = tf.gradients([outputs], [updates], [grad_vals])[0]
    expected_grads = np.array([[1, 2], [3, 4]], dtype=np.float64)
    with self.test_session():
      self.assertAllEqual(expected_grads, grads.eval())

  def testGradientsRank3SliceUpdate(self):
    indices = tf.constant([[[0, 1], [1, 0]], [[0, 0], [1, 1]]], dtype=tf.int32)
    updates = tf.constant(
        [[[5, 7], [2, 4]], [[1, 3], [6, 8]]], dtype=tf.float64)
    shape = tf.constant([2, 2, 2], dtype=tf.int32)
    outputs = tf.scatter_nd(indices, updates, shape)

    grad_vals = tf.constant(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.float64)
    grads = tf.gradients([outputs], [updates], [grad_vals])[0]
    expected_grads = np.array(
        [[[3, 4], [5, 6]], [[1, 2], [7, 8]]], dtype=np.float64)
    with self.test_session():
      self.assertAllEqual(expected_grads, grads.eval())

  def testConcurrentUpdates(self):
    num_updates = 10000
    update_values = np.random.rand(num_updates)
    ref = tf.Variable(np.zeros([2, 2]), dtype=tf.float64)
    indices = tf.constant([[0, 1]] * num_updates, dtype=tf.int32)
    updates = tf.constant(update_values, dtype=tf.float64)

    exepected_result = np.zeros([2, 2], dtype=np.float64)
    exepected_result[0, 1] = np.sum(update_values)

    scatter = tf.scatter_nd_add(ref, indices, updates)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
      sess.run(init)
      result = sess.run(scatter)
      assert np.allclose(result, exepected_result)

  # TODO(fpmc): Re-enable this test when gpu_pip test actually runs on a GPU.
  def _disabledTestScatterOutOfRangeGpu(self):
    if not tf.test.IsBuiltWithCuda():
      return
    # TODO(simister): Re-enable once binary size increase due to
    # scatter_nd ops is under control.
    # tf.scatter_nd_mul, tf.scatter_nd_div,
    for op in (tf.scatter_nd_add, tf.scatter_nd_sub, tf.scatter_nd_update):
      params = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32)
      updates = np.array([-3, -4, -5]).astype(np.float32)
      # With GPU, the code ignores indices that are out of range.
      # We don't test the implementation; just test there's no failures.
      with self.test_session(force_gpu=True):
        ref = tf.Variable(params)
        ref.initializer.run()

        # Indices all in range, no problem.
        indices = np.array([2, 0, 5])
        op(ref, indices, updates).eval()

        # Indicies out of range should not fail.
        indices = np.array([-1, 0, 5])
        op(ref, indices, updates).eval()
        indices = np.array([2, 0, 6])
        op(ref, indices, updates).eval()

  def testScatterNdRepatedIndicesAdd(self):
    indices = tf.zeros([100000, 1], tf.int32)
    values = np.random.randn(100000)
    shape = [1]
    with self.test_session():
      val = tf.scatter_nd(indices, values, shape).eval()
    self.assertAllClose([np.sum(values)], val)


if __name__ == "__main__":
  tf.test.main()
