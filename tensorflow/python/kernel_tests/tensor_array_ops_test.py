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
"""Tests for tensorflow.ops.tensor_array_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_grad
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class TensorArrayTest(test.TestCase):

  def testTensorArrayWriteRead(self):
    with self.test_session(use_gpu=True) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=3,
          infer_shape=False)

      w0 = ta.write(0, [[4.0, 5.0]])
      w1 = w0.write(1, [[1.0]])
      w2 = w1.write(2, -3.0)

      r0 = w2.read(0)
      r1 = w2.read(1)
      r2 = w2.read(2)

      d0, d1, d2 = session.run([r0, r1, r2])
      self.assertAllEqual([[4.0, 5.0]], d0)
      self.assertAllEqual([[1.0]], d1)
      self.assertAllEqual(-3.0, d2)

  def _testTensorArrayWritePack(self, tf_dtype):
    dtype = tf_dtype.as_numpy_dtype()
    with self.test_session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=tf_dtype, tensor_array_name="foo", size=3)

      if tf_dtype == dtypes.string:
        # In Python3, np.str is unicode, while we always want bytes
        convert = lambda x: np.asarray(x).astype("|S")
      else:
        convert = lambda x: np.asarray(x).astype(dtype)

      w0 = ta.write(0, convert([[4.0, 5.0]]))
      w1 = w0.write(1, convert([[6.0, 7.0]]))
      w2 = w1.write(2, convert([[8.0, 9.0]]))

      c0 = w2.stack()

      self.assertAllEqual(
          convert([[[4.0, 5.0]], [[6.0, 7.0]], [[8.0, 9.0]]]), c0.eval())

  def _testTensorArrayWritePackMaybeLegacy(self):
    self._testTensorArrayWritePack(dtypes.float32)
    self._testTensorArrayWritePack(dtypes.float64)
    self._testTensorArrayWritePack(dtypes.int32)
    self._testTensorArrayWritePack(dtypes.int64)
    self._testTensorArrayWritePack(dtypes.complex64)
    self._testTensorArrayWritePack(dtypes.complex128)
    self._testTensorArrayWritePack(dtypes.string)

  def testTensorArrayWritePack(self):
    self._testTensorArrayWritePackMaybeLegacy()

  def _testTensorArrayWriteConcat(self, tf_dtype):
    dtype = tf_dtype.as_numpy_dtype()
    with self.test_session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=tf_dtype, tensor_array_name="foo", size=3, infer_shape=False)

      if tf_dtype == dtypes.string:
        # In Python3, np.str is unicode, while we always want bytes
        convert = lambda x: np.asarray(x).astype("|S")
      else:
        convert = lambda x: np.asarray(x).astype(dtype)

      w0 = ta.write(0, convert([[4.0, 5.0], [104.0, 105.0], [204.0, 205.0]]))
      w1 = w0.write(1, convert([[6.0, 7.0], [106.0, 107.0]]))
      w2 = w1.write(2, convert([[8.0, 9.0]]))

      c0 = w2.concat()

      self.assertAllEqual(
          convert([[4.0, 5.0], [104.0, 105.0], [204.0, 205.0], [6.0, 7.0],
                   [106.0, 107.0], [8.0, 9.0]]), c0.eval())

  def testTensorArrayWriteConcat(self):
    self._testTensorArrayWriteConcat(dtypes.float32)
    self._testTensorArrayWriteConcat(dtypes.float64)
    self._testTensorArrayWriteConcat(dtypes.int32)
    self._testTensorArrayWriteConcat(dtypes.int64)
    self._testTensorArrayWriteConcat(dtypes.complex64)
    self._testTensorArrayWriteConcat(dtypes.complex128)
    self._testTensorArrayWriteConcat(dtypes.string)

  def _testTensorArrayPackNotAllValuesAvailableFails(self):
    with self.test_session():
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=3)

      with self.assertRaisesOpError("Could not read from TensorArray index 1 "
                                    "because it has not yet been written to."):
        ta.write(0, [[4.0, 5.0]]).stack().eval()

  def testTensorArrayPackNotAllValuesAvailableFails(self):
    self._testTensorArrayPackNotAllValuesAvailableFails()

  def _testTensorArrayUnpackRead(self, tf_dtype):
    dtype = tf_dtype.as_numpy_dtype()
    with self.test_session(use_gpu=True) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=tf_dtype, tensor_array_name="foo", size=3)

      if tf_dtype is dtypes.string:
        # In Python3, np.str is unicode, while we always want bytes
        convert = lambda x: np.asarray(x).astype("|S")
      else:
        convert = lambda x: np.asarray(x).astype(dtype)

      # Unpack a vector into scalars
      w0 = ta.unstack(convert([1.0, 2.0, 3.0]))
      r0 = w0.read(0)
      r1 = w0.read(1)
      r2 = w0.read(2)

      d0, d1, d2 = session.run([r0, r1, r2])
      self.assertAllEqual(convert(1.0), d0)
      self.assertAllEqual(convert(2.0), d1)
      self.assertAllEqual(convert(3.0), d2)

      ta = tensor_array_ops.TensorArray(
          dtype=tf_dtype, tensor_array_name="foo", size=3)

      # Unpack a matrix into vectors
      w1 = ta.unstack(convert([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]]))
      r0 = w1.read(0)
      r1 = w1.read(1)
      r2 = w1.read(2)

      d0, d1, d2 = session.run([r0, r1, r2])
      self.assertAllEqual(convert([1.0, 1.1]), d0)
      self.assertAllEqual(convert([2.0, 2.1]), d1)
      self.assertAllEqual(convert([3.0, 3.1]), d2)

      # Reset ta because we're going to change the shape, else shape
      # inference will throw an error.
      ta = tensor_array_ops.TensorArray(
          dtype=tf_dtype, tensor_array_name="foo", size=3)

      # Try unpacking an empty matrix, which should not cause an error.
      w2 = ta.unstack(convert([[], [], []]))
      r0 = w2.read(0)
      r1 = w2.read(1)
      r2 = w2.read(2)

      d0, d1, d2 = session.run([r0, r1, r2])
      self.assertAllEqual(convert([]), d0)
      self.assertAllEqual(convert([]), d1)
      self.assertAllEqual(convert([]), d2)

  def _testTensorArrayUnpackReadMaybeLegacy(self):
    self._testTensorArrayUnpackRead(dtypes.float32)
    self._testTensorArrayUnpackRead(dtypes.float64)
    self._testTensorArrayUnpackRead(dtypes.int32)
    self._testTensorArrayUnpackRead(dtypes.int64)
    self._testTensorArrayUnpackRead(dtypes.complex64)
    self._testTensorArrayUnpackRead(dtypes.complex128)
    self._testTensorArrayUnpackRead(dtypes.string)

  def testTensorArrayUnpackRead(self):
    self._testTensorArrayUnpackReadMaybeLegacy()

  def _testTensorArraySplitRead(self, tf_dtype):
    dtype = tf_dtype.as_numpy_dtype()
    with self.test_session(use_gpu=True) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=tf_dtype, tensor_array_name="foo", size=3, infer_shape=False)

      if tf_dtype == dtypes.string:
        # In Python3, np.str is unicode, while we always want bytes
        convert = lambda x: np.asarray(x).astype("|S")
      else:
        convert = lambda x: np.asarray(x).astype(dtype)

      # Split an empty vector
      lengths = constant_op.constant([0, 0, 0])
      w0 = ta.split(convert([]), lengths=lengths)
      r0 = w0.read(0)
      r1 = w0.read(1)
      r2 = w0.read(2)

      d0, d1, d2 = session.run([r0, r1, r2])
      self.assertAllEqual(convert([]), d0)
      self.assertAllEqual(convert([]), d1)
      self.assertAllEqual(convert([]), d2)

      # Split a vector
      lengths = constant_op.constant([2, 0, 1])
      w0 = ta.split(convert([1.0, 2.0, 3.0]), lengths=lengths)
      r0 = w0.read(0)
      r1 = w0.read(1)
      r2 = w0.read(2)

      d0, d1, d2 = session.run([r0, r1, r2])
      self.assertAllEqual(convert([1.0, 2.0]), d0)
      self.assertAllEqual(convert([]), d1)
      self.assertAllEqual(convert([3.0]), d2)

      # Split a matrix
      lengths = constant_op.constant([2, 0, 1])
      w0 = ta.split(
          convert([[1.0, 101.0], [2.0, 201.0], [3.0, 301.0]]), lengths=lengths)
      r0 = w0.read(0)
      r1 = w0.read(1)
      r2 = w0.read(2)

      d0, d1, d2 = session.run([r0, r1, r2])
      self.assertAllEqual(convert([[1.0, 101.0], [2.0, 201.0]]), d0)
      self.assertAllEqual(convert([]).reshape(0, 2), d1)
      self.assertAllEqual(convert([[3.0, 301.0]]), d2)

  def testTensorArraySplitRead(self):
    self._testTensorArraySplitRead(dtypes.float32)
    self._testTensorArraySplitRead(dtypes.float64)
    self._testTensorArraySplitRead(dtypes.int32)
    self._testTensorArraySplitRead(dtypes.int64)
    self._testTensorArraySplitRead(dtypes.complex64)
    self._testTensorArraySplitRead(dtypes.complex128)
    self._testTensorArraySplitRead(dtypes.string)

  def testTensorGradArrayWriteRead(self):
    with self.test_session(use_gpu=True) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=3,
          infer_shape=False)
      g_ta = ta.grad("grad")

      w0 = ta.write(0, [[4.0, 5.0]])
      w1 = w0.write(1, [[1.0]])
      w2 = w1.write(2, -3.0)

      g_w0 = g_ta.write(0, [[5.0, 6.0]])
      g_w1 = g_w0.write(1, [[2.0]])
      g_w2 = g_w1.write(2, -2.0)

      r0 = w2.read(0)
      r1 = w2.read(1)
      r2 = w2.read(2)

      g_r0 = g_w2.read(0)
      g_r1 = g_w2.read(1)
      g_r2 = g_w2.read(2)

      d0, d1, d2, g_d0, g_d1, g_d2 = session.run([r0, r1, r2, g_r0, g_r1, g_r2])
      self.assertAllEqual([[4.0, 5.0]], d0)
      self.assertAllEqual([[1.0]], d1)
      self.assertAllEqual(-3.0, d2)
      self.assertAllEqual([[5.0, 6.0]], g_d0)
      self.assertAllEqual([[2.0]], g_d1)
      self.assertAllEqual(-2.0, g_d2)

  def testTensorGradArrayDynamicWriteRead(self):
    with self.test_session(use_gpu=True) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=0,
          dynamic_size=True,
          infer_shape=False)

      w0 = ta.write(0, [[4.0, 5.0]])
      w1 = w0.write(1, [[1.0]])
      w2 = w1.write(2, -3.0)

      g_ta = w2.grad("grad")  # Get gradient array here so we know the shape

      s = w2.size()
      g_s = g_ta.size()

      g_w0 = g_ta.write(0, [[5.0, 6.0]])
      g_w1 = g_w0.write(1, [[2.0]])
      g_w2 = g_w1.write(2, -2.0)

      r0 = w2.read(0)
      r1 = w2.read(1)
      r2 = w2.read(2)

      g_r0 = g_w2.read(0)
      g_r1 = g_w2.read(1)
      g_r2 = g_w2.read(2)

      d0, d1, d2, g_d0, g_d1, g_d2, vs, g_vs = session.run(
          [r0, r1, r2, g_r0, g_r1, g_r2, s, g_s])
      self.assertAllEqual([[4.0, 5.0]], d0)
      self.assertAllEqual([[1.0]], d1)
      self.assertAllEqual(-3.0, d2)
      self.assertAllEqual([[5.0, 6.0]], g_d0)
      self.assertAllEqual([[2.0]], g_d1)
      self.assertAllEqual(-2.0, g_d2)
      self.assertAllEqual(3, vs)
      self.assertAllEqual(3, g_vs)

  def testTensorGradAccessTwiceReceiveSameObject(self):
    with self.test_session(use_gpu=True) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=3)
      g_ta_0 = ta.grad("grad")
      g_ta_1 = ta.grad("grad")

      with ops.control_dependencies([g_ta_0.write(0, [[4.0, 5.0]]).flow]):
        # Write with one gradient handle, read with another copy of it
        r1_0 = g_ta_1.read(0)

      t_g_ta_0, t_g_ta_1, d_r1_0 = session.run(
          [g_ta_0.handle.op, g_ta_1.handle.op, r1_0])
      self.assertAllEqual(t_g_ta_0, t_g_ta_1)
      self.assertAllEqual([[4.0, 5.0]], d_r1_0)

  def testTensorArrayWriteWrongIndexOrDataTypeFails(self):
    with self.test_session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=3)

      # Test writing the wrong datatype
      with self.assertRaisesOpError(
          "TensorArray dtype is float but Op is trying to write dtype string"):
        ta.write(-1, "wrong_type_scalar").flow.eval()

      # Test writing to a negative index
      with self.assertRaisesOpError(
          "Tried to write to index -1 but array is not "
          "resizeable and size is: 3"):
        ta.write(-1, 3.0).flow.eval()

      # Test reading from too large an index
      with self.assertRaisesOpError(
          "Tried to write to index 3 but array is not "
          "resizeable and size is: 3"):
        ta.write(3, 3.0).flow.eval()

  def testTensorArrayReadWrongIndexOrDataTypeFails(self):
    with self.test_session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=3)

      w0 = ta.write(0, [[4.0, 5.0]])

      # Test reading wrong datatype
      r0_bad = gen_data_flow_ops._tensor_array_read_v3(
          handle=w0.handle, index=0, dtype=dtypes.float64, flow_in=w0.flow)
      with self.assertRaisesOpError(
          "TensorArray dtype is float but Op requested dtype double."):
        r0_bad.eval()

      # Test reading from a different index than the one we wrote to
      r1 = w0.read(1)
      with self.assertRaisesOpError(
          "Could not read from TensorArray index 1 because "
          "it has not yet been written to."):
        r1.eval()

      # Test reading from a negative index
      with self.assertRaisesOpError(
          r"Tried to read from index -1 but array size is: 3"):
        ta.read(-1).eval()

      # Test reading from too large an index
      with self.assertRaisesOpError(
          "Tried to read from index 3 but array size is: 3"):
        ta.read(3).eval()

  def testTensorArrayWriteMultipleFails(self):
    with self.test_session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=3)

      with self.assertRaisesOpError(
          "Could not write to TensorArray index 2 because "
          "it has already been written to."):
        ta.write(2, 3.0).write(2, 3.0).flow.eval()

  def testTensorArrayConcatIncompatibleShapesFails(self):
    with self.test_session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=3,
          infer_shape=False)

      w1 = ta.write(0, 3.0)
      w2 = w1.write(1, 4.0)
      w3 = w2.write(2, [3.0])

      with self.assertRaisesOpError(
          "Concat saw a scalar shape at index 0 but requires at least vectors"):
        w3.concat().eval()

      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=3,
          infer_shape=False)

      w1 = ta.write(0, [3.0])
      w2 = w1.write(1, [4.0])
      w3 = w2.write(2, [[3.0]])

      with self.assertRaisesOpError(
          r"TensorArray has inconsistent shapes.  Index 0 has "
          r"\(excepting dimension 0\) shape: \[\] but index 2 has \(excepting "
          r"dimension 0\) shape: \[1\]"):
        w3.concat().eval()

  def testTensorArraySplitIncompatibleShapesFails(self):
    with self.test_session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=3,
          infer_shape=False)

      with self.assertRaisesOpError(
          r"Expected lengths to be a vector, received shape: \[\]"):
        lengths = array_ops.placeholder(dtypes.int64)
        ta.split([1.0, 2.0, 3.0], lengths).flow.eval(feed_dict={lengths: 1})

      with self.assertRaisesOpError(
          r"Expected sum of lengths to be equal to values.shape\[0\], "
          r"but sum of lengths is 1 and value's shape is: \[3\]"):
        ta.split([1.0, 2.0, 3.0], [1]).flow.eval()

      with self.assertRaisesOpError(
          r"Expected value to be at least a vector, but received shape: \[\]"):
        ta.split(1.0, [1]).flow.eval()

      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=2,
          infer_shape=False)

      with self.assertRaisesOpError(
          r"TensorArray's size is not equal to the size of lengths "
          r"\(2 vs. 1\), and the TensorArray is not marked as "
          r"dynamically resizeable"):
        ta.split([1.0], [1]).flow.eval()

  def _testTensorArrayWriteGradientAddMultipleAdds(self, dtype):
    with self.test_session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=dtype, tensor_array_name="foo", size=3, infer_shape=False)
      ta_grad = ta.grad("grad")

      c = lambda x: np.asarray(x, dtype=dtype.as_numpy_dtype)

      w0 = ta.write(2, c(3.0))
      w1 = w0.write(2, c(4.0))

      w0_grad = ta_grad.write(2, c(3.0))
      w1_grad = w0_grad.write(2, c(4.0))
      w2_grad = w1_grad.write(2, c(5.0))

      # Assert that aggregation works correctly
      self.assertAllEqual(c(12.00), w2_grad.read(2).eval())

      # Assert that if multiple_writes_aggregate is not enabled,
      # multiple writes raise an exception.
      with self.assertRaisesOpError(
          r"TensorArray foo_.*: Could not write to TensorArray index 2 because "
          r"it has already been written to."):
        w1.flow.eval()

      # Using differing shapes causes an exception
      wb0_grad = ta_grad.write(1, c(1.0))
      wb1_grad = wb0_grad.write(1, c([1.0]))

      with self.assertRaisesOpError(
          r"Could not aggregate to TensorArray index 1 because the "
          r"existing shape is \[\] but the new input shape is \[1\]"):
        wb1_grad.flow.eval()

  def testTensorArrayWriteGradientAddMultipleAdds(self):
    for dtype in (dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64,
                  dtypes.complex64, dtypes.complex128):
      self._testTensorArrayWriteGradientAddMultipleAdds(dtype)

  def testMultiTensorArray(self):
    with self.test_session(use_gpu=True):
      h1 = tensor_array_ops.TensorArray(
          size=1, dtype=dtypes.float32, tensor_array_name="foo")
      w1 = h1.write(0, 4.0)
      r1 = w1.read(0)

      h2 = tensor_array_ops.TensorArray(
          size=1, dtype=dtypes.float32, tensor_array_name="bar")

      w2 = h2.write(0, 5.0)
      r2 = w2.read(0)
      r = r1 + r2
      self.assertAllClose(9.0, r.eval())

  def _testTensorArrayGradientWriteReadType(self, dtype):
    with self.test_session(use_gpu=True) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.as_dtype(dtype),
          tensor_array_name="foo",
          size=3,
          infer_shape=False)

      c = lambda x: np.array(x, dtype=dtype)

      value_0 = constant_op.constant(c([[4.0, 5.0]]))
      value_1 = constant_op.constant(c(3.0))

      w0 = ta.write(0, value_0)
      w1 = w0.write(1, value_1)
      r0 = w1.read(0)
      r1 = w1.read(1)
      r0_2 = w1.read(0)

      # Test individual components' gradients
      grad_just_r0 = gradients_impl.gradients(
          ys=[r0], xs=[value_0], grad_ys=[c([[2.0, 3.0]])])
      grad_just_r0_vals = session.run(grad_just_r0)
      self.assertAllEqual(c([[2.0, 3.0]]), grad_just_r0_vals[0])

      grad_r0_r0_2 = gradients_impl.gradients(
          ys=[r0, r0_2],
          xs=[value_0],
          grad_ys=[c([[2.0, 3.0]]), c([[1.0, -1.0]])])
      grad_r0_r0_2_vals = session.run(grad_r0_r0_2)
      self.assertAllEqual(c([[3.0, 2.0]]), grad_r0_r0_2_vals[0])

      grad_just_r1 = gradients_impl.gradients(
          ys=[r1], xs=[value_1], grad_ys=[c(-2.0)])
      grad_just_r1_vals = session.run(grad_just_r1)
      self.assertAllEqual(c(-2.0), grad_just_r1_vals[0])

      # Test combined gradients
      grad = gradients_impl.gradients(
          ys=[r0, r0_2, r1],
          xs=[value_0, value_1],
          grad_ys=[c([[2.0, 3.0]]), c([[1.0, -1.0]]), c(-2.0)])
      grad_vals = session.run(grad)
      self.assertEqual(len(grad_vals), 2)
      self.assertAllEqual(c([[3.0, 2.0]]), grad_vals[0])
      self.assertAllEqual(c(-2.0), grad_vals[1])

  def testTensorArrayGradientWriteRead(self):
    for dtype in (np.float32, np.float64, np.int32, np.int64, np.complex64,
                  np.complex128):
      self._testTensorArrayGradientWriteReadType(dtype)

  def _testTensorArrayGradientWritePackConcatAndRead(self):
    with self.test_session(use_gpu=True) as sess:
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=2,
          clear_after_read=False)

      value_0 = constant_op.constant([-1.0, 1.0])
      value_1 = constant_op.constant([-10.0, 10.0])

      w0 = ta.write(0, value_0)
      w1 = w0.write(1, value_1)
      p0 = w1.stack()
      r0 = w1.read(0)
      s0 = w1.concat()

      # Test gradient accumulation between read(0), pack(), and concat()
      with ops.control_dependencies([p0, r0, s0]):
        grad_r = gradients_impl.gradients(
            ys=[p0, r0, s0],
            xs=[value_0, value_1],
            grad_ys=[
                [[2.0, 3.0], [4.0, 5.0]],  # pack gradient
                [-0.5, 1.5],  # read(0) gradient
                [20.0, 30.0, 40.0, 50.0]
            ])  # concat gradient
      grad_vals = sess.run(grad_r)  # 2 + 2 entries

      self.assertAllClose([2.0 - 0.5 + 20.0, 3.0 + 1.5 + 30.0], grad_vals[0])
      self.assertAllEqual([4.0 + 40.0, 5.0 + 50.0], grad_vals[1])

  def testTensorArrayGradientWritePackConcatAndRead(self):
    self._testTensorArrayGradientWritePackConcatAndRead()

  def testTensorArrayReadTwice(self):
    with self.test_session(use_gpu=True):
      value = constant_op.constant([[1.0, -1.0], [10.0, -10.0]])

      ta_readonce = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=2)

      w_readonce = ta_readonce.unstack(value)
      r0_readonce = w_readonce.read(0)
      with ops.control_dependencies([r0_readonce]):
        r1_readonce = w_readonce.read(0)

      with self.assertRaisesOpError(
          r"Could not read index 0 twice because it was cleared after a "
          r"previous read \(perhaps try setting clear_after_read = false\?\)"):
        r1_readonce.eval()

      ta_readtwice = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=2,
          clear_after_read=False)
      w_readtwice = ta_readtwice.unstack(value)
      r0_readtwice = w_readtwice.read(0)
      with ops.control_dependencies([r0_readtwice]):
        r1_readtwice = w_readtwice.read(0)

      self.assertAllEqual([1.0, -1.0], r1_readtwice.eval())

  def _testTensorArrayGradientUnpackRead(self):
    with self.test_session(use_gpu=True) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=2,
          clear_after_read=False)

      value = constant_op.constant([[1.0, -1.0], [10.0, -10.0]])

      w = ta.unstack(value)
      r0 = w.read(0)
      r0_1 = w.read(0)
      r1 = w.read(1)

      # Test combined gradients + aggregation of read(0)
      grad = gradients_impl.gradients(
          ys=[r0, r0_1, r1],
          xs=[value],
          grad_ys=[[2.0, 3.0], [-1.5, 1.5], [4.0, 5.0]])
      grad_vals = session.run(grad)

      self.assertEqual(len(grad_vals), 1)
      self.assertAllEqual([[2.0 - 1.5, 3.0 + 1.5], [4.0, 5.0]], grad_vals[0])

  def testTensorArrayGradientUnpackRead(self):
    self._testTensorArrayGradientUnpackRead()

  def testTensorArrayGradientSplitConcat(self):
    with self.test_session(use_gpu=True) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=2)

      value = constant_op.constant(
          [[1.0, -1.0], [10.0, -10.0], [100.0, -100.0]])

      w = ta.split(value, [2, 1])
      r = w.concat()

      # Test combined gradients
      grad = gradients_impl.gradients(
          ys=[r],
          xs=[value],
          grad_ys=[[[2.0, -2.0], [20.0, -20.0], [200.0, -200.0]]])
      grad_vals = session.run(grad)

      self.assertEqual(len(grad_vals), 1)
      self.assertAllEqual([[2.0, -2.0], [20.0, -20.0], [200.0, -200.0]],
                          grad_vals[0])

  def _testTensorArrayGradientDynamicUnpackRead(self):
    with self.test_session(use_gpu=True) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=0,
          dynamic_size=True)

      value = constant_op.constant([[1.0, -1.0], [10.0, -10.0]])

      w = ta.unstack(value)
      r0 = w.read(0)
      r1 = w.read(1)

      # Test combined gradients + aggregation of read(0)
      grad = gradients_impl.gradients(
          ys=[r0, r1], xs=[value], grad_ys=[[2.0, 3.0], [4.0, 5.0]])
      grad_vals = session.run(grad)

      self.assertEqual(len(grad_vals), 1)
      self.assertAllEqual([[2.0, 3.0], [4.0, 5.0]], grad_vals[0])

  def testTensorArrayGradientDynamicUnpackRead(self):
    self._testTensorArrayGradientDynamicUnpackRead()

  def testCloseTensorArray(self):
    with self.test_session(use_gpu=True) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=3)
      c1 = ta.close()
      session.run(c1)

  def testSizeTensorArray(self):
    with self.test_session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=3)
      s = ta.size()
      self.assertAllEqual(3, s.eval())

  def testWriteCloseTensorArray(self):
    with self.test_session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=3,
          infer_shape=False)
      w0 = ta.write(0, [[4.0, 5.0]])
      w1 = w0.write(1, [3.0])
      w1.close().run()  # Expected to run without problems

  def _testWhileLoopWritePackGradients(self, dynamic_size, dtype):
    np_dtype = dtype.as_numpy_dtype
    with self.test_session(use_gpu=True) as session:
      v0 = array_ops.identity(np.arange(3 * 5, dtype=np_dtype).reshape(3, 5))
      var = variables.Variable(np.arange(100, 105, dtype=np_dtype))
      state0 = array_ops.identity(np.array([1] * 5, dtype=np_dtype))
      ta = tensor_array_ops.TensorArray(
          dtype=dtype,
          tensor_array_name="foo",
          size=0 if dynamic_size else 3,
          dynamic_size=dynamic_size)
      time_0 = array_ops.identity(0)

      def body(time, ta_t, state):
        sliced = array_ops.slice(
            v0, begin=array_ops.stack([time, 0]), size=[1, -1])
        sliced = array_ops.squeeze(sliced)
        out = sliced + var + state
        state += sliced
        ta_t = ta_t.write(time, out)
        return (time + 1, ta_t, state)

      (unused_0, h_final, unused_2) = control_flow_ops.while_loop(
          cond=lambda time, unused_1, unused_2: time < 3,
          body=body,
          loop_vars=(time_0, ta, state0),
          shape_invariants=(time_0.get_shape(), tensor_shape.unknown_shape(),
                            tensor_shape.unknown_shape()),
          parallel_iterations=3)
      vout = h_final.stack()

      grad_val = -np.arange(3 * 5, dtype=np_dtype).reshape(3, 5)
      v0_grad = gradients_impl.gradients([vout], [v0], [grad_val])[0]
      state0_grad = gradients_impl.gradients([vout], [state0], [grad_val])[0]
      var_grad = gradients_impl.gradients([vout], [var], [grad_val])[0]

      variables.global_variables_initializer().run()
      state0_t, var_t, v0_t, vout_t, v0_grad_t, var_grad_t, state0_grad_t = (
          session.run([state0, var, v0, vout, v0_grad, var_grad, state0_grad]))
      just_v0_grad_t, = session.run([v0_grad])

      # state = [ state0 | state0 + v0[0] | state0 + v0[0] + v0[1] ]
      # vout = [ v0[0] + var + state[0] |
      #          v0[1] + var + state[1] |
      #          v0[2] + var + state[2] ]
      #      = [ v0[0] + var + state0 |
      #          v0[1] + var + state0 + v0[0] |
      #          v0[2] + var + state0 + v0[0] + v0[1] ]
      #
      # d(vout[0])/d(v0) = [1 | 0 | 0 ]
      # d(vout[1])/d(v0) = [1 | 1 | 0 ]
      # d(vout[2])/d(v0) = [1 | 1 | 1 ]
      # d(vout)/d(var) = [1 | 1 | 1]
      # d(vout)/d(state0) = [ 1 | 1 | 1 ]

      state_per_time = np.array(
          [state0_t, state0_t + v0_t[0, :], state0_t + v0_t[0, :] + v0_t[1, :]])

      # Compare forward prop
      self.assertAllClose(v0_t + var_t + state_per_time, vout_t)

      # Compare backward prop
      expected_v0_grad_t = np.array([
          grad_val[0, :] + grad_val[1, :] + grad_val[2, :],
          grad_val[1, :] + grad_val[2, :], grad_val[2, :]
      ])

      self.assertAllEqual(expected_v0_grad_t, v0_grad_t)
      self.assertAllEqual(expected_v0_grad_t, just_v0_grad_t)
      self.assertAllClose(grad_val.sum(axis=0), var_grad_t)
      self.assertAllClose(grad_val.sum(axis=0), state0_grad_t)

  def testWhileLoopWritePackGradients(self):
    self._testWhileLoopWritePackGradients(
        dynamic_size=False, dtype=dtypes.float32)
    # TODO(ebrevdo): re-enable when While supports non-float32 gradients.
    # self._testWhileLoopWritePackGradients(
    #     dynamic_size=False, dtype=tf.int64)

  def testWhileLoopDynamicWritePackGradients(self):
    self._testWhileLoopWritePackGradients(
        dynamic_size=True, dtype=dtypes.float32)

  def testGradSerialTwoLoops(self):
    with self.test_session():
      num_steps = 100
      acc = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          size=num_steps,
          clear_after_read=False,
          element_shape=tensor_shape.scalar())
      i = constant_op.constant(0, name="i")
      x = constant_op.constant(2.0, name="x")

      c = lambda i, acc: i < 5

      def b(i, acc):
        x1 = control_flow_ops.cond(
            math_ops.equal(i, 0), lambda: x,
            lambda: math_ops.multiply(acc.read(i - 1), 2.0))
        return i + 1, acc.write(i, x1)

      i1, acc1 = control_flow_ops.while_loop(c, b, [i, acc])

      z = constant_op.constant(0.0)

      def fn(i, acc):
        return i + 1, acc.write(i, z)

      _, acc2 = control_flow_ops.while_loop(lambda i, acc: i < num_steps, fn,
                                            [i1, acc1])

      r = acc2.stack()
      grad = gradients_impl.gradients(r, [x])[0]
      self.assertAllClose(31.0, grad.eval())

  def testSumOfTwoReadVariablesWithoutRepeatGrad(self):
    with self.test_session(use_gpu=True) as session:
      a = array_ops.identity(
          np.arange(
              3 * 5, dtype=np.float32).reshape(3, 5) + 1)
      b = array_ops.identity(
          np.arange(
              3 * 5, dtype=np.float32).reshape(3, 5) + 1 + 3 * 5)
      ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=2)
      ta = ta.write(0, a, name="write_a")
      ta = ta.write(1, b, name="write_b")
      c = (
          ta.read(
              0, name="read_a_0") +  # a + b
          ta.read(
              1, name="read_b_0"))
      g0 = -(np.arange(3 * 5, dtype=np.float32).reshape(3, 5) + 1)
      grad_a = gradients_impl.gradients([c], [a], [g0])[0]  # d(a+b)/da = 1
      grad_b = gradients_impl.gradients([c], [b], [g0])[0]  # d(a+b)/db = 1

      # Test gradients calculated individually
      grad_a_t, = session.run([grad_a])
      self.assertAllEqual(grad_a_t, g0)

      grad_b_t, = session.run([grad_b])
      self.assertAllEqual(grad_b_t, g0)

      # Test gradients calculated jointly
      joint_grad_a_t, joint_grad_b_t = session.run([grad_a, grad_b])
      self.assertAllEqual(joint_grad_a_t, g0)
      self.assertAllEqual(joint_grad_b_t, g0)

  def _grad_source_for_name(self, name):
    return tensor_array_grad._GetGradSource(constant_op.constant(0, name=name))

  def testGetGradSource_Invalid(self):
    with self.assertRaises(ValueError):
      self._grad_source_for_name("")
    with self.assertRaises(ValueError):
      self._grad_source_for_name("foo")
    with self.assertRaises(ValueError):
      self._grad_source_for_name("foo/bar")

  def testGetGradSource_NoEnclosingScope(self):
    self.assertEqual("gradients:0", self._grad_source_for_name("gradients"))
    self.assertEqual("gradients_0:0", self._grad_source_for_name("gradients_0"))
    self.assertEqual("gradients", self._grad_source_for_name("gradients/foo"))
    self.assertEqual("gradients_0",
                     self._grad_source_for_name("gradients_0/foo"))
    self.assertEqual("gradients",
                     self._grad_source_for_name("gradients/foo/bar"))
    self.assertEqual("gradients_0",
                     self._grad_source_for_name("gradients_0/foo/bar"))

  def testGetGradSource_EnclosingScope(self):
    self.assertEqual("foo/gradients:0",
                     self._grad_source_for_name("foo/gradients"))
    self.assertEqual("foo/gradients_0:0",
                     self._grad_source_for_name("foo/gradients_0"))
    self.assertEqual("foo/gradients",
                     self._grad_source_for_name("foo/gradients/bar"))
    self.assertEqual("foo/gradients_0",
                     self._grad_source_for_name("foo/gradients_0/bar"))
    self.assertEqual("foo/bar/gradients",
                     self._grad_source_for_name("foo/bar/gradients/baz"))
    self.assertEqual("foo/bar/gradients_0",
                     self._grad_source_for_name("foo/bar/gradients_0/baz"))

  def testGetGradSource_NestedUsesInnermost(self):
    self.assertEqual(
        "foo/gradients/bar/gradients_0",
        self._grad_source_for_name("foo/gradients/bar/gradients_0/baz"))

  def testWriteShape(self):
    with self.test_session():
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=3)
      c0 = constant_op.constant([4.0, 5.0])
      w0 = ta.write(0, c0)
      r0 = w0.read(0)
      self.assertAllEqual(c0.get_shape(), r0.get_shape())

      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=3)
      c1 = constant_op.constant([6.0, 7.0])
      w1 = w0.write(1, c1)
      r0 = w1.read(0)
      r1 = w1.read(1)
      self.assertAllEqual(c0.get_shape(), r0.get_shape())
      self.assertAllEqual(c1.get_shape(), r1.get_shape())

      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=3)
      c2 = constant_op.constant([4.0, 5.0, 6.0])
      with self.assertRaises(ValueError):
        w0.write(0, c2)

  def testPartlyUnknownShape(self):
    with self.test_session():
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, tensor_array_name="foo", size=6)

      c0 = array_ops.placeholder(dtypes.float32, [None, None, None, 3])
      w0 = ta.write(0, c0)
      r0 = w0.read(0)
      self.assertAllEqual([None, None, None, 3], r0.get_shape().as_list())

      c1 = array_ops.placeholder(dtypes.float32, [None, None, None, 3])
      w1 = w0.write(1, c1)
      r1 = w1.read(0)
      self.assertAllEqual([None, None, None, 3], r1.get_shape().as_list())

      # Writing less specific shape (doesn't change type.)
      c2 = array_ops.placeholder(dtypes.float32, [None, None, None, None])
      w2 = w1.write(2, c2)
      r2 = w2.read(0)
      self.assertAllEqual([None, None, None, 3], r2.get_shape().as_list())

      # Writing more specific shape in one dimension and less specific in
      # another.
      c3 = array_ops.placeholder(dtypes.float32, [None, None, 2, None])
      w3 = w2.write(3, c3)
      r3 = w3.read(0)
      self.assertAllEqual([None, None, 2, 3], r3.get_shape().as_list())

      # Writing partly defined shape using TensorArray.scatter.
      c4 = array_ops.placeholder(dtypes.float32, [2, None, 4, 2, 3])
      w4 = w3.scatter([4, 5], c4)
      r4 = w4.read(0)
      self.assertAllEqual([None, 4, 2, 3], r4.get_shape().as_list())

      # Writing fully defined shape using TensorArray.split.
      c5 = array_ops.placeholder(dtypes.float32, [10, 4, 2, 3])
      w5 = w4.split(c5, constant_op.constant([5, 5]))
      r5 = w5.read(0)
      self.assertAllEqual([5, 4, 2, 3], r5.get_shape().as_list())

  def _testUnpackShape(self):
    with self.test_session():
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=0,
          dynamic_size=True,
          infer_shape=True)
      value = constant_op.constant(
          [[1.0, -1.0], [10.0, -10.0], [100.0, -100.0]])
      w0 = ta.unstack(value)
      r0 = w0.read(0)
      self.assertAllEqual((2,), r0.get_shape())

      c1 = constant_op.constant([4.0, 5.0])
      w1 = w0.write(3, c1)
      r1 = w1.read(0)
      self.assertAllEqual(c1.get_shape(), r1.get_shape())

      c2 = constant_op.constant([4.0, 5.0, 6.0])
      with self.assertRaises(ValueError):
        w1.write(4, c2)

  def testUnpackShape(self):
    self._testUnpackShape()

  def testSplitShape(self):
    with self.test_session():
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=0,
          dynamic_size=True,
          infer_shape=True)
      value = constant_op.constant([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]])
      w0 = ta.split(value, [1, 1, 1])
      r0 = w0.read(0)
      self.assertAllEqual((1, 2), r0.get_shape())

      ta1 = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo1",
          size=0,
          dynamic_size=True,
          infer_shape=True)
      w0 = ta1.split(value, [1, 2])
      r0 = w0.read(0)
      self.assertAllEqual(r0.get_shape(), tensor_shape.unknown_shape())

  def testWriteUnknownShape(self):
    with self.test_session():
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=3,
          infer_shape=True)
      c0 = array_ops.placeholder(dtypes.float32)
      w0 = ta.write(0, c0)
      r0 = w0.read(0)
      self.assertAllEqual(r0.get_shape(), tensor_shape.unknown_shape())

  def _testGradientWhenNotAllComponentsRead(self):
    with self.test_session(use_gpu=True) as session:
      ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=2)
      x = constant_op.constant([2.0, 3.0])
      w = ta.unstack(x)
      r0 = w.read(0)
      # calculate (dr0/dx0, dr0/dx1).  since r0 = x0, gradients are (1, 0).
      grad_r0 = gradients_impl.gradients(ys=[r0], xs=[x], grad_ys=[1.0])
      grad_r0_vals = session.run(grad_r0)[0]
      self.assertAllEqual(grad_r0_vals, [1.0, 0.0])

  def testGradientWhenNotAllComponentsRead(self):
    self._testGradientWhenNotAllComponentsRead()

  def _testTensorArrayUnpackDynamic(self):
    with self.test_session(use_gpu=True) as sess:
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, size=3, dynamic_size=True)
      x = constant_op.constant([1.0, 2.0, 3.0])
      w0 = ta.unstack(x)
      w1 = w0.write(3, 4.0)
      r = w1.stack()
      self.assertAllEqual(np.array([1.0, 2.0, 3.0, 4.0]), r.eval())
      grad = gradients_impl.gradients(ys=[r], xs=[x])
      self.assertAllEqual(np.array([1.0, 1.0, 1.0]), sess.run(grad)[0])

  def testTensorArrayUnpackDynamic(self):
    self._testTensorArrayUnpackDynamic()

  def testTensorArraySplitDynamic(self):
    with self.test_session(use_gpu=True) as sess:
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, size=3, dynamic_size=True)
      x = constant_op.constant([1.0, 2.0, 3.0])
      w0 = ta.split(x, [1, 1, 1])
      w1 = w0.write(3, [4.0])
      r = w1.concat()
      self.assertAllEqual(np.array([1.0, 2.0, 3.0, 4.0]), r.eval())
      grad = gradients_impl.gradients(ys=[r], xs=[x])
      self.assertAllEqual(np.array([1.0, 1.0, 1.0]), sess.run(grad)[0])

  def _testTensorArrayEvalEmpty(self):
    with self.test_session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, size=0, dynamic_size=False, infer_shape=False)
      with self.assertRaisesOpError(
          "TensorArray has size zero, but element shape <unknown> is not fully "
          "defined. Currently only static shapes are supported when packing "
          "zero-size TensorArrays."):
        ta.stack().eval()

  def testTensorArrayEvalEmpty(self):
    self._testTensorArrayEvalEmpty()

  def _testTensorArrayEvalEmptyWithDefault(self):
    with self.test_session(use_gpu=True):
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, size=0, dynamic_size=False, infer_shape=True)
      self.assertEqual(0, ta.size().eval())
      # Don't actually perform the pack.  This stores the static shape.
      ta.unstack(array_ops.zeros([0, 3, 5]))
      packed = ta.stack()
      self.assertAllEqual([0, 3, 5], packed.eval().shape)
      # Concatenating zero tensors along their first dimension gives a
      # first dimension of zero
      self.assertAllEqual([0, 5], ta.concat().eval().shape)

  def testTensorArrayEvalEmptyWithDefault(self):
    self._testTensorArrayEvalEmptyWithDefault()

  def testTensorArrayScatterReadAndGradients(self):
    with self.test_session(use_gpu=True) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=0,
          dynamic_size=True)

      indices = constant_op.constant([1, 8])
      value = constant_op.constant([[1.0, -1.0], [10.0, -10.0]])

      w = ta.scatter(indices, value)
      r0 = w.read(1)
      r1 = w.read(8)

      # Test combined gradients + aggregation of read(0)
      grad = gradients_impl.gradients(
          ys=[r0, r1], xs=[value], grad_ys=[[2.0, 3.0], [4.0, 5.0]])
      read_vals, grad_vals = session.run([[r0, r1], grad])

      self.assertEqual(len(read_vals), 2)
      self.assertEqual(len(grad_vals), 1)
      self.assertAllEqual([1.0, -1.0], read_vals[0])
      self.assertAllEqual([10.0, -10.0], read_vals[1])
      self.assertAllEqual([[2.0, 3.0], [4.0, 5.0]], grad_vals[0])

  def testTensorArrayWriteGatherAndGradients(self):
    with self.test_session(use_gpu=True) as session:
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=0,
          dynamic_size=True)

      values = constant_op.constant([[1.0 * x, -1.0 * x] for x in range(10)])
      indices = constant_op.constant([1, 8])

      w = ta.unstack(values)
      g = w.gather(indices)

      # Test combined gradients + aggregation of read(0)
      grad = gradients_impl.gradients(
          ys=[g], xs=[values], grad_ys=[[[2.0, 3.0], [4.0, 5.0]]])
      g_vals, grad_vals = session.run([[g], grad])

      # Gradients for 8 of the 10 unread components are zero.
      expected_grad = np.zeros((10, 2))
      expected_grad[1] = [2.0, 3.0]
      expected_grad[8] = [4.0, 5.0]

      self.assertEqual(len(g_vals), 1)
      self.assertEqual(len(grad_vals), 1)
      self.assertAllEqual([[1.0, -1.0], [8.0, -8.0]], g_vals[0])
      self.assertAllEqual(expected_grad, grad_vals[0])

  def testTensorArrayGetsDeviceFromFirstWrite(self):
    with ops.device("/gpu:1"):
      ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=2)
    # parent device was ignored when creating the TensorArray
    self.assertEqual(ta.handle.device, "")
    self.assertEqual(ta.flow.device, "")
    with ops.device("/gpu:0"):
      # the first write sets the op's device
      ta = ta.write(0, 1.0)
    self.assertTrue("gpu:0" in ta.handle.device.lower())
    self.assertTrue("gpu:0" in ta.flow.device.lower())
    with ops.device("/gpu:1"):
      # subsequent writes do not modify the op's device
      ta = ta.write(1, 1.0)
    self.assertTrue("gpu:0" in ta.handle.device.lower())
    self.assertTrue("gpu:0" in ta.flow.device.lower())

    ta_grad = ta.grad("grad")
    self.assertTrue("gpu:0" in ta_grad.handle.device.lower())
    self.assertTrue("gpu:0" in ta_grad.flow.device.lower())

    # Similar tests for unpack and split
    ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=2)
    self.assertEqual(ta.handle.device, "")
    self.assertEqual(ta.flow.device, "")
    with ops.device("/gpu:0"):
      ta = ta.unstack([1.0, 2.0])
    self.assertTrue("gpu:0" in ta.handle.device.lower())
    self.assertTrue("gpu:0" in ta.flow.device.lower())
    with ops.device("/gpu:1"):
      ta = ta.unstack([1.0, 2.0])
    self.assertTrue("gpu:0" in ta.handle.device.lower())
    self.assertTrue("gpu:0" in ta.flow.device.lower())

    ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=2)
    self.assertEqual(ta.handle.device, "")
    self.assertEqual(ta.flow.device, "")
    with ops.device("/gpu:0"):
      ta = ta.split([1.0, 2.0], [1, 1])
    self.assertTrue("gpu:0" in ta.handle.device.lower())
    self.assertTrue("gpu:0" in ta.flow.device.lower())
    with ops.device("/gpu:1"):
      ta = ta.split([1.0, 2.0], [1, 1])
    self.assertTrue("gpu:0" in ta.handle.device.lower())
    self.assertTrue("gpu:0" in ta.flow.device.lower())

  def testTensorArrayGetsDeviceFromFirstWriteInWhileLoop(self):
    ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=2)

    def _body(i, ta_i):
      with ops.device("/gpu:0"):
        return i + 1, ta_i.write(i, 0.0)

    self.assertEqual(ta.handle.device, "")
    self.assertEqual(ta.flow.device, "")

    _, ta_out = control_flow_ops.while_loop(
        lambda i, ta: i < 2, _body, loop_vars=[0, ta])

    self.assertTrue("gpu:0" in ta_out.handle.device.lower())
    self.assertTrue("gpu:0" in ta.handle.device.lower())

  def testTensorArrayLazyDeviceSettingDoesNotConfuseInitialAccess(self):
    with self.test_session(use_gpu=True) as session:
      ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=2)
      self.assertEqual(ta.handle.device, "")

      with ops.device("/cpu:0"):
        size = ta.size()
      with ops.device("/gpu:0"):
        ta = ta.write(0, 0.0)

      self.assertTrue("gpu:0" in ta.handle.device.lower())

      # This should use the TensorArray on /gpu:0
      size_value, _ = session.run((size, ta.flow))
      self.assertEqual(2, size_value)

  def testTensorArrayIdentity(self):
    with self.test_session() as session:
      ta0 = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=2,
                                         infer_shape=False)
      ta1 = tensor_array_ops.TensorArray(dtype=dtypes.int32, size=4,
                                         infer_shape=True)

      ta0 = ta0.write(0, 0.)
      ta1 = ta1.write(0, 1)

      v0 = variables.Variable(0)
      v1 = variables.Variable(0)

      with ops.control_dependencies([v0.assign_add(1)]):
        ta0 = ta0.identity()

      with ops.control_dependencies([v1.assign_add(1)]):
        ta1 = ta1.identity()

      read0 = ta0.read(0)
      read1 = ta1.read(0)

      size0 = ta0.size()
      size1 = ta1.size()

      # Tests correct properties on new TensorArrays.
      self.assertEqual(dtypes.float32, ta0.dtype)
      self.assertEqual(dtypes.int32, ta1.dtype)
      self.assertEqual(tensor_shape.unknown_shape(), read0.get_shape())
      self.assertEqual(tensor_shape.scalar(), read1.get_shape())

      variables.global_variables_initializer().run()

      read0_v, read1_v, size0_v, size1_v = session.run(
          (read0, read1, size0, size1))

      # Tests that the control dependencies was added and executed.
      self.assertEqual(1, v0.eval())
      self.assertEqual(1, v1.eval())

      # Tests correct TensorArray.
      self.assertEqual(read0_v, 0)
      self.assertEqual(read1_v, 1)
      self.assertEqual(size0_v, 2)
      self.assertEqual(size1_v, 4)


if __name__ == "__main__":
  test.main()
