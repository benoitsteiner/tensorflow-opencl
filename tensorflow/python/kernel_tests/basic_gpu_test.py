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
"""Functional tests for basic component wise operations using a GPU device."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import math
import numpy as np
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops.gen_array_ops import _broadcast_gradient_args

class GPUBinaryOpsTest(tf.test.TestCase):
  def _compareGPU(self, x, y, np_func, tf_func):
    with self.test_session(use_gpu=True) as sess:
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_gpu = sess.run(out)

    with self.test_session(use_gpu=False) as sess:
      inx = tf.convert_to_tensor(x)
      iny = tf.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_cpu = sess.run(out)

    self.assertAllClose(tf_cpu, tf_gpu)
    
  def testFloatBasic(self):
    x = np.linspace(-5, 20, 15).reshape(1, 3, 5).astype(np.float32)
    y = np.linspace(20, -5, 15).reshape(1, 3, 5).astype(np.float32)
    self._compareGPU(x, y, np.add, tf.add)
    self._compareGPU(x, y, np.subtract, tf.sub)
    self._compareGPU(x, y, np.multiply, tf.mul)
    self._compareGPU(x, y + 0.1, np.true_divide, tf.truediv)
    self._compareGPU(x, y + 0.1, np.floor_divide, tf.floordiv)
    self._compareGPU(x, y, np.power, tf.pow)

  def _GetGradientArgs(self, xs, ys):
    with self.test_session(use_gpu=True) as sess:
      return sess.run(_broadcast_gradient_args(xs, ys))

  def testBroadcast(self):
    r0, r1 = self._GetGradientArgs([2, 3, 5], [1])
    self.assertAllEqual(r0, [])
    self.assertAllEqual(r1, [0, 1, 2])

class MathBuiltinUnaryTest(tf.test.TestCase):

  def _compare(self, x, np_func, tf_func, use_gpu):
    np_out = np_func(x)
    with self.test_session(use_gpu=use_gpu) as sess:
     inx = tf.convert_to_tensor(x)
     ofunc = tf_func(inx)
     tf_out = sess.run(ofunc)
    self.assertAllClose(np_out, tf_out)

  def _inv(self, x):
    return 1.0 / x

  def _rsqrt(self, x):
    return self._inv(np.sqrt(x))

  def _testDtype(self, dtype, use_gpu):
    data = (np.arange(-3, 3) / 4.).reshape([1, 3, 2]).astype(dtype)
    self._compare(data, np.abs, tf.abs, use_gpu)
    self._compare(data, np.arcsin, tf.asin, use_gpu)
    self._compare(data, np.arctan, tf.atan, use_gpu)
    self._compare(data, np.ceil, tf.ceil, use_gpu)
    self._compare(data, np.cos, tf.cos, use_gpu)
    self._compare(data, np.exp, tf.exp, use_gpu)
    self._compare(data, np.floor, tf.floor, use_gpu)
    self._compare(data, np.log, tf.log, use_gpu)
    self._compare(data, np.log1p, tf.log1p, use_gpu)
    self._compare(data, np.negative, tf.neg, use_gpu)
    self._compare(data, self._rsqrt, tf.rsqrt, use_gpu)
    self._compare(data, np.sin, tf.sin, use_gpu)
    self._compare(data, np.sqrt, tf.sqrt, use_gpu)
    self._compare(data, np.square, tf.square, use_gpu)
    self._compare(data, np.tan, tf.tan, use_gpu)
    self._compare(data, np.tanh, tf.tanh, use_gpu)

  def testTypes(self):
    for dtype in [np.float32]:
      self._testDtype(dtype, use_gpu=True)

class IsFiniteInfNanTest(tf.test.TestCase):

  def _compare(self, x, use_gpu):
    np_finite, np_inf, np_nan = np.isfinite(x), np.isinf(x), np.isnan(x)
    with self.test_session(use_gpu=use_gpu) as sess:
      inx = tf.convert_to_tensor(x)
      ofinite, oinf, onan = tf.is_finite(inx), tf.is_inf(
          inx), tf.is_nan(inx)
      tf_finite, tf_inf, tf_nan = sess.run([ofinite, oinf, onan])
    self.assertAllEqual(np_inf, tf_inf)
    self.assertAllEqual(np_nan, tf_nan)
    self.assertAllEqual(np_finite, tf_finite)
    self.assertShapeEqual(np_inf, oinf)
    self.assertShapeEqual(np_nan, onan)
    self.assertShapeEqual(np_finite, ofinite)

  def _testDtype(self, dtype):
    fi = np.finfo(dtype)
    data = np.array([0, -1, 1, fi.resolution, -fi.resolution, fi.min, fi.max,
                     -np.inf, np.inf, np.nan]).astype(dtype)
    self._compare(data, use_gpu=False)
    self._compare(data, use_gpu=True)

  def testFloat(self):
    self._testDtype(np.float32)

  def testSqrt(self):
    for dtype in [np.float32]:
      fi = np.finfo(dtype)
      for size in [1, 3, 4, 7, 8, 63, 64, 65]:
        # For float32 Eigen uses Carmack's fast vectorized sqrt algorithm.
        # It is not accurate for very large arguments, so we test for
        # fi.max/100 instead of fi.max here.
        for value in [fi.min, -2, -1, 0, fi.tiny, 1, 2, 1000, fi.max/100]:
          x = np.full((size,), value, dtype=dtype)
          np_y = np.sqrt(x)
          np_nan = np.isnan(np_y)
          with self.test_session(use_gpu=True):
            tf_y = tf.sqrt(x)
            tf_nan = tf.is_nan(tf_y)
            if value < 0:
              self.assertAllEqual(np_nan, tf_nan.eval())
            else:
              self.assertAllCloseAccordingToType(np_y, tf_y.eval())

if __name__ == "__main__":
  tf.test.main()
