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
"""Tests for tensorflow.ops.linalg_grad."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def _AddTest(test, op_name, testcase_name, fn):
  test_name = '_'.join(['test', op_name, testcase_name])
  if hasattr(test, test_name):
    raise RuntimeError('Test %s defined more than once' % test_name)
  setattr(test, test_name, fn)


class ShapeTest(tf.test.TestCase):

  def testBatchGradientUnknownSize(self):
    with self.test_session():
      batch_size = tf.constant(3)
      matrix_size = tf.constant(4)
      batch_identity = tf.tile(
          tf.expand_dims(tf.diag(tf.ones([matrix_size])), 0),
          [batch_size, 1, 1])
      determinants = tf.matrix_determinant(batch_identity)
      reduced = tf.reduce_sum(determinants)
      sum_grad = tf.gradients(reduced, batch_identity)[0]
      self.assertAllClose(batch_identity.eval(), sum_grad.eval())


class MatrixUnaryFunctorGradientTest(tf.test.TestCase):
  pass  # Filled in below


def _GetMatrixUnaryFunctorGradientTest(functor_, dtype_, shape_, **kwargs_):

  def Test(self):
    with self.test_session():
      np.random.seed(1)
      a_np = np.random.uniform(
          low=-1.0, high=1.0,
          size=np.prod(shape_)).reshape(shape_).astype(dtype_)
      a = tf.constant(a_np)
      b = functor_(a, **kwargs_)

      # Optimal stepsize for central difference is O(epsilon^{1/3}).
      epsilon = np.finfo(dtype_).eps
      delta = epsilon**(1.0 / 3.0)
      # tolerance obtained by looking at actual differences using
      # np.linalg.norm(theoretical-numerical, np.inf) on -mavx build
      tol = 1e-6 if dtype_ == np.float64 else 0.05

      theoretical, numerical = tf.test.compute_gradient(
          a,
          a.get_shape().as_list(),
          b,
          b.get_shape().as_list(),
          x_init_value=a_np,
          delta=delta)
      self.assertAllClose(theoretical, numerical, atol=tol, rtol=tol)

  return Test


class MatrixBinaryFunctorGradientTest(tf.test.TestCase):
  pass  # Filled in below


def _GetMatrixBinaryFunctorGradientTest(functor_,
                                        dtype_,
                                        shape_,
                                        float32_tol_fudge=1.0,
                                        **kwargs_):

  def Test(self):
    with self.test_session():
      np.random.seed(1)
      a_np = np.random.uniform(
          low=-1.0, high=1.0,
          size=np.prod(shape_)).reshape(shape_).astype(dtype_)
      a = tf.constant(a_np)

      b_np = np.random.uniform(
          low=-1.0, high=1.0,
          size=np.prod(shape_)).reshape(shape_).astype(dtype_)
      b = tf.constant(b_np)
      c = functor_(a, b, **kwargs_)

      # Optimal stepsize for central difference is O(epsilon^{1/3}).
      epsilon = np.finfo(dtype_).eps
      delta = epsilon**(1.0 / 3.0)
      # tolerance obtained by looking at actual differences using
      # np.linalg.norm(theoretical-numerical, np.inf) on -mavx build
      tol = 1e-6 if dtype_ == np.float64 else float32_tol_fudge * 0.04
      # The gradients for a and b may be of very different magnitudes,
      # so to not get spurious failures we test them separately.
      for factor, factor_init in [a, a_np], [b, b_np]:
        theoretical, numerical = tf.test.compute_gradient(
            factor,
            factor.get_shape().as_list(),
            c,
            c.get_shape().as_list(),
            x_init_value=factor_init,
            delta=delta)
        self.assertAllClose(theoretical, numerical, atol=tol, rtol=tol)

  return Test


if __name__ == '__main__':
  # Tests for gradients of binary matrix operations.
  for dtype in np.float32, np.float64:
    for size in 2, 5, 10:
      # We skip the rank 4, size 10 case: it is slow and conceptually covered
      # by the other cases.
      for extra in [(), (2,), (3,)] + [(3, 2)] * (size < 10):
        for adjoint in False, True:
          shape = extra + (size, size)
          name = '%s_%s_adj_%s' % (dtype.__name__, '_'.join(map(str, shape)),
                                   str(adjoint))
          _AddTest(
              MatrixBinaryFunctorGradientTest,
              'MatrixSolveGradient',
              name,
              _GetMatrixBinaryFunctorGradientTest(
                  tf.matrix_solve, dtype, shape, adjoint=adjoint))
          if dtype == np.float64:
            # TODO(rmlarsen): The gradients of triangular solves seems
            # particularly sensitive to round-off when computed in float32.
            # In some tests, a few gradient elements differ by 25% between the
            # numerical and theoretical values. Disable tests for float32 until
            # we understand this better.
            _AddTest(
                MatrixBinaryFunctorGradientTest,
                'MatrixTriangularSolveGradient',
                name + '_low_True',
                _GetMatrixBinaryFunctorGradientTest(
                    tf.matrix_triangular_solve,
                    dtype,
                    shape,
                    adjoint=adjoint,
                    lower=True))
            _AddTest(
                MatrixBinaryFunctorGradientTest,
                'MatrixTriangularSolveGradient',
                name + '_low_False',
                _GetMatrixBinaryFunctorGradientTest(
                    tf.matrix_triangular_solve,
                    dtype,
                    shape,
                    adjoint=adjoint,
                    lower=False))

  # Tests for gradients of unary matrix operations.
  for dtype in np.float32, np.float64:
    for size in 2, 5, 10:
      # We skip the rank 4, size 10 case: it is slow and conceptually covered
      # by the other cases.
      for extra in [(), (2,), (3,)] + [(3, 2)] * (size < 10):
        shape = extra + (size, size)
        name = '%s_%s' % (dtype.__name__, '_'.join(map(str, shape)))
        _AddTest(MatrixUnaryFunctorGradientTest, 'MatrixInverseGradient', name,
                 _GetMatrixUnaryFunctorGradientTest(tf.matrix_inverse, dtype,
                                                    shape))
        _AddTest(MatrixUnaryFunctorGradientTest, 'MatrixDeterminantGradient',
                 name,
                 _GetMatrixUnaryFunctorGradientTest(tf.matrix_determinant,
                                                    dtype, shape))

  # Tests for gradients of matrix_solve_ls
  for dtype in np.float32, np.float64:
    for rows in 2, 5, 10:
      for cols in 2, 5, 10:
        for l2_regularization in 0.0, 0.001, 1.0:
          shape = (rows, cols)
          name = '%s_%s_%s' % (dtype.__name__, '_'.join(map(str, shape)),
                               l2_regularization)
          _AddTest(
              MatrixBinaryFunctorGradientTest,
              'MatrixSolveLsGradient',
              name,
              _GetMatrixBinaryFunctorGradientTest(
                  lambda a, b, l=l2_regularization: tf.matrix_solve_ls(a, b, l),
                  dtype,
                  shape,
                  float32_tol_fudge=4.0))

  tf.test.main()
