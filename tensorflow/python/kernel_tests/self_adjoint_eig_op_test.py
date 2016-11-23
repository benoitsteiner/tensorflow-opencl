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
"""Tests for tensorflow.ops.math_ops.matrix_inverse."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class SelfAdjointEigTest(tf.test.TestCase):

  def testWrongDimensions(self):
    # The input to self_adjoint_eig should be a tensor of
    # at least rank 2.
    scalar = tf.constant(1.)
    with self.assertRaises(ValueError):
      tf.self_adjoint_eig(scalar)
    vector = tf.constant([1., 2.])
    with self.assertRaises(ValueError):
      tf.self_adjoint_eig(vector)


def SortEigenDecomposition(e, v):
  if v.ndim < 2:
    return e, v
  else:
    perm = np.argsort(e, -1)
    return np.take(e, perm, -1), np.take(v, perm, -1)


def _GetSelfAdjointEigTest(dtype_, shape_):

  def CompareEigenVectors(self, x, y, tol):
    # Eigenvectors are only unique up to sign so we normalize the signs first.
    signs = np.sign(np.sum(np.divide(x, y), -2, keepdims=True))
    x *= signs
    self.assertAllClose(x, y, atol=tol, rtol=tol)

  def CompareEigenDecompositions(self, x_e, x_v, y_e, y_v, tol):
    num_batches = int(np.prod(x_e.shape[:-1]))
    n = x_e.shape[-1]
    x_e = np.reshape(x_e, [num_batches] + [n])
    x_v = np.reshape(x_v, [num_batches] + [n, n])
    y_e = np.reshape(y_e, [num_batches] + [n])
    y_v = np.reshape(y_v, [num_batches] + [n, n])
    for i in range(num_batches):
      x_ei, x_vi = SortEigenDecomposition(x_e[i, :], x_v[i, :, :])
      y_ei, y_vi = SortEigenDecomposition(y_e[i, :], y_v[i, :, :])
      self.assertAllClose(x_ei, y_ei, atol=tol, rtol=tol)
      CompareEigenVectors(self, x_vi, y_vi, tol)

  def Test(self):
    np.random.seed(1)
    n = shape_[-1]
    batch_shape = shape_[:-2]
    a = np.random.uniform(
        low=-1.0, high=1.0, size=n * n).reshape([n, n]).astype(dtype_)
    a += a.T
    a = np.tile(a, batch_shape + (1, 1))
    if dtype_ == np.float32:
      atol = 1e-4
    else:
      atol = 1e-12
    for compute_v in False, True:
      np_e, np_v = np.linalg.eig(a)
      with self.test_session():
        if compute_v:
          tf_e, tf_v = tf.self_adjoint_eig(tf.constant(a))

          # Check that V*diag(E)*V^T is close to A.
          a_ev = tf.matmul(
              tf.matmul(tf_v, tf.matrix_diag(tf_e)), tf_v, adjoint_b=True)
          self.assertAllClose(a_ev.eval(), a, atol=atol)

          # Compare to numpy.linalg.eig.
          CompareEigenDecompositions(self, np_e, np_v, tf_e.eval(), tf_v.eval(),
                                     atol)
        else:
          tf_e = tf.self_adjoint_eigvals(tf.constant(a))
          self.assertAllClose(
              np.sort(np_e, -1), np.sort(tf_e.eval(), -1), atol=atol)

  return Test


class SelfAdjointEigGradTest(tf.test.TestCase):
  pass  # Filled in below


def _GetSelfAdjointEigGradTest(dtype_, shape_):

  def Test(self):
    np.random.seed(1)
    n = shape_[-1]
    batch_shape = shape_[:-2]
    a = np.random.uniform(
        low=-1.0, high=1.0, size=n * n).reshape([n, n]).astype(dtype_)
    a += a.T
    a = np.tile(a, batch_shape + (1, 1))
    # Optimal stepsize for central difference is O(epsilon^{1/3}).
    epsilon = np.finfo(dtype_).eps
    delta = 0.1 * epsilon**(1.0 / 3.0)
    # tolerance obtained by looking at actual differences using
    # np.linalg.norm(theoretical-numerical, np.inf) on -mavx build
    if dtype_ == np.float32:
      tol = 1e-2
    else:
      tol = 1e-7
    with self.test_session():
      tf_a = tf.constant(a)
      tf_e, tf_v = tf.self_adjoint_eig(tf_a)
      for b in tf_e, tf_v:
        x_init = np.random.uniform(
            low=-1.0, high=1.0, size=n * n).reshape([n, n]).astype(dtype_)
        x_init += x_init.T
        x_init = np.tile(x_init, batch_shape + (1, 1))
        theoretical, numerical = tf.test.compute_gradient(
            tf_a,
            tf_a.get_shape().as_list(),
            b,
            b.get_shape().as_list(),
            x_init_value=x_init,
            delta=delta)
        self.assertAllClose(theoretical, numerical, atol=tol, rtol=tol)

  return Test


if __name__ == '__main__':
  for dtype in np.float32, np.float64:
    for size in 1, 2, 5, 10:
      for batch_dims in [(), (3,)] + [(3, 2)] * (max(size, size) < 10):
        shape = batch_dims + (size, size)
        name = '%s_%s' % (dtype.__name__, '_'.join(map(str, shape)))
        setattr(SelfAdjointEigTest, 'testSelfAdjointEig_' + name,
                _GetSelfAdjointEigTest(dtype, shape))
        setattr(SelfAdjointEigGradTest, 'testSelfAdjointEigGrad_' + name,
                _GetSelfAdjointEigGradTest(dtype, shape))
  tf.test.main()
