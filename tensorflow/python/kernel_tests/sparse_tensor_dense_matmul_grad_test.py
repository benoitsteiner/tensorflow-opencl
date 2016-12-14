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

"""Tests for the gradient of `tf.sparse_tensor_dense_matmul()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class SparseTensorDenseMatMulGradientTest(tf.test.TestCase):

  def _sparsify(self, x):
    x[x < 0.5] = 0

    non_zero = np.where(x)
    x_indices = np.vstack(non_zero).astype(np.int64).T
    x_values = x[non_zero]
    x_shape = x.shape

    return tf.SparseTensor(
        indices=x_indices, values=x_values, dense_shape=x_shape), len(x_values)

  def _randomTensor(self, size, np_dtype, adjoint=False, sparse=False):
    n, m = size
    x = np.random.randn(n, m).astype(np_dtype)

    if adjoint:
      x = x.transpose()

    if sparse:
      return self._sparsify(x)
    else:
      return tf.constant(x, dtype=np_dtype)

  def _testGradients(self, adjoint_a, adjoint_b, name, np_dtype):
    n, k, m = np.random.randint(1, 10, size=3)
    sp_t, nnz = self._randomTensor(
        [n, k], np_dtype, adjoint=adjoint_a, sparse=True)
    dense_t = self._randomTensor([k, m], np_dtype, adjoint=adjoint_b)

    matmul = tf.sparse_tensor_dense_matmul(
        sp_t, dense_t, adjoint_a=adjoint_a, adjoint_b=adjoint_b, name=name)

    with self.test_session(use_gpu=True):
      dense_t_shape = [m, k] if adjoint_b else [k, m]
      sp_t_val_shape = [nnz]
      err = tf.test.compute_gradient_error([dense_t, sp_t.values],
                                           [dense_t_shape, sp_t_val_shape],
                                           matmul, [n, m])
      print("%s gradient err = %s" % (name, err))
      self.assertLess(err, 1e-3)

  def _testGradientsType(self, np_dtype):
    for adjoint_a in [True, False]:
      for adjoint_b in [True, False]:
        name = "sparse_tensor_dense_matmul_%s_%s_%s" % (adjoint_a, adjoint_b,
                                                        np_dtype.__name__)
        self._testGradients(adjoint_a, adjoint_b, name, np_dtype)

  def testGradients(self):
    np.random.seed(5)  # Fix seed to avoid flakiness
    self._testGradientsType(np.float32)
    self._testGradientsType(np.float64)


if __name__ == "__main__":
  tf.test.main()
