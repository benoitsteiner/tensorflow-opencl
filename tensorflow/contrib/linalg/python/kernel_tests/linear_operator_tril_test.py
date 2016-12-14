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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.linalg.python.ops import linear_operator_test_util


linalg = tf.contrib.linalg
tf.set_random_seed(23)


class LinearOperatorTriLTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  @property
  def _dtypes_to_test(self):
    # TODO(langmore) Test complex types once supported by
    # matrix_triangular_solve.
    return [tf.float32, tf.float64]

  def _operator_and_mat_and_feed_dict(self, shape, dtype, use_placeholder):
    shape = list(shape)
    diag_shape = shape[:-1]

    # Upper triangle will be ignored.
    # Use a diagonal that ensures this matrix is well conditioned.
    tril = tf.random_normal(shape=shape, dtype=dtype.real_dtype)
    diag = tf.random_uniform(
        shape=diag_shape, dtype=dtype.real_dtype, minval=2., maxval=3.)
    if dtype.is_complex:
      tril = tf.complex(
          tril, tf.random_normal(shape, dtype=dtype.real_dtype))
      diag = tf.complex(
          diag, tf.random_uniform(
              shape=diag_shape, dtype=dtype.real_dtype, minval=2., maxval=3.))

    tril = tf.matrix_set_diag(tril, diag)

    tril_ph = tf.placeholder(dtype=dtype)

    if use_placeholder:
      # Evaluate the tril here because (i) you cannot feed a tensor, and (ii)
      # tril is random and we want the same value used for both mat and
      # feed_dict.
      tril = tril.eval()
      operator = linalg.LinearOperatorTriL(tril_ph)
      feed_dict = {tril_ph: tril}
    else:
      operator = linalg.LinearOperatorTriL(tril)
      feed_dict = None

    mat = tf.matrix_band_part(tril, -1, 0)

    return operator, mat, feed_dict

  def test_assert_positive_definite(self):
    # Singlular matrix with one positive eigenvalue and one negative eigenvalue.
    with self.test_session():
      tril = [[1., 0.], [1., -1.]]
      operator = linalg.LinearOperatorTriL(tril)
      with self.assertRaisesOpError("was not positive definite"):
        operator.assert_positive_definite().run()

  def test_assert_non_singular(self):
    # Singlular matrix with one positive eigenvalue and one zero eigenvalue.
    with self.test_session():
      tril = [[1., 0.], [1., 0.]]
      operator = linalg.LinearOperatorTriL(tril)
      with self.assertRaisesOpError("Singular operator"):
        operator.assert_non_singular().run()

  def test_is_x_flags(self):
    # Matrix with one two positive eigenvalues.
    tril = [[1., 0.], [1., 1.]]
    operator = linalg.LinearOperatorTriL(
        tril,
        is_positive_definite=True,
        is_non_singular=True,
        is_self_adjoint=False)
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertFalse(operator.is_self_adjoint)


if __name__ == "__main__":
  tf.test.main()
