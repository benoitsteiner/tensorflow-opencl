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

import numpy as np
import tensorflow as tf

linalg = tf.contrib.linalg
rng = np.random.RandomState(123)


class LinearOperatorShape(linalg.LinearOperator):
  """LinearOperator that implements the methods ._shape and _shape_dynamic."""

  def __init__(self,
               shape,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None):
    self._stored_shape = shape
    super(LinearOperatorShape, self).__init__(
        dtype=tf.float32,
        graph_parents=None,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,)

  def _shape(self):
    return tf.TensorShape(self._stored_shape)

  def _shape_dynamic(self):
    return tf.constant(self._stored_shape, dtype=tf.int32)


class LinearOperatorApplyOnly(linalg.LinearOperator):
  """LinearOperator that simply wraps a [batch] matrix and implements apply."""

  def __init__(self,
               matrix,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None):
    self._matrix = tf.convert_to_tensor(matrix, name="matrix")
    super(LinearOperatorApplyOnly, self).__init__(
        dtype=matrix.dtype,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,)

  def _shape(self):
    return self._matrix.get_shape()

  def _shape_dynamic(self):
    return tf.shape(self._matrix)

  def _apply(self, x, adjoint=False):
    return tf.matmul(self._matrix, x, adjoint_a=adjoint)


class LinearOperatorTest(tf.test.TestCase):

  def test_all_shape_properties_defined_by_the_one_property_shape(self):

    shape = (1, 2, 3, 4)
    operator = LinearOperatorShape(shape)

    self.assertAllEqual(shape, operator.shape)
    self.assertAllEqual(4, operator.tensor_rank)
    self.assertAllEqual((1, 2), operator.batch_shape)
    self.assertAllEqual(4, operator.domain_dimension)
    self.assertAllEqual(3, operator.range_dimension)

  def test_all_shape_methods_defined_by_the_one_method_shape(self):
    with self.test_session():
      shape = (1, 2, 3, 4)
      operator = LinearOperatorShape(shape)

      self.assertAllEqual(shape, operator.shape_dynamic().eval())
      self.assertAllEqual(4, operator.tensor_rank_dynamic().eval())
      self.assertAllEqual((1, 2), operator.batch_shape_dynamic().eval())
      self.assertAllEqual(4, operator.domain_dimension_dynamic().eval())
      self.assertAllEqual(3, operator.range_dimension_dynamic().eval())

  def test_is_x_properties(self):
    operator = LinearOperatorShape(
        shape=(2, 2),
        is_non_singular=False,
        is_self_adjoint=True,
        is_positive_definite=False)
    self.assertFalse(operator.is_non_singular)
    self.assertTrue(operator.is_self_adjoint)
    self.assertFalse(operator.is_positive_definite)

  def test_generic_to_dense_method_non_square_matrix_static(self):
    matrix = rng.randn(2, 3, 4)
    operator = LinearOperatorApplyOnly(matrix)
    with self.test_session():
      operator_dense = operator.to_dense()
      self.assertAllEqual((2, 3, 4), operator_dense.get_shape())
      self.assertAllClose(matrix, operator_dense.eval())

  def test_generic_to_dense_method_non_square_matrix_dynamic(self):
    matrix = rng.randn(2, 3, 4)
    matrix_ph = tf.placeholder(tf.float64)
    operator = LinearOperatorApplyOnly(matrix_ph)
    with self.test_session():
      operator_dense = operator.to_dense()
      self.assertAllClose(
          matrix, operator_dense.eval(feed_dict={matrix_ph: matrix}))


if __name__ == "__main__":
  tf.test.main()
