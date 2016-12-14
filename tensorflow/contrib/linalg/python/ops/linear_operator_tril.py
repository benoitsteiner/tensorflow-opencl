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
"""`LinearOperator` acting like a lower triangular matrix."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.linalg.python.ops import linear_operator
from tensorflow.contrib.linalg.python.ops import linear_operator_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops

__all__ = ["LinearOperatorTriL",]


class LinearOperatorTriL(linear_operator.LinearOperator):
  """`LinearOperator` acting like a [batch] square lower triangular matrix.

  This operator acts like a [batch] matrix `A` with shape
  `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x N` matrix.

  `LinearOperatorTriL` is initialized with a `Tensor` having dimensions
  `[B1,...,Bb, N, N]`. The upper triangle of the last two dimensions is ignored.

  ```python
  # Create a 2 x 2 lower-triangular linear operator.
  tril = [[1., 2.], [3., 4.]]
  operator = LinearOperatorTriL(tril)

  # The upper triangle is ignored.
  operator.to_dense()
  ==> [[1., 0.]
       [3., 4.]]

  operator.shape
  ==> [2, 2]

  operator.log_determinant()
  ==> scalar Tensor

  x = ... Shape [2, 4] Tensor
  operator.apply(x)
  ==> Shape [2, 4] Tensor

  # Create a [2, 3] batch of 4 x 4 linear operators.
  tril = tf.random_normal(shape=[2, 3, 4, 4])
  operator = LinearOperatorTriL(tril)

  # Create a shape [2, 1, 4, 2] vector.  Note that this shape is compatible
  # since the batch dimensions, [2, 1], are brodcast to
  # operator.batch_shape = [2, 3].
  y = tf.random_normal(shape=[2, 1, 4, 2])
  x = operator.solve(y)
  ==> operator.apply(x) = y
  ```

  ### Shape compatibility

  This operator acts on [batch] matrix with compatible shape.
  `x` is a batch matrix with compatible shape for `apply` and `solve` if

  ```
  operator.shape = [B1,...,Bb] + [N, N],  with b >= 0
  x.shape =        [B1,...,Bb] + [N, R],  with R >= 0.
  ```

  ### Performance

  Suppose `operator` is a `LinearOperatorTriL` of shape `[N, N]`,
  and `x.shape = [N, R]`.  Then

  * `operator.apply(x)` involves `N^2 * R` multiplications.
  * `operator.solve(x)` involves `N * R` size `N` back-substitutions.
  * `operator.determinant()` involves a size `N` `reduce_prod`.

  If instead `operator` and `x` have shape `[B1,...,Bb, N, N]` and
  `[B1,...,Bb, N, R]`, every operation increases in complexity by `B1*...*Bb`.

  ### Matrix property hints

  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint` etc...
  These have the following meaning
  * If `is_X == True`, callers should expect the operator to have the
    property `X`.  This is a promise that should be fulfilled, but is *not* a
    runtime assert.  For example, finite floating point precision may result
    in these promises being violated.
  * If `is_X == False`, callers should expect the operator to not have `X`.
  * If `is_X == None` (the default), callers should have no expectation either
    way.
  """

  def __init__(self,
               tril,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               name="LinearOperatorTriL"):
    """Initialize a `LinearOperatorTriL`.

    Args:
      tril:  Shape `[B1,...,Bb, N, N]` with `b >= 0`, `N >= 0`.
        The lower triangular part of `tril` defines this operator.  The strictly
        upper triangle is ignored.  Allowed dtypes: `float32`, `float64`.
      is_non_singular:  Expect that this operator is non-singular.
        This operator is non-singular if and only if its diagonal elements are
        all non-zero.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  This operator is self-adjoint only if it is diagonal with
        real-valued diagonal entries.  In this case it is advised to use
        `LinearOperatorDiag`.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the real part of all eigenvalues is positive.  We do not require
        the operator to be self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix
            #Extension_for_non_symmetric_matrices
      name: A name for this `LinearOperator`.

    Raises:
      TypeError:  If `diag.dtype` is not an allowed type.
    """

    # TODO(langmore) Add complex types once matrix_triangular_solve works for
    # them.
    allowed_dtypes = [dtypes.float32, dtypes.float64]

    with ops.name_scope(name, values=[tril]):
      self._tril = array_ops.matrix_band_part(tril, -1, 0)
      self._diag = array_ops.matrix_diag_part(self._tril)

      dtype = self._tril.dtype
      if dtype not in allowed_dtypes:
        raise TypeError(
            "Argument diag must have dtype in %s.  Found: %s"
            % (allowed_dtypes, dtype))

      super(LinearOperatorTriL, self).__init__(
          dtype=self._tril.dtype,
          graph_parents=[self._tril],
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          name=name)

  def _shape(self):
    return self._tril.get_shape()

  def _shape_dynamic(self):
    return array_ops.shape(self._tril)

  def _assert_non_singular(self):
    return linear_operator_util.assert_no_entries_with_modulus_zero(
        self._diag,
        message="Singular operator:  Diagonal contained zero values.")

  def _assert_positive_definite(self):
    if self.dtype.is_complex:
      message = (
          "Diagonal operator had diagonal entries with non-positive real part, "
          "thus was not positive definite.")
    else:
      message = (
          "Real diagonal operator had non-positive diagonal entries, "
          "thus was not positive definite.")

    return check_ops.assert_positive(
        math_ops.real(self._diag),
        message=message)

  def _apply(self, x, adjoint=False):
    return math_ops.matmul(self._tril, x, adjoint_a=adjoint)

  def _determinant(self):
    return math_ops.reduce_prod(self._diag, reduction_indices=[-1])

  def _log_abs_determinant(self):
    return math_ops.reduce_sum(
        math_ops.log(math_ops.abs(self._diag)), reduction_indices=[-1])

  def _solve(self, rhs, adjoint=False):
    return linalg_ops.matrix_triangular_solve(
        self._tril, rhs, lower=True, adjoint=adjoint)

  def _to_dense(self):
    return self._tril

  def _add_to_tensor(self, x):
    return self._tril + x
