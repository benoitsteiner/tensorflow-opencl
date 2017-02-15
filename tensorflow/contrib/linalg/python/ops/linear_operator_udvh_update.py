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
"""Perturb a `LinearOperator` with a rank `K` update."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework.python.framework import tensor_util as contrib_tensor_util
from tensorflow.contrib.linalg.python.ops import linear_operator
from tensorflow.contrib.linalg.python.ops import linear_operator_diag
from tensorflow.contrib.linalg.python.ops import linear_operator_identity
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops

__all__ = ["LinearOperatorUDVHUpdate",]


class LinearOperatorUDVHUpdate(linear_operator.LinearOperator):
  """Perturb a `LinearOperator` with a rank `K` update.

  This operator acts like a [batch] matrix `A` with shape
  `[B1,...,Bb, M, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `M x N` matrix.

  `LinearOperatorUDVHUpdate` represents `A = L + U D V^H`, where

  ```
  L, is a LinearOperator representing [batch] M x N matrices
  U, is a [batch] M x K matrix.  Typically K << M.
  D, is a [batch] K x K matrix.
  V, is a [batch] N x K matrix.  Typically K << N.
  V^H is the Hermitian transpose (adjoint) of V.
  ```

  If `M = N`, determinants and solves are done using the matrix determinant
  lemma and Woodbury identities, and thus require L and D to be non-singular.

  Solves and determinants will be attempted unless the "is_non_singular"
  property of L and D is False.

  In the event that L and D are positive-definite, and U = V, solves and
  determinants can be done using a Cholesky factorization.

  ```python
  # Create a 3 x 3 diagonal linear operator.
  diag_operator = LinearOperatorDiag(
      diag=[1., 2., 3.], is_non_singular=True, is_self_adjoint=True,
      is_positive_definite=True)

  # Perturb with a rank 2 perturbation
  operator = LinearOperatorUDVHUpdate(
      operator=diag_operator,
      u=[[1., 2.], [-1., 3.], [0., 0.]],
      diag=[11., 12.],
      v=[[1., 2.], [-1., 3.], [10., 10.]])

  operator.shape
  ==> [3, 3]

  operator.log_determinant()
  ==> scalar Tensor

  x = ... Shape [3, 4] Tensor
  operator.apply(x)
  ==> Shape [3, 4] Tensor
  ```

  ### Shape compatibility

  This operator acts on [batch] matrix with compatible shape.
  `x` is a batch matrix with compatible shape for `apply` and `solve` if

  ```
  operator.shape = [B1,...,Bb] + [M, N],  with b >= 0
  x.shape =        [B1,...,Bb] + [N, R],  with R >= 0.
  ```

  ### Performance

  Suppose `operator` is a `LinearOperatorUDVHUpdate` of shape `[M, N]`,
  made from a rank `K` update of `base_operator` which performs `.apply(x)` on
  `x` having `x.shape = [N, R]` with `O(L_apply*N*R)` complexity (and similarly
  for `solve`, `determinant`.  Then, if `x.shape = [N, R]`,

  * `operator.apply(x)` is `O(L_apply*N*R + K*N*R)`

  and if `M = N`,

  * `operator.solve(x)` is `O(L_apply*N*R + N*K*R + K^2*R + K^3)`
  * `operator.determinant()` is `O(L_determinant + L_solve*N*K + K^2*N + K^3)`

  If instead `operator` and `x` have shape `[B1,...,Bb, M, N]` and
  `[B1,...,Bb, N, R]`, every operation increases in complexity by `B1*...*Bb`.

  #### Matrix property hints

  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite, diag_positive, square`
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
               base_operator,
               u,
               diag=None,
               v=None,
               is_diag_positive=None,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name="LinearOperatorUDVHUpdate"):
    """Initialize a `LinearOperatorUDVHUpdate`.

    This creates a `LinearOperator` of the form `A = L + U D V^H`, with
    `L` a `LinearOperator`, `U, V` both [batch] matrices, and `D` a [batch]
    diagonal matrix.

    If `L` is non-singular, solves and determinants are available.
    Solves/determinants both involve a solve/determinant of a `K x K` system.
    In the event that L and D are self-adjoint positive-definite, and U = V,
    this can be done using a Cholesky factorization.  The user should set the
    `is_X` matrix property hints, which will trigger the appropriate code path.

    Args:
      base_operator:  Shape `[B1,...,Bb, M, N]` real `float32` or `float64`
        `LinearOperator`.  This is `L` above.
      u:  Shape `[B1,...,Bb, M, K]` `Tensor` of same `dtype` as `base_operator`.
        This is `U` above.
      diag:  Optional shape `[B1,...,Bb, K]` `Tensor` with same `dtype` as
        `base_operator`.  This is the diagonal of `D` above.
         Defaults to `D` being the identity operator.
      v:  Optional `Tensor` of same `dtype` as `u` and shape `[B1,...,Bb, N, K]`
         Defaults to `v = u`, in which case the perturbation is symmetric.
         If `M != N`, then `v` must be set since the pertrubation is not square.
      is_diag_positive:  Python `bool`.  If `True`, expect `diag > 0`.
      is_non_singular:  Expect that this operator is non-singular.
        Default is `None`, unless `is_positive_definite` is auto-set to be
        `True` (see below).
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  Default is `None`, unless `base_operator` is self-adjoint
        and `v = None` (meaning `u=v`), in which case this defaults to `True`.
      is_positive_definite:  Expect that this operator is positive definite.
        Default is `None`, unless `base_operator` is positive-definite
        `v = None` (meaning `u=v`), and `is_diag_positive`, in which case this
        defaults to `True`.
      is_square:  Expect that this operator acts like square [batch] matrices.
      name: A name for this `LinearOperator`.

    Raises:
      ValueError:  If `is_X` flags are set in an inconsistent way.
    """
    # TODO(langmore) support complex types.
    # Complex types are not allowed due to tf.cholesky() requiring float.
    # If complex dtypes are allowed, we update the following
    # 1. is_diag_positive should still imply that `diag > 0`, but we need to
    #    remind the user that this implies diag is real.  This is needed because
    #    if diag has non-zero imaginary part, it will not be self-adjoint
    #    positive definite.
    dtype = base_operator.dtype
    allowed_dtypes = [dtypes.float32, dtypes.float64]
    if dtype not in allowed_dtypes:
      raise TypeError(
          "Argument matrix must have dtype in %s.  Found: %s"
          % (allowed_dtypes, dtype))

    if diag is None:
      if is_diag_positive is False:
        raise ValueError(
            "Default diagonal is the identity, which is positive.  However, "
            "user set 'is_diag_positive' to False.")
      is_diag_positive = True

    # In this case, we can use a Cholesky decomposition to help us solve/det.
    self._use_cholesky = (
        base_operator.is_positive_definite and base_operator.is_self_adjoint
        and is_diag_positive
        and v is None)

    # Possibly auto-set some characteristic flags from None to True.
    # If the Flags were set (by the user) incorrectly to False, then raise.
    if base_operator.is_self_adjoint and v is None and not dtype.is_complex:
      if is_self_adjoint is False:
        raise ValueError(
            "A = L + UDU^H, with L self-adjoint and D real diagonal.  Since"
            " UDU^H is self-adjoint, this must be a self-adjoint operator.")
      is_self_adjoint = True

    # The condition for using a cholesky is sufficient for SPD, and
    # we no weaker choice of these hints leads to SPD.  Therefore,
    # the following line reads "if hints indicate SPD..."
    if self._use_cholesky:
      if (
          is_positive_definite is False
          or is_self_adjoint is False
          or is_non_singular is False):
        raise ValueError(
            "Arguments imply this is self-adjoint positive-definite operator.")
      is_positive_definite = True
      is_self_adjoint = True

    values = base_operator.graph_parents + [u, diag, v]
    with ops.name_scope(name, values=values):

      # Create U and V.
      self._u = ops.convert_to_tensor(u, name="u")
      if v is None:
        self._v = self._u
      else:
        self._v = ops.convert_to_tensor(v, name="v")

      if diag is None:
        self._diag = None
      else:
        self._diag = ops.convert_to_tensor(diag, name="diag")

      # Create base_operator L.
      self._base_operator = base_operator
      graph_parents = base_operator.graph_parents + [self.u, self._diag, self.v]
      graph_parents = [p for p in graph_parents if p is not None]

      super(LinearOperatorUDVHUpdate, self).__init__(
          dtype=self._base_operator.dtype,
          graph_parents=graph_parents,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          name=name)

      # Create the diagonal operator D.
      self._set_diag_operators(diag, is_diag_positive)
      self._is_diag_positive = is_diag_positive

      contrib_tensor_util.assert_same_float_dtype(
          (base_operator, self.u, self.v, self._diag))
      self._check_shapes()

      # Pre-compute the so-called "capacitance" matrix
      #   C := D^{-1} + V^H L^{-1} U
      self._capacitance = self._make_capacitance()
      if self._use_cholesky:
        self._chol_capacitance = linalg_ops.cholesky(self._capacitance)

  def _check_shapes(self):
    """Static check that shapes are compatible."""
    # Broadcast shape also checks that u and v are compatible.
    uv_shape = array_ops.broadcast_static_shape(
        self.u.get_shape(), self.v.get_shape())

    batch_shape = array_ops.broadcast_static_shape(
        self.base_operator.batch_shape, uv_shape[:-2])

    self.base_operator.domain_dimension.assert_is_compatible_with(
        uv_shape[-2])

    if self._diag is not None:
      uv_shape[-1].assert_is_compatible_with(self._diag.get_shape()[-1])
      array_ops.broadcast_static_shape(
          batch_shape, self._diag.get_shape()[:-1])

  def _set_diag_operators(self, diag, is_diag_positive):
    """Set attributes self._diag and self._diag_operator."""
    if diag is not None:
      self._diag_operator = linear_operator_diag.LinearOperatorDiag(
          self._diag, is_positive_definite=is_diag_positive)
      self._diag_inv_operator = linear_operator_diag.LinearOperatorDiag(
          1. / self._diag, is_positive_definite=is_diag_positive)
    else:
      if self.u.get_shape()[-1].value is not None:
        r = self.u.get_shape()[-1].value
      else:
        r = array_ops.shape(self.u)[-1]
      self._diag_operator = linear_operator_identity.LinearOperatorIdentity(
          num_rows=r, dtype=self.dtype)
      self._diag_inv_operator = self._diag_operator

  @property
  def u(self):
    """If this operator is `A = L + U D V^H`, this is the `U`."""
    return self._u

  @property
  def v(self):
    """If this operator is `A = L + U D V^H`, this is the `V`."""
    return self._v

  @property
  def is_diag_positive(self):
    """If this operator is `A = L + U D V^H`, this hints `D > 0` elementwise."""
    return self._is_diag_positive

  @property
  def diag_arg(self):
    """If this operator is `A = L + U D V^H`, this is the diagonal of `D`."""
    return self._diag

  @property
  def diag_operator(self):
    """If this operator is `A = L + U D V^H`, this is `D`."""
    return self._diag_operator

  @property
  def base_operator(self):
    """If this operator is `A = L + U D V^H`, this is the `L`."""
    return self._base_operator

  def _shape(self):
    batch_shape = array_ops.broadcast_static_shape(
        self.base_operator.batch_shape,
        self.u.get_shape()[:-2])
    return batch_shape.concatenate(self.base_operator.shape[-2:])

  def _shape_tensor(self):
    batch_shape = array_ops.broadcast_dynamic_shape(
        self.base_operator.batch_shape_tensor(),
        array_ops.shape(self.u)[:-2])
    return array_ops.concat(
        [batch_shape, self.base_operator.shape_tensor()[-2:]], axis=0)

  def _apply(self, x, adjoint=False):
    u = self.u
    v = self.v
    l = self.base_operator
    d = self.diag_operator

    leading_term = l.apply(x, adjoint=adjoint)

    if adjoint:
      uh_x = math_ops.matmul(u, x, adjoint_a=True)
      d_uh_x = d.apply(uh_x, adjoint=adjoint)
      v_d_uh_x = math_ops.matmul(v, d_uh_x)
      return leading_term + v_d_uh_x
    else:
      vh_x = math_ops.matmul(v, x, adjoint_a=True)
      d_vh_x = d.apply(vh_x, adjoint=adjoint)
      u_d_vh_x = math_ops.matmul(u, d_vh_x)
      return leading_term + u_d_vh_x

  def _determinant(self):
    if self.is_positive_definite:
      return math_ops.exp(self.log_abs_determinant())
    # The matrix determinant lemma gives
    # https://en.wikipedia.org/wiki/Matrix_determinant_lemma
    #   det(L + UDV^H) = det(D^{-1} + V^H L^{-1} U) det(D) det(L)
    #                  = det(C) det(D) det(L)
    # where C is sometimes known as the capacitance matrix,
    #   C := D^{-1} + V^H L^{-1} U
    det_c = linalg_ops.matrix_determinant(self._capacitance)
    det_d = self.diag_operator.determinant()
    det_l = self.base_operator.determinant()
    return det_c * det_d * det_l

  def _log_abs_determinant(self):
    # Recall
    #   det(L + UDV^H) = det(D^{-1} + V^H L^{-1} U) det(D) det(L)
    #                  = det(C) det(D) det(L)
    log_abs_det_d = self.diag_operator.log_abs_determinant()
    log_abs_det_l = self.base_operator.log_abs_determinant()

    if self._use_cholesky:
      chol_cap_diag = array_ops.matrix_diag_part(self._chol_capacitance)
      log_abs_det_c = 2 * math_ops.reduce_sum(
          math_ops.log(chol_cap_diag), reduction_indices=[-1])
    else:
      det_c = linalg_ops.matrix_determinant(self._capacitance)
      log_abs_det_c = math_ops.log(math_ops.abs(det_c))

    return log_abs_det_c + log_abs_det_d + log_abs_det_l

  def _solve(self, rhs, adjoint=False):
    if self.base_operator.is_non_singular is False:
      raise ValueError(
          "Solve not implemented unless this is a perturbation of a "
          "non-singular LinearOperator.")
    # The Woodbury formula gives:
    # https://en.wikipedia.org/wiki/Woodbury_matrix_identity
    #   (L + UDV^H)^{-1}
    #   = L^{-1} - L^{-1} U (D^{-1} + V^H L^{-1} U)^{-1} V^H L^{-1}
    #   = L^{-1} - L^{-1} U C^{-1} V^H L^{-1}
    # where C is the capacitance matrix, C := D^{-1} + V^H L^{-1} U
    # Note also that, with ^{-H} being the inverse of the adjoint,
    #   (L + UDV^H)^{-H}
    #   = L^{-H} - L^{-H} V C^{-H} U^H L^{-H}
    l = self.base_operator
    if adjoint:
      v = self.u
      u = self.v
    else:
      v = self.v
      u = self.u

    # L^{-1} rhs
    linv_rhs = l.solve(rhs, adjoint=adjoint)
    # V^H L^{-1} rhs
    vh_linv_rhs = math_ops.matmul(v, linv_rhs, adjoint_a=True)
    # C^{-1} V^H L^{-1} rhs
    if self._use_cholesky:
      capinv_vh_linv_rhs = linalg_ops.cholesky_solve(
          self._chol_capacitance, vh_linv_rhs)
    else:
      capinv_vh_linv_rhs = linalg_ops.matrix_solve(
          self._capacitance, vh_linv_rhs, adjoint=adjoint)
    # U C^{-1} V^H M^{-1} rhs
    u_capinv_vh_linv_rhs = math_ops.matmul(u, capinv_vh_linv_rhs)
    # L^{-1} U C^{-1} V^H L^{-1} rhs
    linv_u_capinv_vh_linv_rhs = l.solve(u_capinv_vh_linv_rhs, adjoint=adjoint)

    # L^{-1} - L^{-1} U C^{-1} V^H L^{-1}
    return linv_rhs - linv_u_capinv_vh_linv_rhs

  def _make_capacitance(self):
    # C := D^{-1} + V^H L^{-1} U
    # which is sometimes known as the "capacitance" matrix.

    # L^{-1} U
    linv_u = self.base_operator.solve(self.u)
    # V^H L^{-1} U
    vh_linv_u = math_ops.matmul(self.v, linv_u, adjoint_a=True)

    # D^{-1} + V^H L^{-1} V
    capacitance = self._diag_inv_operator.add_to_tensor(vh_linv_u)
    return capacitance
