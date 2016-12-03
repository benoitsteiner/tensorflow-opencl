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
"""Base class for linear operators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from tensorflow.contrib import framework as contrib_framework
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops

__all__ = ["LinearOperator"]


class LinearOperator(object):
  """Base class defining a [batch of] linear operator[s].

  Subclasses of `LinearOperator` provide a access to common methods on a
  (batch) matrix, without the need to materialize the matrix.  This allows:

  * Matrix free computations
  * Operators that take advantage of special structure, while providing a
    consistent API to users.

  ### Subclassing

  To enable a public method, subclasses should implement the leading-underscore
  version of the method.  The argument signature should be identical except for
  the omission of `name="..."`.  For example, to enable
  `apply(x, adjoint=False, name="apply")` a subclass should implement
  `_apply(x, adjoint=False)`.

  ### Performance contract

  Subclasses should implement a method only if it can be done with a reasonable
  performance increase over generic dense operations, either in time, parallel
  scalability, or memory usage.  For example, if the determinant can only be
  computed using `tf.matrix_determinant(self.to_dense())`, then determinants
  should not be implemented.

  Class docstrings should contain an explanation of computational complexity.
  Since this is a high-performance library, attention should be paid to detail,
  and explanations can include constants as well as Big-O notation.

  ### Shape compatibility

  `LinearOperator` sub classes should operate on a [batch] matrix with
  compatible shape.  Class docstrings should define what is meant by compatible
  shape.  Some sub-classes may not support batching.

  An example is:

  `x` is a batch matrix with compatible shape for `apply` if

  ```
  operator.shape = [B1,...,Bb] + [M, N],  b >= 0,
  x.shape =   [B1,...,Bb] + [N, R]
  ```

  `rhs` is a batch matrix with compatible shape for `solve` if

  ```
  operator.shape = [B1,...,Bb] + [M, N],  b >= 0,
  rhs.shape =   [B1,...,Bb] + [M, R]
  ```

  ### Example docstring for subclasses.

  This operator acts like a (batch) matrix `A` with shape
  `[B1,...,Bb, M, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `m x n` matrix.  Again, this matrix `A` may not be materialized, but for
  purposes of identifying and working with compatible arguments the shape is
  relevant.

  Examples:

  ```python
  some_tensor = ... shape = ????
  operator = MyLinOp(some_tensor)

  operator.shape()
  ==> [2, 4, 4]

  operator.log_determinant()
  ==> Shape [2] Tensor

  x = ... Shape [2, 4, 5] Tensor

  operator.apply(x)
  ==> Shape [2, 4, 5] Tensor
  ```

  ### Shape compatibility

  This operator acts on batch matrices with compatible shape.
  FILL IN WHAT IS MEANT BY COMPATIBLE SHAPE

  ### Performance

  FILL THIS IN

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
               dtype,
               graph_parents=None,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               name=None):
    r"""Initialize the `LinearOperator`.

    **This is a private method for subclass use.**
    **Subclasses should copy-paste this `__init__` documentation.**

    Args:
      dtype: The type of the this `LinearOperator`.  Arguments to `apply` and
        `solve` will have to be this type.
      graph_parents: Python list of graph prerequisites of this `LinearOperator`
        Typically tensors that are passed during initialization.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  If `dtype` is real, this is equivalent to being symmetric.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the real part of all eigenvalues is positive.  We do not require
        the operator to be self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix\
            #Extension_for_non_symmetric_matrices
      name: A name for this `LinearOperator`.

    Raises:
      ValueError: if any member of graph_parents is `None` or not a `Tensor`.
    """
    # Check and auto-set flags.
    if is_positive_definite:
      if is_non_singular is False:
        raise ValueError("A positive definite matrix is always non-singular.")
      is_non_singular = True

    graph_parents = [] if graph_parents is None else graph_parents
    for i, t in enumerate(graph_parents):
      if t is None or not contrib_framework.is_tensor(t):
        raise ValueError("Graph parent item %d is not a Tensor; %s." % (i, t))
    self._dtype = dtype
    self._graph_parents = graph_parents
    self._is_non_singular = is_non_singular
    self._is_self_adjoint = is_self_adjoint
    self._is_positive_definite = is_positive_definite
    self._name = name or type(self).__name__

  @contextlib.contextmanager
  def _name_scope(self, name=None, values=None):
    """Helper function to standardize op scope."""
    with ops.name_scope(self.name):
      with ops.name_scope(
          name, values=((values or []) + self._graph_parents)) as scope:
        yield scope

  @property
  def dtype(self):
    """The `DType` of `Tensor`s handled by this `LinearOperator`."""
    return self._dtype

  @property
  def name(self):
    """Name prepended to all ops created by this `LinearOperator`."""
    return self._name

  @property
  def graph_parents(self):
    """List of graph dependencies of this `LinearOperator`."""
    return self._graph_parents

  @property
  def is_non_singular(self):
    return self._is_non_singular

  @property
  def is_self_adjoint(self):
    return self._is_self_adjoint

  @property
  def is_positive_definite(self):
    return self._is_positive_definite

  def _shape(self):
    # Write this in derived class to enable all static shape methods.
    raise NotImplementedError("_shape is not implemented.")

  @property
  def shape(self):
    """`TensorShape` of this `LinearOperator`.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns
    `TensorShape([B1,...,Bb, M, N])`, equivalent to `A.get_shape()`.

    Returns:
      `TensorShape`, statically determined, may be undefined.
    """
    return self._shape()

  def _shape_dynamic(self):
    raise NotImplementedError("_shape_dynamic is not implemented.")

  def shape_dynamic(self, name="shape_dynamic"):
    """Shape of this `LinearOperator`, determined at runtime.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
    `[B1,...,Bb, M, N]`, equivalent to `tf.shape(A)`.

    Args:
      name:  A name for this `Op.

    Returns:
      `int32` `Tensor`
    """
    with self._name_scope(name):
      return self._shape_dynamic()

  @property
  def batch_shape(self):
    """`TensorShape` of batch dimensions of this `LinearOperator`.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns
    `TensorShape([B1,...,Bb])`, equivalent to `A.get_shape()[:-2]`

    Returns:
      `TensorShape`, statically determined, may be undefined.
    """
    # Derived classes get this "for free" once .shape is implemented.
    return self.shape[:-2]

  def batch_shape_dynamic(self, name="batch_shape_dynamic"):
    """Shape of batch dimensions of this operator, determined at runtime.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
    `[B1,...,Bb]`.

    Args:
      name:  A name for this `Op.

    Returns:
      `int32` `Tensor`
    """
    # Derived classes get this "for free" once .shape() is implemented.
    with self._name_scope(name):
      return array_ops.slice(self.shape_dynamic(), [0],
                             [self.tensor_rank_dynamic() - 2])

  @property
  def tensor_rank(self, name="tensor_rank"):
    """Rank (in the sense of tensors) of matrix corresponding to this operator.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

    Args:
      name:  A name for this `Op.

    Returns:
      Python integer, or None if the tensor rank is undefined.
    """
    # Derived classes get this "for free" once .shape() is implemented.
    with self._name_scope(name):
      return self.shape.ndims

  def tensor_rank_dynamic(self, name="tensor_rank_dynamic"):
    """Rank (in the sense of tensors) of matrix corresponding to this operator.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

    Args:
      name:  A name for this `Op.

    Returns:
      `int32` `Tensor`, determined at runtime.
    """
    # Derived classes get this "for free" once .shape() is implemented.
    with self._name_scope(name):
      return array_ops.size(self.shape_dynamic())

  @property
  def domain_dimension(self):
    """Dimension (in the sense of vector spaces) of the domain of this operator.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

    Returns:
      Python integer if vector space dimension can be determined statically,
        otherwise `None`.
    """
    # Derived classes get this "for free" once .shape is implemented.
    return self.shape[-1]

  def domain_dimension_dynamic(self, name="domain_dimension_dynamic"):
    """Dimension (in the sense of vector spaces) of the domain of this operator.

    Determined at runtime.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

    Args:
      name:  A name for this `Op`.

    Returns:
      `int32` `Tensor`
    """
    # Derived classes get this "for free" once .shape() is implemented.
    with self._name_scope(name):
      return array_ops.gather(self.shape_dynamic(),
                              self.tensor_rank_dynamic() - 1)

  @property
  def range_dimension(self):
    """Dimension (in the sense of vector spaces) of the range of this operator.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

    Returns:
      Python integer if vector space dimension can be determined statically,
        otherwise `None`.
    """
    # Derived classes get this "for free" once .shape is implemented.
    return self.shape[-2]

  def range_dimension_dynamic(self, name="range_dimension_dynamic"):
    """Dimension (in the sense of vector spaces) of the range of this operator.

    Determined at runtime.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

    Args:
      name:  A name for this `Op`.

    Returns:
      `int32` `Tensor`
    """
    # Derived classes get this "for free" once .shape() is implemented.
    with self._name_scope(name):
      return array_ops.gather(self.shape_dynamic(),
                              self.tensor_rank_dynamic() - 2)

  def _assert_non_singular(self):
    raise NotImplementedError("assert_non_singular is not implemented.")

  def assert_non_singular(self, name="assert_non_singular"):
    """Returns an `Op` that asserts this operator is non singular."""
    with self._name_scope(name):
      return self._assert_non_singular()

  def _assert_positive_definite(self):
    raise NotImplementedError("assert_positive_definite is not implemented.")

  def assert_positive_definite(self, name="assert_positive_definite"):
    """Returns an `Op` that asserts this operator is positive definite.

    Here, positive definite means the real part of all eigenvalues is positive.
    We do not require the operator to be self-adjoint.

    Args:
      name:  A name to give this `Op`.

    Returns:
      An `Op` that asserts this operator is positive definite.
    """
    with self._name_scope(name):
      return self._assert_positive_definite()

  def _assert_self_adjoint(self):
    raise NotImplementedError("assert_self_adjoint is not implemented.")

  def assert_self_adjoint(self, name="assert_self_adjoint"):
    """Returns an `Op` that asserts this operator is self-adjoint."""
    with self._name_scope(name):
      return self._assert_self_adjoint()

  def _apply(self, x, adjoint=False):
    raise NotImplementedError("_apply is not implemented.")

  def apply(self, x, adjoint=False, name="apply"):
    """Transform `x` with left multiplication:  `x --> Ax`.

    Args:
      x: `Tensor` with compatible shape and same `dtype` as `self`.
        See class docstring for definition of compatibility.
      adjoint: Python `bool`.  If `True`, left multiply by the adjoint.
      name:  A name for this `Op.

    Returns:
      A `Tensor` with shape `[..., M, R]` and same `dtype` as `self`.
    """
    with self._name_scope(name, values=[x]):
      x = ops.convert_to_tensor(x, name="x")
      return self._apply(x, adjoint=adjoint)

  def _determinant(self):
    raise NotImplementedError("_det is not implemented.")

  def determinant(self, name="det"):
    """Determinant for every batch member.

    Args:
      name:  A name for this `Op.

    Returns:
      `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.
    """
    with self._name_scope(name):
      return self._determinant()

  def _log_abs_determinant(self):
    raise NotImplementedError("_log_abs_det is not implemented.")

  def log_abs_determinant(self, name="log_abs_det"):
    """Log absolute value of determinant for every batch member.

    Args:
      name:  A name for this `Op.

    Returns:
      `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.
    """
    with self._name_scope(name):
      return self._log_abs_determinant()

  def _solve(self, rhs, adjoint=False):
    # Since this is an exact solve method for all rhs, this will only be
    # available for non-singular (batch) operators, in particular the operator
    # must be square.
    raise NotImplementedError("_solve is not implemented.")

  def solve(self, rhs, adjoint=False, name="solve"):
    """Solve `R` (batch) systems of equations exactly: `A X = rhs`.

    Examples:

    ```python
    # Create an operator acting like a 10 x 2 x 2 matrix.
    operator = LinearOperator(...)
    operator.shape # = 10 x 2 x 2

    # Solve one linear system (R = 1) for every member of the length 10 batch.
    RHS = ... # shape 10 x 2 x 1
    X = operator.solve(RHS)  # shape 10 x 2 x 1

    # Solve five linear systems (R = 5) for every member of the length 10 batch.
    RHS = ... # shape 10 x 2 x 5
    X = operator.solve(RHS)
    X[3, :, 2]  # Solution to the linear system A[3, :, :] X = RHS[3, :, 2]
    ```

    Args:
      rhs: `Tensor` with same `dtype` as this operator and compatible shape.
        See class docstring for definition of compatibility.
      adjoint: Python `bool`.  If `True`, solve the system involving the adjoint
        of this `LinearOperator`.
      name:  A name scope to use for ops added by this method.

    Returns:
      `Tensor` with shape `[...,N, R]` and same `dtype` as `rhs`.

    Raises:
      ValueError:  If self.is_non_singular is False.
    """
    if self.is_non_singular is False:
      raise ValueError(
          "Exact solve cannot be called with an operator that is expected to "
          "be singular.")
    with self._name_scope(name, values=[rhs]):
      rhs = ops.convert_to_tensor(rhs, name="rhs")
      return self._solve(rhs, adjoint=adjoint)

  def _to_dense(self):
    """Generic and often inefficient implementation.  Override often."""
    if self.batch_shape.is_fully_defined():
      batch_shape = self.batch_shape
    else:
      batch_shape = self.batch_shape_dynamic()

    if self.domain_dimension.value is not None:
      n = self.domain_dimension.value
    else:
      n = self.domain_dimension_dynamic()

    eye = linalg_ops.eye(num_rows=n, batch_shape=batch_shape, dtype=self.dtype)
    return self.apply(eye)

  def to_dense(self, name="to_dense"):
    """Return a dense (batch) matrix representing this operator."""
    with self._name_scope(name):
      return self._to_dense()

  def _add_to_tensor(self, x):
    raise NotImplementedError("_add_to_tensor is not implemented.")

  def add_to_tensor(self, x, name="add_to_tensor"):
    """Add matrix represented by this operator to `x`.  Equivalent to `A + x`.

    Args:
      x:  `Tensor` with same `dtype` and shape broadcastable to `self.shape`.
      name:  A name to give this `Op`.

    Returns:
      A `Tensor` with broadcast shape and same `dtype` as `self`.
    """
    with self._name_scope(name, values=[x]):
      x = ops.convert_to_tensor(x, name="x")
      return self._add_to_tensor(x)
