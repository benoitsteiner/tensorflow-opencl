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
"""Python layer for set_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework.python.framework import tensor_util

from tensorflow.contrib.util import loader
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.platform import resource_loader


_set_ops = loader.load_op_library(
    resource_loader.get_path_to_datafile("_set_ops.so"))

_VALID_DTYPES = set([
    dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64,
    dtypes.uint8, dtypes.uint16, dtypes.string])


ops.RegisterShape("SetSize")(common_shapes.call_cpp_shape_fn)


def set_size(a, validate_indices=True):
  """Compute number of unique elements along last dimension of `a`.

  Args:
    a: `SparseTensor`, with indices sorted in row-major order.
    validate_indices: Whether to validate the order and range of sparse indices
       in `a`.

  Returns:
    `int32` `Tensor` of set sizes. For `a` ranked `n`, this is a `Tensor` with
    rank `n-1`, and the same 1st `n-1` dimensions as `a`. Each value is the
    number of unique elements in the corresponding `[0...n-1]` dimension of `a`.

  Raises:
    TypeError: If `a` is an invalid types.
  """
  a = tensor_util.convert_to_tensor_or_sparse_tensor(a, name="a")
  if not isinstance(a, sparse_tensor.SparseTensor):
    raise TypeError("Expected `SparseTensor`, got %s." % a)
  if a.values.dtype.base_dtype not in _VALID_DTYPES:
    raise TypeError("Invalid dtype %s." % a.values.dtype)
  # pylint: disable=protected-access
  return _set_ops.set_size(a.indices, a.values, a.shape, validate_indices)

ops.NotDifferentiable("SetSize")


ops.RegisterShape("DenseToDenseSetOperation")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("DenseToSparseSetOperation")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("SparseToSparseSetOperation")(common_shapes.call_cpp_shape_fn)


ops.NotDifferentiable("DenseToDenseSetOperation")
ops.NotDifferentiable("DenseToSparseSetOperation")
ops.NotDifferentiable("SparseToSparseSetOperation")


def _set_operation(a, b, set_operation, validate_indices=True):
  """Compute set operation of elements in last dimension of `a` and `b`.

  All but the last dimension of `a` and `b` must match.

  Args:
    a: `Tensor` or `SparseTensor` of the same type as `b`. If sparse, indices
        must be sorted in row-major order.
    b: `Tensor` or `SparseTensor` of the same type as `a`. Must be
        `SparseTensor` if `a` is `SparseTensor`. If sparse, indices must be
        sorted in row-major order.
    set_operation: String indicating set operaiton. See
        SetOperationOp::SetOperationFromContext for valid values.
    validate_indices: Whether to validate the order and range of sparse indices
       in `a` and `b`.

  Returns:
    A `SparseTensor` with the same rank as `a` and `b`, and all but the last
    dimension the same. Elements along the last dimension contain the results
    of the set operation.

  Raises:
    TypeError: If inputs are invalid types.
    ValueError: If `a` is sparse and `b` is dense.
  """
  a = tensor_util.convert_to_tensor_or_sparse_tensor(a, name="a")
  if a.dtype.base_dtype not in _VALID_DTYPES:
    raise TypeError("'a' invalid dtype %s." % a.dtype)
  b = tensor_util.convert_to_tensor_or_sparse_tensor(b, name="b")
  if b.dtype.base_dtype != a.dtype.base_dtype:
    raise TypeError("Types don't match, %s vs %s." % (a.dtype, b.dtype))
  # pylint: disable=protected-access
  if isinstance(a, sparse_tensor.SparseTensor):
    if isinstance(b, sparse_tensor.SparseTensor):
      indices, values, shape = _set_ops.sparse_to_sparse_set_operation(
          a.indices, a.values, a.shape, b.indices, b.values, b.shape,
          set_operation, validate_indices)
    else:
      raise ValueError("Sparse,Dense is not supported, but Dense,Sparse is. "
                       "Please flip the order of your inputs.")
  elif isinstance(b, sparse_tensor.SparseTensor):
    indices, values, shape = _set_ops.dense_to_sparse_set_operation(
        a, b.indices, b.values, b.shape, set_operation, validate_indices)
  else:
    indices, values, shape = _set_ops.dense_to_dense_set_operation(
        a, b, set_operation, validate_indices)
  # pylint: enable=protected-access
  return sparse_tensor.SparseTensor(indices, values, shape)


def set_intersection(a, b, validate_indices=True):
  """Compute set intersection of elements in last dimension of `a` and `b`.

  All but the last dimension of `a` and `b` must match.

  Args:
    a: `Tensor` or `SparseTensor` of the same type as `b`. If sparse, indices
        must be sorted in row-major order.
    b: `Tensor` or `SparseTensor` of the same type as `a`. Must be
        `SparseTensor` if `a` is `SparseTensor`. If sparse, indices must be
        sorted in row-major order.
    validate_indices: Whether to validate the order and range of sparse indices
       in `a` and `b`.

  Returns:
    A `SparseTensor` with the same rank as `a` and `b`, and all but the last
    dimension the same. Elements along the last dimension contain the
    intersections.
  """
  return _set_operation(a, b, "intersection", validate_indices)


def set_difference(a, b, aminusb=True, validate_indices=True):
  """Compute set difference of elements in last dimension of `a` and `b`.

  All but the last dimension of `a` and `b` must match.

  Args:
    a: `Tensor` or `SparseTensor` of the same type as `b`. If sparse, indices
        must be sorted in row-major order.
    b: `Tensor` or `SparseTensor` of the same type as `a`. Must be
        `SparseTensor` if `a` is `SparseTensor`. If sparse, indices must be
        sorted in row-major order.
    aminusb: Whether to subtract `b` from `a`, vs vice versa.
    validate_indices: Whether to validate the order and range of sparse indices
       in `a` and `b`.

  Returns:
    A `SparseTensor` with the same rank as `a` and `b`, and all but the last
    dimension the same. Elements along the last dimension contain the
    differences.
  """
  return _set_operation(a, b, "a-b" if aminusb else "b-a", validate_indices)


def set_union(a, b, validate_indices=True):
  """Compute set union of elements in last dimension of `a` and `b`.

  All but the last dimension of `a` and `b` must match.

  Args:
    a: `Tensor` or `SparseTensor` of the same type as `b`. If sparse, indices
        must be sorted in row-major order.
    b: `Tensor` or `SparseTensor` of the same type as `a`. Must be
        `SparseTensor` if `a` is `SparseTensor`. If sparse, indices must be
        sorted in row-major order.
    validate_indices: Whether to validate the order and range of sparse indices
       in `a` and `b`.

  Returns:
    A `SparseTensor` with the same rank as `a` and `b`, and all but the last
    dimension the same. Elements along the last dimension contain the
    unions.
  """
  return _set_operation(a, b, "union", validate_indices)
