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

"""Operations that generate constants.  See the ${@python/constant} guide.

@@zeros
@@zeros_like
@@ones
@@ones_like
@@fill
@@constant
@@linspace
@@range
@@random_normal
@@truncated_normal
@@random_uniform
@@random_shuffle
@@random_crop
@@multinomial
@@random_gamma
@@random_poisson
@@set_random_seed
"""

# Must be separate from array_ops to avoid a cyclic dependency.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util


def constant(value, dtype=None, shape=None, name="Const", verify_shape=False):
  """Creates a constant tensor.

   The resulting tensor is populated with values of type `dtype`, as
   specified by arguments `value` and (optionally) `shape` (see examples
   below).

   The argument `value` can be a constant value, or a list of values of type
   `dtype`. If `value` is a list, then the length of the list must be less
   than or equal to the number of elements implied by the `shape` argument (if
   specified). In the case where the list length is less than the number of
   elements specified by `shape`, the last element in the list will be used
   to fill the remaining entries.

   The argument `shape` is optional. If present, it specifies the dimensions of
   the resulting tensor. If not present, the shape of `value` is used.

   If the argument `dtype` is not specified, then the type is inferred from
   the type of `value`.

   For example:

   ```python
   # Constant 1-D Tensor populated with value list.
   tensor = tf.constant([1, 2, 3, 4, 5, 6, 7]) => [1 2 3 4 5 6 7]

   # Constant 2-D tensor populated with scalar value -1.
   tensor = tf.constant(-1.0, shape=[2, 3]) => [[-1. -1. -1.]
                                                [-1. -1. -1.]]
   ```

  Args:
    value:          A constant value (or list) of output type `dtype`.

    dtype:          The type of the elements of the resulting tensor.

    shape:          Optional dimensions of resulting tensor.

    name:           Optional name for the tensor.

    verify_shape:   Boolean that enables verification of a shape of values.

  Returns:
    A Constant Tensor.
  """
  g = ops.get_default_graph()
  tensor_value = attr_value_pb2.AttrValue()
  tensor_value.tensor.CopyFrom(
      tensor_util.make_tensor_proto(value, dtype=dtype, shape=shape, verify_shape=verify_shape))
  dtype_value = attr_value_pb2.AttrValue(type=tensor_value.tensor.dtype)
  const_tensor = g.create_op(
      "Const", [], [dtype_value.type],
      attrs={"value": tensor_value, "dtype": dtype_value}, name=name).outputs[0]
  return const_tensor


def _constant_tensor_conversion_function(v, dtype=None, name=None,
                                         as_ref=False):
  _ = as_ref
  return constant(v, dtype=dtype, name=name)


ops.register_tensor_conversion_function(
    (list, tuple), _constant_tensor_conversion_function, 100)
ops.register_tensor_conversion_function(
    np.ndarray, _constant_tensor_conversion_function, 100)
ops.register_tensor_conversion_function(
    np.generic, _constant_tensor_conversion_function, 100)
ops.register_tensor_conversion_function(
    object, _constant_tensor_conversion_function, 200)


def _tensor_shape_tensor_conversion_function(s, dtype=None, name=None,
                                             as_ref=False):
  _ = as_ref
  if not s.is_fully_defined():
    raise ValueError(
        "Cannot convert a partially known TensorShape to a Tensor: %s" % s)
  if dtype is not None:
    if dtype not in (dtypes.int32, dtypes.int64):
      raise TypeError("Cannot convert a TensorShape to dtype: %s" % dtype)
  else:
    dtype = dtypes.int32
  if name is None:
    name = "shape_as_tensor"
  return constant(s.as_list(), dtype=dtype, name=name)

ops.register_tensor_conversion_function(
    tensor_shape.TensorShape, _tensor_shape_tensor_conversion_function, 100)


def _dimension_tensor_conversion_function(d, dtype=None, name=None,
                                          as_ref=False):
  _ = as_ref
  if d.value is None:
    raise ValueError("Cannot convert an unknown Dimension to a Tensor: %s" % d)
  if dtype is not None:
    if dtype not in (dtypes.int32, dtypes.int64):
      raise TypeError("Cannot convert a TensorShape to dtype: %s" % dtype)
  else:
    dtype = dtypes.int32
  if name is None:
    name = "shape_as_tensor"
  return constant(d.value, dtype=dtype, name=name)

ops.register_tensor_conversion_function(
    tensor_shape.Dimension, _dimension_tensor_conversion_function, 100)
