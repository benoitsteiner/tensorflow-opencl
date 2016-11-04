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
"""A library of common shape functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six.moves

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util


def scalar_shape(unused_op):
  """Shape function for ops that output a scalar value."""
  return [tensor_shape.scalar()]


def unchanged_shape(op):
  """Shape function for ops that output an tensor like their first input."""
  return [op.inputs[0].get_shape()]


def unchanged_shape_with_rank(rank):
  """Returns a shape function for ops that constrain the rank of their input.

  Args:
    rank: The exact rank of the input and output.

  Returns:
    A shape function for ops that output a tensor of the same size as their
    input, with a particular rank.
  """

  def _ShapeFunction(op):
    return [op.inputs[0].get_shape().with_rank(rank)]

  return _ShapeFunction


def unchanged_shape_with_rank_at_least(rank):
  """Returns a shape function for ops that constrain the rank of their input.

  Args:
    rank: A lower bound on the rank of the input and output.

  Returns:
    A shape function for ops that output a tensor of the same size as their
    input, with a particular rank.
  """

  def _ShapeFunction(op):
    return [op.inputs[0].get_shape().with_rank_at_least(rank)]

  return _ShapeFunction


def unchanged_shape_with_rank_at_most(rank):
  """Returns a shape function for ops that constrain the rank of their input.

  Args:
    rank: An upper bound on the rank of the input and output.

  Returns:
    A shape function for ops that output a tensor of the same size as their
    input, with a particular rank.
  """

  def _ShapeFunction(op):
    return [op.inputs[0].get_shape().with_rank_at_most(rank)]

  return _ShapeFunction


def matmul_shape(op):
  """Shape function for a MatMul op."""
  a_shape = op.inputs[0].get_shape().with_rank(2)
  transpose_a = op.get_attr("transpose_a")
  b_shape = op.inputs[1].get_shape().with_rank(2)
  transpose_b = op.get_attr("transpose_b")
  output_rows = a_shape[1] if transpose_a else a_shape[0]
  output_cols = b_shape[0] if transpose_b else b_shape[1]
  inner_a = a_shape[0] if transpose_a else a_shape[1]
  inner_b = b_shape[1] if transpose_b else b_shape[0]
  inner_a.assert_is_compatible_with(inner_b)
  return [tensor_shape.TensorShape([output_rows, output_cols])]


def get_conv_output_size(input_size, filter_size, strides, padding_type):
  """Returns the spatial size of a n-d convolution/pooling output."""
  input_size = tuple([tensor_shape.as_dimension(x).value for x in input_size])
  filter_size = tuple([tensor_shape.as_dimension(x).value for x in filter_size])
  strides = [int(x) for x in strides]

  if all(x == 1 for x in input_size) and all(x == 1 for x in filter_size):
    return input_size

  if any(x is not None and y is not None and x > y for x, y in
         zip(filter_size, input_size)):
    raise ValueError("Filter must not be larger than the input: "
                     "Filter: %r Input: %r" % (filter_size, input_size))

  if padding_type == b"VALID":

    def _valid(in_dim, k_dim, s_dim):
      if in_dim is not None and k_dim is not None:
        return (in_dim - k_dim + s_dim) // s_dim
      else:
        return None

    output_size = [
        _valid(in_dim, k_dim, s_dim)
        for in_dim, k_dim, s_dim in zip(input_size, filter_size, strides)
    ]
  elif padding_type == b"SAME":

    def _same(in_dim, s_dim):
      if in_dim is not None:
        return (in_dim + s_dim - 1) // s_dim
      else:
        return None

    output_size = [_same(in_dim, s_dim)
                   for in_dim, s_dim in zip(input_size, strides)]
  else:
    raise ValueError("Invalid padding: %r" % padding_type)

  return tuple(output_size)


def get2d_conv_output_size(input_height, input_width, filter_height,
                           filter_width, row_stride, col_stride, padding_type):
  """Returns the number of rows and columns in a convolution/pooling output."""
  return get_conv_output_size((input_height, input_width),
                              (filter_height, filter_width),
                              (row_stride, col_stride), padding_type)


def conv2d_shape(op):
  """Shape function for a Conv2D op.

  This op has two inputs:

  * input, a 4D tensor with shape = [batch_size, rows, cols, depth_in]
  * filter, a 4D tensor with shape =  [filter_rows, filter_cols,
    depth_in, depth_out]

  The output is a 4D tensor with shape = [batch_size, out_rows,
  out_cols, depth_out], where out_rows and out_cols depend on the
  value of the op's "padding" and "strides" attrs.

  Args:
    op: A Conv2D Operation.

  Returns:
    A list containing the Shape of the Conv2D output.

  Raises:
    ValueError: If the shapes of the input or filter are incompatible.
  """
  input_shape = op.inputs[0].get_shape().with_rank(4)
  filter_shape = op.inputs[1].get_shape().with_rank(4)

  try:
    data_format = op.get_attr("data_format")
  except ValueError:
    data_format = None

  if data_format == b"NCHW":
    # Convert input shape to the default NHWC for inference.
    input_shape = [input_shape[0], input_shape[2], input_shape[3],
                   input_shape[1]]

  batch_size = input_shape[0]
  in_rows = input_shape[1]
  in_cols = input_shape[2]

  filter_rows = filter_shape[0]
  filter_cols = filter_shape[1]
  depth_out = filter_shape[3]
  # Check that the input depths are compatible.
  input_shape[3].assert_is_compatible_with(filter_shape[2])

  if data_format == b"NCHW":
    stride_b, stride_d, stride_r, stride_c = op.get_attr("strides")
  else:
    stride_b, stride_r, stride_c, stride_d = op.get_attr("strides")

  if stride_b != 1 or stride_d != 1:
    raise ValueError("Current implementation does not yet support "
                     "strides in the batch and depth dimensions.")
  # TODO(mrry,shlens): Raise an error if the stride would cause
  # information in the input to be ignored. This will require a change
  # in the kernel implementation.
  padding = op.get_attr("padding")
  out_rows, out_cols = get2d_conv_output_size(in_rows, in_cols, filter_rows,
                                              filter_cols, stride_r, stride_c,
                                              padding)

  output_shape = [batch_size, out_rows, out_cols, depth_out]
  if data_format == b"NCHW":
    # Convert output shape back to NCHW.
    output_shape = [output_shape[0], output_shape[3], output_shape[1],
                    output_shape[2]]
  return [tensor_shape.TensorShape(output_shape)]


def depthwise_conv2d_native_shape(op):
  """Shape function for a DepthwiseConv2D op.

  This op has two inputs:

  * input, a 4D tensor with shape = [batch_size, rows, cols, depth_in]
  * filter, a 4D tensor with shape =  [filter_rows, filter_cols,
    depth_in, depthwise_multiplier]

  The output is a 4D tensor with shape = [batch_size, out_rows,
  out_cols, depth_in*depthwise_multiplier], where out_rows and out_cols depend
  on the value of the op's "padding" and "strides" attrs.

  Args:
    op: A DepthwiseConv2dNative Operation.

  Returns:
    A list containing the Shape of the DepthwiseConv2DNative output.

  Raises:
    ValueError: If the shapes of the input or filter are incompatible.
  """
  input_shape = op.inputs[0].get_shape().with_rank(4)
  filter_shape = op.inputs[1].get_shape().with_rank(4)

  batch_size = input_shape[0]
  in_rows = input_shape[1]
  in_cols = input_shape[2]

  filter_rows = filter_shape[0]
  filter_cols = filter_shape[1]
  depth_out = filter_shape[3] * filter_shape[2]
  # Check that the input depths are compatible.
  input_shape[3].assert_is_compatible_with(filter_shape[2])

  stride_b, stride_r, stride_c, stride_d = op.get_attr("strides")
  if stride_b != 1 or stride_d != 1:
    raise ValueError("Current implementation does not yet support "
                     "strides in the batch and depth dimensions.")
  if stride_r != stride_c:
    # TODO(shlens): Add support for this.
    raise ValueError("Current implementation only supports equal length "
                     "strides in the row and column dimensions.")

  # TODO(mrry,shlens): Raise an error if the stride would cause
  # information in the input to be ignored. This will require a change
  # in the kernel implementation.
  stride = stride_r
  padding = op.get_attr("padding")
  out_rows, out_cols = get2d_conv_output_size(in_rows, in_cols, filter_rows,
                                              filter_cols, stride, stride,
                                              padding)

  return [tensor_shape.TensorShape([batch_size, out_rows, out_cols, depth_out])]


def separable_conv2d_shape(op):
  """Shape function for a SeparableConv2D op.

  This op has three inputs:

  * input, a 4D tensor with shape = [batch_size, rows, cols, depth_in]

  * depthwise_filter, a 4D tensor with shape = [filter_rows,
    filter_cols, depth_in, depth_multiplier]

  * pointwise_filter, a 4D tensor with shape = [1, 1, depth_in *
    depth_multiplier, depth_out]

  The output is a 4D tensor with shape = [batch_size, out_rows,
  out_cols, depth_out], where out_rows and out_cols depend on the
  value of the op's "padding" and "strides" attrs.

  Args:
    op: A SeparableConv2D Operation.

  Returns:
    A list containing the Shape of the SeparableConv2D output.

  Raises:
    ValueError: If the shapes of the input or filter are incompatible.
  """
  input_shape = op.inputs[0].get_shape().with_rank(4)
  depthwise_filter_shape = op.inputs[1].get_shape().merge_with(
      tensor_shape.TensorShape([None, None, input_shape[3], None]))
  pointwise_depth_in = depthwise_filter_shape[2] * depthwise_filter_shape[3]

  pointwise_filter_shape = op.inputs[2].get_shape().merge_with(
      tensor_shape.TensorShape([1, 1, pointwise_depth_in, None]))

  batch_size = input_shape[0]
  in_rows = input_shape[1]
  in_cols = input_shape[2]

  filter_rows = depthwise_filter_shape[0]
  filter_cols = depthwise_filter_shape[1]
  depth_out = pointwise_filter_shape[3]

  stride_b, stride_r, stride_c, stride_d = op.get_attr("strides")
  if stride_b != 1 or stride_d != 1:
    raise ValueError("Current implementation does not yet support "
                     "strides in the batch and depth dimensions.")
  if stride_r != stride_c:
    # TODO(shlens): Add support for this.
    raise ValueError("Current implementation only supports equal length "
                     "strides in the row and column dimensions.")

  # TODO(mrry,shlens): Raise an error if the stride would cause
  # information in the input to be ignored. This will require a change
  # in the kernel implementation.
  stride = stride_r
  padding = op.get_attr("padding")
  out_rows, out_cols = get2d_conv_output_size(in_rows, in_cols, filter_rows,
                                              filter_cols, stride, stride,
                                              padding)

  return [tensor_shape.TensorShape([batch_size, out_rows, out_cols, depth_out])]


def avg_pool_shape(op):
  """Shape function for an AvgPool op.

  This op has one input:

  * input, a 4D tensor with shape = [batch_size, rows, cols, depth]

  The output is a 4D tensor with shape = [batch_size, out_rows,
  out_cols, depth_out], where out_rows and out_cols depend on the
  value of the op's "ksize", "strides", and "padding" attrs.

  Args:
    op: An AvgPool Operation.

  Returns:
    A single-element list containing the Shape of the AvgPool output.

  Raises:
    ValueError: If the shape of the input is invalid or incompatible with
      the values of the attrs.
  """
  input_shape = op.inputs[0].get_shape().with_rank(4)
  try:
    data_format = op.get_attr("data_format")
  except ValueError:
    data_format = None

  if data_format == b"NCHW":
    # Convert input shape to the default NHWC for inference.
    input_shape = [input_shape[0], input_shape[2], input_shape[3],
                   input_shape[1]]

  if data_format == b"NCHW":
    ksize_b, ksize_d, ksize_r, ksize_c = op.get_attr("ksize")
    stride_b, stride_d, stride_r, stride_c = op.get_attr("strides")
  else:
    ksize_b, ksize_r, ksize_c, ksize_d = op.get_attr("ksize")
    stride_b, stride_r, stride_c, stride_d = op.get_attr("strides")

  batch_size = input_shape[0]
  in_rows = input_shape[1]
  in_cols = input_shape[2]
  depth = input_shape[3]

  if ksize_b != 1 or ksize_d != 1:
    raise ValueError("Current implementation does not support pooling "
                     "in the batch and depth dimensions.")
  if stride_b != 1 or stride_d != 1:
    raise ValueError("Current implementation does not support strides "
                     "in the batch and depth dimensions.")

  # TODO(mrry,shlens): Raise an error if the stride would cause
  # information in the input to be ignored. This will require a change
  # in the kernel implementation.
  padding = op.get_attr("padding")

  out_rows, out_cols = get2d_conv_output_size(in_rows, in_cols, ksize_r,
                                              ksize_c, stride_r, stride_c,
                                              padding)

  output_shape = [batch_size, out_rows, out_cols, depth]
  if data_format == b"NCHW":
    # Convert output shape back to NCHW.
    output_shape = [output_shape[0], output_shape[3], output_shape[1],
                    output_shape[2]]
  return [tensor_shape.TensorShape(output_shape)]


def max_pool_shape(op):
  """Shape function for a MaxPool op.

  This op has one input:

  * input, a 4D tensor with shape = [batch_size, rows, cols, depth_in]

  The output is a 4D tensor with shape = [batch_size, out_rows,
  out_cols, depth_out], where out_rows, out_cols, and depth_out depend
  on the value of the op's "ksize", "strides", and "padding" attrs.

  Args:
    op: A MaxPool Operation.

  Returns:
    A single-element list containing the Shape of the MaxPool output.

  Raises:
    ValueError: If the shape of the input is invalid or incompatible with
      the values of the attrs.
  """
  input_shape = op.inputs[0].get_shape().with_rank(4)
  try:
    data_format = op.get_attr("data_format")
  except ValueError:
    data_format = None

  if data_format == b"NCHW":
    # Convert input shape to the default NHWC for inference.
    input_shape = [input_shape[0], input_shape[2], input_shape[3],
                   input_shape[1]]

  if data_format == b"NCHW":
    ksize_b, ksize_d, ksize_r, ksize_c = op.get_attr("ksize")
    stride_b, stride_d, stride_r, stride_c = op.get_attr("strides")
  else:
    ksize_b, ksize_r, ksize_c, ksize_d = op.get_attr("ksize")
    stride_b, stride_r, stride_c, stride_d = op.get_attr("strides")

  batch_size = input_shape[0]
  in_rows = input_shape[1]
  in_cols = input_shape[2]
  depth = input_shape[3]

  if ksize_b != 1:
    raise ValueError("Current implementation does not support pooling "
                     "in the batch dimension.")
  if stride_b != 1:
    raise ValueError("Current implementation does not support strides "
                     "in the batch dimension.")

  if not ((ksize_r == 1 and ksize_c == 1) or ksize_d == 1):
    raise ValueError("MaxPooling supports exactly one of pooling across depth "
                     "or pooling across width/height.")

  # TODO(mrry,shlens): Raise an error if the stride would cause
  # information in the input to be ignored. This will require a change
  # in the kernel implementation.
  if ksize_d == 1:
    padding = op.get_attr("padding")
    out_rows, out_cols = get2d_conv_output_size(in_rows, in_cols, ksize_r,
                                                ksize_c, stride_r, stride_c,
                                                padding)
    output_shape = [batch_size, out_rows, out_cols, depth]
  else:
    if depth % ksize_d > 0:
      raise ValueError("Depthwise max pooling requires the depth window "
                       "to evenly divide the input depth.")
    if stride_d != ksize_d:
      raise ValueError("Depthwise max pooling requires the depth window "
                       "to equal the depth stride.")
    output_shape = [batch_size, in_rows, in_cols, depth // ksize_d]

  if data_format == b"NCHW":
    # Convert output shape back to NCHW.
    output_shape = [output_shape[0], output_shape[3], output_shape[1],
                    output_shape[2]]
  return [tensor_shape.TensorShape(output_shape)]


def no_outputs(unused_op):
  """Shape function for use with ops that have no outputs."""
  return []


def unknown_shape(op):
  """Shape function for use with ops whose output shapes are unknown."""
  return [tensor_shape.unknown_shape() for _ in op.outputs]


def broadcast_shape(shape_x, shape_y):
  """Returns the broadcasted shape between `shape_x` and `shape_y`.

  Args:
    shape_x: A `TensorShape`
    shape_y: A `TensorShape`

  Returns:
    A `TensorShape` representing the broadcasted shape.

  Raises:
    ValueError: If the two shapes can not be broadcasted.
  """
  if shape_x.ndims is None or shape_y.ndims is None:
    return tensor_shape.unknown_shape()

  # To compute the broadcasted dimensions, we zip together shape_x and shape_y,
  # and pad with 1 to make them the same length.
  broadcasted_dims = reversed(list(six.moves.zip_longest(
      reversed(shape_x.dims),
      reversed(shape_y.dims),
      fillvalue=tensor_shape.Dimension(1))))
  # Next we combine the dimensions according to the numpy broadcasting rules.
  # http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
  return_dims = []
  for (dim_x, dim_y) in broadcasted_dims:
    if dim_x.value is None or dim_y.value is None:
      # One or both dimensions is unknown. If either dimension is greater than
      # 1, we assume that the program is correct, and the other dimension will
      # be broadcast to match it.
      # TODO(mrry): If we eliminate the shape checks in C++, we must still
      # assert that the unknown dim is either 1 or the same as the known dim.
      if dim_x.value is not None and dim_x.value > 1:
        return_dims.append(dim_x)
      elif dim_y.value is not None and dim_y.value > 1:
        return_dims.append(dim_y)
      else:
        return_dims.append(None)
    elif dim_x.value == 1:
      # We will broadcast dim_x to dim_y.
      return_dims.append(dim_y)
    elif dim_y.value == 1:
      # We will broadcast dim_y to dim_x.
      return_dims.append(dim_x)
    elif dim_x.value == dim_y.value:
      # The dimensions are compatible, so output is the same size in that
      # dimension.
      return_dims.append(dim_x.merge_with(dim_y))
    else:
      raise ValueError("Incompatible shapes for broadcasting: %s and %s"
                       % (shape_x, shape_y))
  return tensor_shape.TensorShape(return_dims)


def call_cpp_shape_fn(op,
                      input_tensors_needed=None,
                      input_tensors_as_shapes_needed=None,
                      debug_python_shape_fn=None):
  """A shape function that delegates to the registered C++ shape function.

  Args:
    op: the node in the graph for which to compute output shapes.
    input_tensors_needed: a list of input tensor indices for which to compute
      the input tensor's value and pass to the C++ shape function.
    input_tensors_as_shapes_needed: a list of input tensor indices for which to
      compute the constant_value_as_shape and pass to the C++ shape function.
    debug_python_shape_fn: For testing only during migration to using
      call_cpp_shape_fn. Do not submit calls that set this,
      as the comparison is slow. If non-None, the python shape function;
      this function will be called and its output compared to that of
      the C++ shape function.

  Returns:
    A dictionary with the following keys:
      shapes: A TensorShape list of the output shapes of the op, as computed
        using the C++ shape inference function registered for the op.
      handle_shapes: A TensorShape list of the shapes for handle outputs, if
         any.
      handle_dtypes: A list of DataType enums for the handle outputs, if any.

  Raises:
    ValueError: If the C++ shape function returned an error (e.g. because the
    shapes of the inputs are of the wrong rank or otherwise incompatible
    according to the shape function).
  """
  node_def_str = op.node_def.SerializeToString()

  def tensor_to_inference_result(t):
    r = cpp_shape_inference_pb2.CppShapeInferenceResult()
    r.shape.CopyFrom(t.get_shape().as_proto())
    # pylint: disable=protected-access
    r.handle_shape.CopyFrom(t._handle_shape)
    r.handle_dtype = t._handle_dtype
    # pylint: enable=protected-access
    return r.SerializeToString()
  input_shapes = [tensor_to_inference_result(i) for i in op.inputs]

  input_tensors = [None for i in input_shapes]
  if input_tensors_needed:
    for idx in input_tensors_needed:
      v = tensor_util.constant_value(op.inputs[idx])
      if v is not None:
        input_tensors[idx] = np.asarray(v)

  serialized_unknown_shape = (
      tensor_shape.TensorShape(None).as_proto().SerializeToString())
  arr = [serialized_unknown_shape for i in input_shapes]
  if input_tensors_as_shapes_needed:
    for idx in input_tensors_as_shapes_needed:
      s = tensor_util.constant_value_as_shape(op.inputs[idx])
      if s is not None:
        arr[idx] = s.as_proto().SerializeToString()
  input_tensors_as_shapes = arr

  try:
    with errors.raise_exception_on_not_ok_status() as status:
      output_shapes = pywrap_tensorflow.RunCppShapeInference(
          node_def_str, input_shapes, input_tensors, input_tensors_as_shapes,
          status)
  except errors.InvalidArgumentError as err:
    raise ValueError(err.message)

  # Convert TensorShapeProto values in output_shapes.
  result_protos = [
      cpp_shape_inference_pb2.CppShapeInferenceResult().FromString(s)
      for s in output_shapes
  ]
  result = [r.shape for r in result_protos]
  result_handle_shapes = [r.handle_shape for r in result_protos]
  result_handle_dtypes = [r.handle_dtype for r in result_protos]

  if debug_python_shape_fn:
    try:
      python_result = [tensor_shape.as_shape(s)
                       for s in debug_python_shape_fn(op)]
    except Exception as err:
      raise AssertionError("Python shape function return error but "
                           "C++ shape functon did not: %s" % str(err))
    if str(result) != str(python_result):
      raise ValueError(
          ("Python vs CPP shape mismatch.  "
           "CPP: %s vs python: %s on node %s "
           "with input shapes %s") % (
               str(result), str(python_result), str(op.node_def),
               ",".join([str(i.get_shape()) for i in op.inputs])))

  return {"shapes": result,
          "handle_shapes": result_handle_shapes,
          "handle_dtypes": result_handle_dtypes}
