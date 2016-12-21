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

"""Gradients for operators defined in array_ops.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from math import ceil

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops


@ops.RegisterGradient("Pack")
def _PackGrad(op, grad):
  """Gradient for pack op."""
  return array_ops.unstack(grad, num=op.get_attr("N"), axis=op.get_attr("axis"))


@ops.RegisterGradient("Unpack")
def _UnpackGrad(op, *grads):
  """Gradient for unpack op."""
  return array_ops.stack(grads, axis=op.get_attr("axis"))


def _ConcatGradHelper(op, grad, start_value_index, end_value_index, dim_index):
  """Gradient for concat op.

  Args:
    op: An operation.
    grad: `Tensor` or `IndexedSlices` representing the gradients with respect
      to each output of the op.
    start_value_index: An integer index of the first value in the op.inputs.
    end_value_index: An integer index of the last value in the op.inputs.
    dim_index: An interger index of concat_dim or axis parameter in op.inputs.

  Returns:
    Tensors represending the partial gradients with respect to each input
    of the op.

  Raises:
    ValueError: if concat_dim/axis is not statically known.
  """

  def _CreateDenseMaskAndBegin(sizes, concat_dim):
    """Create variables for iteratively slicing a dense gradients tensor."""
    # Since shape is 1-D, shape_of_shape = [rank-of-inputs]
    shape_of_shape = array_ops.shape(sizes[dim_index])
    # Make a vector of length equal to the input's dimensions,
    # with 0's everywhere and 1 in the concat dim position.
    # Note: Can't use sparse_to_dense since it isn't GPU-capable (for now)
    mask = array_ops.concat_v2(
        [array_ops.fill(
            array_ops.expand_dims(concat_dim, 0), 0),
         [1],
         array_ops.fill(
             shape_of_shape - concat_dim - 1, 0)],
        0)
    begin = array_ops.fill(shape_of_shape, 0)
    return mask, begin

  def _ExtractInputShapes(inputs):
    """Extract the shapes of a set of input tensors."""
    sizes = []
    fully_known = True
    for x in inputs:
      input_shape = array_ops.shape(x)
      if not isinstance(input_shape,
                        ops.Tensor) or input_shape.op.type != "Const":
        fully_known = False
        break
      else:
        sizes.append(input_shape)

    if fully_known:
      return sizes
    else:
      return array_ops.shape_n(inputs)

  # Degenerate concatenation, just return grad.
  if len(op.inputs) == 2:
    return grad + [None] if end_value_index <= dim_index else [None] + grad

  concat_dim = op.inputs[dim_index]
  input_values = op.inputs[start_value_index:end_value_index]
  out_grads = []
  if isinstance(grad, ops.Tensor):
    # Get the inputs' tensor shapes
    sizes = _ExtractInputShapes(input_values)
    # The magic number of 16 was found through benchmarking a range of sizes
    # on CPUs and a Maxwell TitanX.  A speedup was seen in a large majority of
    # cases when switching implementations at N=16, but it is possible that
    # there will be a small number of performance regressions.
    # pylint: disable=protected-access
    if len(sizes) > 16:
      # extract the size of each input along the concat dimension
      sizes = array_ops.squeeze(
          array_ops.slice(
              array_ops.stack(
                  sizes, axis=1), [concat_dim, 0], [1, -1]))
      out_grads = array_ops.split(grad, sizes, concat_dim)
    else:
      offset = gen_array_ops._concat_offset(concat_dim, sizes)
      for (begin, size) in zip(offset, sizes):
        out_grads.append(array_ops.slice(grad, begin, size))
    # pylint: enable=protected-access
  elif isinstance(grad, ops.IndexedSlices):
    concat_dim_static = tensor_util.constant_value(concat_dim)
    if concat_dim_static is None:
      raise ValueError("Can only compute IndexedSlices gradient with "
                       "statically-known concat_dim")
    # Get the inputs' tensor shapes
    sizes = [array_ops.shape(x) for x in input_values]
    if concat_dim_static > 0:
      # IndexedSlices, concat_dim > 0. Each input gets IndexedSlices gradients
      # with all the indices, but with grad.values sliced accordingly. This
      # is like the Tensor case, except shape(grad.values)[0] is not equal to
      # shape(sizes[i])[0], since only a subset of the dim-0 values are stored.
      mask, begin = _CreateDenseMaskAndBegin(sizes, concat_dim)
      for size in sizes:
        new_values = array_ops.slice(
            grad.values,
            begin,
            array_ops.concat_v2(
                [[-1], array_ops.slice(size, [1], [-1])], 0))
        out_grads.append(
            ops.IndexedSlices(new_values, grad.indices, size))
        # Lint complains begin = begin + ...
        begin = math_ops.add(begin, size * mask)
    else:
      # IndexedSlices, concat_dim == 0. Each input gets IndexedSlices gradients
      # only for the relevant indices.
      start = constant_op.constant(0, dtype=grad.indices.dtype)
      for size in sizes:
        size_concat_dim = array_ops.gather(size, concat_dim)
        if size_concat_dim.dtype != grad.indices.dtype:
          size_concat_dim = math_ops.cast(size_concat_dim,
                                          dtype=grad.indices.dtype)
        end = start + size_concat_dim
        # Compute the 1-D Tensor of indices relevant for this input.
        indices_to_select = array_ops.squeeze(
            array_ops.where(math_ops.logical_and(grad.indices >= start,
                                                 grad.indices < end)),
            squeeze_dims=[1])
        new_indices = array_ops.gather(grad.indices, indices_to_select) - start
        new_values = array_ops.gather(grad.values, indices_to_select)
        out_grads.append(
            ops.IndexedSlices(new_values, new_indices, size))
        start = end
  else:
    raise TypeError("Expected Tensor or IndexedSlices, got %s" % type(grad))

  return (out_grads + [None] if end_value_index <= dim_index
          else [None] + out_grads)


@ops.RegisterGradient("Concat")
def _ConcatGrad(op, grad):
  return _ConcatGradHelper(
      op, grad, start_value_index=1, end_value_index=len(op.inputs),
      dim_index=0)


@ops.RegisterGradient("ConcatV2")
def _ConcatGradV2(op, grad):
  return _ConcatGradHelper(
      op, grad, start_value_index=0, end_value_index=-1, dim_index=-1)


ops.NotDifferentiable("ConcatOffset")


@ops.RegisterGradient("Slice")
def _SliceGrad(op, grad):
  """Gradient for Slice op."""
  # Create an Nx2 padding where the first column represents how many
  # zeros are to be prepended for each dimension, and the second
  # column indicates how many zeros are appended.
  #
  # The number of zeros to append is the shape of the input
  # elementwise-subtracted by both the begin vector and sizes vector.
  #
  # Some more reshaping is needed to assemble this tensor with the
  # right dimensions.
  input_vec = op.inputs[0]
  begin_vec = op.inputs[1]
  input_rank = array_ops.rank(input_vec)
  slice_size = array_ops.shape(op.outputs[0])

  shape = array_ops.stack([input_rank, 1])
  before_pad = array_ops.reshape(begin_vec, shape)
  after_pad = array_ops.reshape(
      array_ops.shape(input_vec) - slice_size - begin_vec, shape)
  paddings = array_ops.concat_v2([before_pad, after_pad], 1)
  return array_ops.pad(grad, paddings), None, None


@ops.RegisterGradient("StridedSlice")
def _StridedSliceGrad(op, grad):
  """Gradient for StridedSlice op."""
  x = array_ops.shape(op.inputs[0])
  begin = op.inputs[1]
  end = op.inputs[2]
  strides = op.inputs[3]

  return array_ops.strided_slice_grad(
      x,
      begin,
      end,
      strides,
      grad,
      begin_mask=op.get_attr("begin_mask"),
      end_mask=op.get_attr("end_mask"),
      ellipsis_mask=op.get_attr("ellipsis_mask"),
      new_axis_mask=op.get_attr("new_axis_mask"),
      shrink_axis_mask=op.get_attr("shrink_axis_mask")), None, None, None


@ops.RegisterGradient("StridedSliceGrad")
def _StridedSliceGradGrad(op, grad):
  """Gradient for StridedSliceGrad op."""
  begin = op.inputs[1]
  end = op.inputs[2]
  strides = op.inputs[3]

  return None, None, None, None, array_ops.strided_slice(
      grad,
      begin,
      end,
      strides,
      begin_mask=op.get_attr("begin_mask"),
      end_mask=op.get_attr("end_mask"),
      ellipsis_mask=op.get_attr("ellipsis_mask"),
      new_axis_mask=op.get_attr("new_axis_mask"),
      shrink_axis_mask=op.get_attr("shrink_axis_mask"))


@ops.RegisterGradient("Split")
def _SplitGrad(op, *grads):
  return None, array_ops.concat_v2(list(grads), op.inputs[0])

@ops.RegisterGradient("SplitV")
def _SplitVGrad(op, *grads):
  returnval = array_ops.concat_v2(list(grads), op.inputs[2])
  returnval = [returnval] + [None,] * (len(op.inputs) - 1)
  print(returnval)
  return returnval

ops.NotDifferentiable("Const")


@ops.RegisterGradient("Diag")
def _DiagGrad(_, grad):
  return array_ops.diag_part(grad)


@ops.RegisterGradient("DiagPart")
def _DiagPartGrad(_, grad):
  return array_ops.diag(grad)


@ops.RegisterGradient("MatrixDiag")
def _MatrixDiagGrad(_, grad):
  return array_ops.matrix_diag_part(grad)


@ops.RegisterGradient("MatrixDiagPart")
def _MatrixDiagPartGrad(op, grad):
  matrix_shape = op.inputs[0].get_shape()[-2:]
  if matrix_shape.is_fully_defined() and matrix_shape[0] == matrix_shape[1]:
    return array_ops.matrix_diag(grad)
  else:
    return array_ops.matrix_set_diag(array_ops.zeros_like(op.inputs[0]), grad)


@ops.RegisterGradient("MatrixSetDiag")
def _MatrixSetDiagGrad(op, grad):
  input_shape = op.inputs[0].get_shape().merge_with(grad.get_shape())
  diag_shape = op.inputs[1].get_shape()
  batch_shape = input_shape[:-2].merge_with(diag_shape[:-1])
  matrix_shape = input_shape[-2:]
  if batch_shape.is_fully_defined() and matrix_shape.is_fully_defined():
    diag_shape = batch_shape.as_list() + [min(matrix_shape.as_list())]
  else:
    with ops.colocate_with(grad):
      grad_shape = array_ops.shape(grad)
      grad_rank = array_ops.rank(grad)
      batch_shape = array_ops.slice(grad_shape, [0], [grad_rank - 2])
      matrix_shape = array_ops.slice(grad_shape, [grad_rank - 2], [2])
      min_dim = math_ops.reduce_min(matrix_shape)
      diag_shape = array_ops.concat_v2([batch_shape, [min_dim]], 0)
  grad_input = array_ops.matrix_set_diag(
      grad, array_ops.zeros(
          diag_shape, dtype=grad.dtype))
  grad_diag = array_ops.matrix_diag_part(grad)
  return (grad_input, grad_diag)


@ops.RegisterGradient("MatrixBandPart")
def _MatrixBandPartGrad(op, grad):
  num_lower = op.inputs[1]
  num_upper = op.inputs[2]
  return (array_ops.matrix_band_part(grad, num_lower, num_upper), None, None)


# Edit Distance has no gradient (but can be used to eval seq2seq or CTC).
ops.NotDifferentiable("EditDistance")


@ops.RegisterGradient("Fill")
def _FillGrad(_, grad):
  return None, math_ops.reduce_sum(grad)


ops.NotDifferentiable("ZerosLike")


@ops.RegisterGradient("Gather")
def _GatherGrad(op, grad):
  """Gradient for Gather op."""
  # params can be large, so colocate the shape calculation with it.
  params = op.inputs[0]
  with ops.colocate_with(params):
    params_shape = array_ops.shape(params)

  # Build appropriately shaped IndexedSlices
  indices = op.inputs[1]
  size = array_ops.expand_dims(array_ops.size(indices), 0)
  values_shape = array_ops.concat_v2([size, params_shape[1:]], 0)
  values = array_ops.reshape(grad, values_shape)
  indices = array_ops.reshape(indices, size)
  return [ops.IndexedSlices(values, indices, params_shape), None]


@ops.RegisterGradient("GatherNd")
def _GatherNdGrad(op, grad):
  ref = op.inputs[0]
  ref_shape = array_ops.shape(ref)
  indices = op.inputs[1]
  ref_grad = array_ops.scatter_nd(indices, grad, ref_shape)
  return [ref_grad, None]


@ops.RegisterGradient("CheckNumerics")
def _CheckNumericsGrad(_, grad):
  """Gradient for check_numerics op."""
  return array_ops.check_numerics(
      grad, "Not a number (NaN) or infinity (Inf) values detected in gradient.")


@ops.RegisterGradient("Identity")
def _IdGrad(_, grad):
  return grad


@ops.RegisterGradient("RefIdentity")
def _RefIdGrad(_, grad):
  return grad


ops.NotDifferentiable("StopGradient")


@ops.RegisterGradient("Reshape")
def _ReshapeGrad(op, grad):
  return [array_ops.reshape(grad, array_ops.shape(op.inputs[0])), None]


ops.NotDifferentiable("InvertPermutation")


def _ReshapeToInput(op, grad):
  """Reshapes the gradient to the shape of the original input."""
  return array_ops.reshape(grad, array_ops.shape(op.inputs[0]))


@ops.RegisterGradient("ExpandDims")
def _ExpandDimsGrad(op, grad):
  return [_ReshapeToInput(op, grad), None]


@ops.RegisterGradient("Squeeze")
def _SqueezeGrad(op, grad):
  return _ReshapeToInput(op, grad)


@ops.RegisterGradient("Transpose")
def _TransposeGrad(op, grad):
  """Returns unshuffle(grad)."""
  p = op.inputs[1]
  return [array_ops.transpose(grad, array_ops.invert_permutation(p)), None]


ops.NotDifferentiable("Shape")


ops.NotDifferentiable("ShapeN")


ops.NotDifferentiable("Rank")


ops.NotDifferentiable("Size")


@ops.RegisterGradient("Tile")
def _TileGrad(op, grad):
  """Sum reduces grad along the tiled dimensions."""
  assert isinstance(grad, ops.Tensor)
  input_shape = array_ops.shape(op.inputs[0])
  # We interleave multiples and input_shape to get split_shape,
  # reshape grad to split_shape, and reduce along all even
  # dimensions (the tiled dimensions) to get the result
  # with shape input_shape.  For example
  #   input_shape = [20, 30, 40]
  #   multiples = [2, 3, 4]
  #   split_shape = [2, 20, 3, 30, 4, 40]
  #   axes = [0, 2, 4]
  split_shape = array_ops.reshape(
      array_ops.transpose(array_ops.stack([op.inputs[1], input_shape])), [-1])
  axes = math_ops.range(0, array_ops.size(split_shape), 2)
  input_grad = math_ops.reduce_sum(array_ops.reshape(grad, split_shape), axes)
  # Fix shape inference
  input_grad.set_shape(op.inputs[0].get_shape())
  return [input_grad, None]


ops.NotDifferentiable("BroadcastGradientArgs")


@ops.RegisterGradient("Pad")
def _PadGrad(op, grad):
  """Gradient for Pad."""
  # Pad introduces values around the original tensor, so the gradient function
  # slices the original shape out of the gradient."""
  x = op.inputs[0]
  a = op.inputs[1]  # [Rank(x), 2]
  # Takes a slice of a. The 1st column. [Rank(x), 1].
  pad_before = array_ops.slice(a, [0, 0],
                               array_ops.stack([array_ops.rank(x), 1]))
  # Make it a 1-D tensor.
  begin = array_ops.reshape(pad_before, [-1])
  sizes = array_ops.shape(x)
  return array_ops.slice(grad, begin, sizes), None


# ReverseSequence is just a permutation.  The gradient permutes back.
@ops.RegisterGradient("ReverseSequence")
def _ReverseSequenceGrad(op, grad):
  seq_lengths = op.inputs[1]
  return [
      array_ops.reverse_sequence(
          grad,
          batch_axis=op.get_attr("batch_dim"),
          seq_axis=op.get_attr("seq_dim"),
          seq_lengths=seq_lengths), None
  ]


@ops.RegisterGradient("Reverse")
def _ReverseGrad(op, grad):
  reverse_dims = op.inputs[1]
  # pylint: disable=protected-access
  return gen_array_ops._reverse(grad, reverse_dims), None
  # pylint: enable=protected-access


@ops.RegisterGradient("ReverseV2")
def _ReverseV2Grad(op, grad):
  axis = op.inputs[1]
  return array_ops.reverse_v2(grad, axis), None


@ops.RegisterGradient("SpaceToBatch")
def _SpaceToBatchGrad(op, grad):
  # Its gradient is the opposite op: BatchToSpace.
  block_size = op.get_attr("block_size")
  return [array_ops.batch_to_space(grad, op.inputs[1], block_size=block_size),
          None]


@ops.RegisterGradient("SpaceToBatchND")
def _SpaceToBatchNDGrad(op, grad):
  # Its gradient is the opposite op: BatchToSpaceND.
  return [array_ops.batch_to_space_nd(grad, op.inputs[1], op.inputs[2]),
          None, None]


@ops.RegisterGradient("BatchToSpace")
def _BatchToSpaceGrad(op, grad):
  # Its gradient is the opposite op: SpaceToBatch.
  block_size = op.get_attr("block_size")
  return [array_ops.space_to_batch(grad, op.inputs[1], block_size=block_size),
          None]


@ops.RegisterGradient("BatchToSpaceND")
def _BatchToSpaceNDGrad(op, grad):
  # Its gradient is the opposite op: SpaceToBatchND.
  return [array_ops.space_to_batch_nd(grad, op.inputs[1], op.inputs[2]),
          None, None]


@ops.RegisterGradient("SpaceToDepth")
def _SpaceToDepthGrad(op, grad):
  # Its gradient is the opposite op: DepthToSpace.
  block_size = op.get_attr("block_size")
  return array_ops.depth_to_space(grad, block_size)


@ops.RegisterGradient("DepthToSpace")
def _DepthToSpaceGrad(op, grad):
  # Its gradient is the opposite op: SpaceToDepth.
  block_size = op.get_attr("block_size")
  return array_ops.space_to_depth(grad, block_size)


ops.NotDifferentiable("OneHot")


@ops.RegisterGradient("MirrorPad")
def _MirrorPadGrad(op, grad):
  mode = op.get_attr("mode")
  # pylint: disable=protected-access
  return [gen_array_ops._mirror_pad_grad(grad, op.inputs[1], mode=mode), None]
  # pylint: enable=protected-access


@ops.RegisterGradient("MirrorPadGrad")
def _MirrorPadGradGrad(op, grad):
  mode = op.get_attr("mode")
  # pylint: disable=protected-access
  return [gen_array_ops._mirror_pad(grad, op.inputs[1], mode=mode), None]
  # pylint: enable=protected-access


@ops.RegisterGradient("QuantizeAndDequantize")
def _QuantizeAndDequantizeGrad(_, grad):
  return grad


@ops.RegisterGradient("ExtractImagePatches")
def _ExtractImagePatchesGrad(op, grad):

  batch_size, rows_in, cols_in, channels = [
    dim.value for dim in op.inputs[0].get_shape()
  ]
  input_bhwc = array_ops.shape(op.inputs[0])
  batch_size = input_bhwc[0]
  channels = input_bhwc[3]

  _, rows_out, cols_out, _ = [
    dim.value for dim in op.outputs[0].get_shape()
  ]
  _, ksize_r, ksize_c, _ = op.get_attr('ksizes')
  _, stride_r, stride_h, _ = op.get_attr('strides')
  _, rate_r, rate_c, _ = op.get_attr('rates')
  padding = op.get_attr('padding')

  ksize_r_eff = ksize_r + (ksize_r - 1) * (rate_r - 1)
  ksize_c_eff = ksize_c + (ksize_c - 1) * (rate_c - 1)

  if padding == b'SAME':
    rows_out = int(ceil(rows_in / stride_r))
    cols_out = int(ceil(cols_in / stride_h))
    pad_rows = ((rows_out - 1) * stride_r + ksize_r_eff - rows_in) // 2
    pad_cols = ((cols_out - 1) * stride_h + ksize_c_eff - cols_in) // 2

  elif padding == b'VALID':
    rows_out = int(ceil((rows_in - ksize_r_eff + 1) / stride_r))
    cols_out = int(ceil((cols_in - ksize_c_eff + 1) / stride_h))
    pad_rows = (rows_out - 1) * stride_r + ksize_r_eff - rows_in
    pad_cols = (cols_out - 1) * stride_h + ksize_c_eff - cols_in

  pad_rows, pad_cols = max(0, pad_rows), max(0, pad_cols)

  grad_expanded = array_ops.transpose(
    array_ops.reshape(grad, (batch_size, rows_out,
                             cols_out, ksize_r, ksize_c, channels)),
    (1, 2, 3, 4, 0, 5)
  )
  grad_flat = array_ops.reshape(grad_expanded, (-1, batch_size * channels))

  row_steps = range(0, rows_out * stride_r, stride_r)
  col_steps = range(0, cols_out * stride_h, stride_h)

  idx = []
  for i in range(rows_out):
    for j in range(cols_out):
      r_low, c_low = row_steps[i] - pad_rows, col_steps[j] - pad_cols
      r_high, c_high = r_low + ksize_r_eff, c_low + ksize_c_eff

      idx.extend([(r * (cols_in) + c,
                   i * (cols_out * ksize_r * ksize_c) +
                   j * (ksize_r * ksize_c) +
                   ri * (ksize_c) + ci)
                  for (ri, r) in enumerate(range(r_low, r_high, rate_r))
                  for (ci, c) in enumerate(range(c_low, c_high, rate_c))
                  if 0 <= r and r < rows_in and 0 <= c and c < cols_in
      ])

  sp_shape = (rows_in * cols_in,
              rows_out * cols_out * ksize_r * ksize_c)

  sp_mat = sparse_tensor.SparseTensor(
    array_ops.constant(idx, dtype=ops.dtypes.int64),
    array_ops.ones((len(idx),), dtype=ops.dtypes.float32),
    sp_shape
  )

  jac = sparse_ops.sparse_tensor_dense_matmul(sp_mat, grad_flat)

  grad_out = array_ops.reshape(
    jac, (rows_in, cols_in, batch_size, channels)
  )
  grad_out = array_ops.transpose(grad_out, (2, 0, 1, 3))

  return [grad_out]


@ops.RegisterGradient("ScatterNd")
def _ScatterNdGrad(op, grad):
  indices = op.inputs[0]
  updates_grad = array_ops.gather_nd(grad, indices)
  return [None, updates_grad, None]
