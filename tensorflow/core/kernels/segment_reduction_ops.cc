/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/segment_reduction_ops.h"
#include <vector>
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Static routines not in the templated class to reduce code size
static void SegmentReductionValidationHelper(OpKernelContext* context,
                                             const Tensor& input,
                                             const Tensor& segment_ids) {
  OP_REQUIRES(context, TensorShapeUtils::IsVector(segment_ids.shape()),
              errors::InvalidArgument("segment_ids should be a vector."));
  const int64 num_indices = segment_ids.NumElements();
  OP_REQUIRES(context, num_indices == input.dim_size(0),
              errors::InvalidArgument(
                  "segment_ids should be the same size as dimension 0 of"
                  " input."));
}

static bool SegmentReductionDoValidation(OpKernelContext* c,
                                         const Tensor& input,
                                         const Tensor& segment_ids) {
  SegmentReductionValidationHelper(c, input, segment_ids);
  return c->status().ok();
}

// This operator handles reducing segments along the first dimension.
// See core/ops/math_ops.cc for more details.
template <typename Device, class T, class Index, typename Reducer>
class SegmentReductionOp : public OpKernel {
 public:
  explicit SegmentReductionOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& segment_ids = context->input(1);

    if (!SegmentReductionDoValidation(context, input, segment_ids)) {
      return;
    }

    const int64 num_indices = segment_ids.NumElements();
    auto input_flat = input.flat_outer_dims<T>();
    const int64 num_col = input_flat.dimension(1);

    const auto segment_vec = segment_ids.vec<Index>();
    // Note that the current implementation assumes that segment_vec values are
    // sorted.
    const Index output_rows =
        num_indices > 0
            ? internal::SubtleMustCopy(segment_vec(num_indices - 1)) + 1
            : 0;
    OP_REQUIRES(context, output_rows >= 0,
                errors::InvalidArgument("segment ids must be >= 0"));

    TensorShape output_shape = input.shape();
    output_shape.set_dim(0, output_rows);

    // Note that we do not initialize the output buffer with a default value.
    // We require that segment ids be sorted and cover all values (otherwise we
    // return an error).
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    if (num_indices == 0) return;
    OP_REQUIRES(context, output_rows > 0,
                errors::InvalidArgument("segment ids must be >= 0"));
    auto output_flat = output->flat_outer_dims<T>();

#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::DSizes<Eigen::DenseIndex, 1> dims_to_reduce;
    dims_to_reduce[0] = 0;
#else
    Eigen::IndexList<Eigen::type2index<0>> dims_to_reduce;
#endif
    Index start = 0, end = 1;

    Index out_index = internal::SubtleMustCopy(segment_vec(start));
    OP_REQUIRES(context, out_index == 0,
                errors::InvalidArgument("segment ids do not start at 0"));

    // TODO(agarwal): if this loop becomes a bottleneck, consider sharding it
    // across threads.
    Eigen::DSizes<Eigen::DenseIndex, 1> out_slice_shape(num_col);
    while (end <= num_indices) {
      // We initialize next_index to 0 to avoid "warning: 'next_index' may be
      // used uninitialized in this function" in the Mac build (since the
      // compiler isn't smart enough to realize the code is safe).
      Index next_index = 0;
      if (end < num_indices) {
        next_index = internal::SubtleMustCopy(segment_vec(end));
        if (out_index == next_index) {
          ++end;
          continue;
        }
        // We have a new segment here.  Verify that the segment ids grow by one
        // each time, so that we cover every possible output value.
        OP_REQUIRES(
            context, out_index + 1 == next_index,
            errors::InvalidArgument("segment ids are not increasing by 1"));
      }

      // Process segment [start, end)
      const T* in_slice_ptr = &input_flat(start, 0);
      typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>,
                               Eigen::Unaligned>
          OutT;

      OP_REQUIRES(
          context, FastBoundsCheck(out_index, output_rows),
          errors::InvalidArgument(
              "Segment id ", out_index, " out of range [0, ", output_rows,
              "), probably because 'segment_ids' input is not sorted."));
      T* out_slice_ptr = &output_flat(out_index, 0);
      OutT out_slice(out_slice_ptr, out_slice_shape);
      // We don't use out_slice.device(context->eigen_device<Device>)
      // because these pieces of work are likely to be very small and
      // the context switching overhead dwarfs any benefit we get from
      // using another thread to do this work.
      if (start == end - 1) {
        typedef Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor>,
                                 Eigen::Unaligned>
            InT;
        InT in_slice(in_slice_ptr, out_slice_shape);
        out_slice = in_slice;
      } else {
        Eigen::DSizes<Eigen::DenseIndex, 2> in_slice_shape(end - start,
                                                           num_col);
        typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor>,
                                 Eigen::Unaligned>
            InT;
        InT in_slice(in_slice_ptr, in_slice_shape);

        out_slice = in_slice.reduce(dims_to_reduce, Reducer());
      }
      if (end >= num_indices) break;
      start = end;
      ++end;
      out_index = next_index;
    }
  }
};

#define REGISTER_CPU_KERNEL_SEGMENT(name, functor, type, index_type) \
  REGISTER_KERNEL_BUILDER(                                           \
      Name(name)                                                     \
          .Device(DEVICE_CPU)                                        \
          .TypeConstraint<type>("T")                                 \
          .TypeConstraint<index_type>("Tindices"),                   \
      SegmentReductionOp<CPUDevice, type, index_type, functor>)

#define REGISTER_REAL_CPU_KERNELS(type, index_type)                         \
  REGISTER_CPU_KERNEL_SEGMENT(                                              \
      "SegmentSum", Eigen::internal::SumReducer<type>, type, index_type);   \
  REGISTER_CPU_KERNEL_SEGMENT(                                              \
      "SegmentMean", Eigen::internal::MeanReducer<type>, type, index_type); \
  REGISTER_CPU_KERNEL_SEGMENT(                                              \
      "SegmentProd", Eigen::internal::ProdReducer<type>, type, index_type); \
  REGISTER_CPU_KERNEL_SEGMENT(                                              \
      "SegmentMin", Eigen::internal::MinReducer<type>, type, index_type);   \
  REGISTER_CPU_KERNEL_SEGMENT(                                              \
      "SegmentMax", Eigen::internal::MaxReducer<type>, type, index_type)

#define REGISTER_COMPLEX_CPU_KERNELS(type, index_type)                      \
  REGISTER_CPU_KERNEL_SEGMENT(                                              \
      "SegmentSum", Eigen::internal::SumReducer<type>, type, index_type);   \
  REGISTER_CPU_KERNEL_SEGMENT(                                              \
      "SegmentProd", Eigen::internal::ProdReducer<type>, type, index_type)

#define REGISTER_REAL_CPU_KERNELS_ALL(type) \
  REGISTER_REAL_CPU_KERNELS(type, int32);   \
  REGISTER_REAL_CPU_KERNELS(type, int64)

#define REGISTER_COMPLEX_CPU_KERNELS_ALL(type) \
  REGISTER_COMPLEX_CPU_KERNELS(type, int32);   \
  REGISTER_COMPLEX_CPU_KERNELS(type, int64)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_REAL_CPU_KERNELS_ALL);
REGISTER_COMPLEX_CPU_KERNELS_ALL(complex64);
REGISTER_COMPLEX_CPU_KERNELS_ALL(complex128);
#undef REGISTER_CPU_KERNEL_SEGMENT
#undef REGISTER_REAL_CPU_KERNELS
#undef REGISTER_COMPLEX_CPU_KERNELS
#undef REGISTER_REAL_CPU_KERNELS_ALL
#undef REGISTER_COMPLEX_CPU_KERNELS_ALL

namespace functor {

// UnsortedSegmentSumFunctor implementation for CPUDevice.
// todo: Remove duplicate code in UnsortedSegmentSumFunctor and UnsortedSegmentMaxFunctor.
template <typename T, typename Index>
struct UnsortedSegmentSumFunctor<CPUDevice, T, Index>
    : UnsortedSegmentBaseFunctor<CPUDevice, T, Index> {
  void operator()(OpKernelContext* ctx, const CPUDevice& d,
                  const Index output_rows, const TensorShape& segment_ids_shape,
                  typename TTypes<Index>::ConstFlat segment_ids,
                  const Index data_size, const T* data,
                  typename TTypes<T, 2>::Tensor output) override {
    output.setZero();
    if (data_size == 0) {
      return;
    }
    const int64 N = segment_ids.dimension(0);
    auto data_flat = typename TTypes<T, 2>::ConstTensor(data, N, data_size / N);
    for (int64 i = 0; i < N; ++i) {
      Index j = internal::SubtleMustCopy(segment_ids(i));
      OP_REQUIRES(ctx, FastBoundsCheck(j, output_rows),
                  errors::InvalidArgument(
                      "segment_ids", SliceDebugString(segment_ids_shape, i),
                      " = ", j, " is out of range [0, ", output_rows, ")"));
      output.template chip<0>(j) += data_flat.template chip<0>(i);
    }
  }
};
// UnsortedSegmentMaxFunctor implementation for CPUDevice.
template <typename T, typename Index>
struct UnsortedSegmentMaxFunctor<CPUDevice, T, Index>
    : UnsortedSegmentBaseFunctor<CPUDevice, T, Index> {
  void operator()(OpKernelContext* ctx, const CPUDevice& d,
                  const Index output_rows, const TensorShape& segment_ids_shape,
                  typename TTypes<Index>::ConstFlat segment_ids,
                  const Index data_size, const T* data,
                  typename TTypes<T, 2>::Tensor output) override {
    output.setConstant(std::numeric_limits<T>::lowest());
    if (data_size == 0) {
      return;
    }
    const int64 N = segment_ids.dimension(0);
    auto data_flat = typename TTypes<T, 2>::ConstTensor(data, N, data_size / N);
    for (int64 i = 0; i < N; ++i) {
      Index j = internal::SubtleMustCopy(segment_ids(i));
      OP_REQUIRES(ctx, FastBoundsCheck(j, output_rows),
                  errors::InvalidArgument(
                      "segment_ids", SliceDebugString(segment_ids_shape, i),
                      " = ", j, " is out of range [0, ", output_rows, ")"));
      output.template chip<0>(j) =
          data_flat.template chip<0>(i).cwiseMax(output.template chip<0>(j));
    }
  }
};
}  // namespace functor

// Base class for SegmentReductionOps that can handle unsorted segment
// definitions
// and specifying the size of the output in addition to a reduction function
template <typename Device, class T, class Index>
class UnsortedSegmentBaseOp : public OpKernel {
 public:
  explicit UnsortedSegmentBaseOp(
      OpKernelConstruction* context,
      functor::UnsortedSegmentBaseFunctor<Device, T, Index>& functor)
      : OpKernel(context), reduction_functor_(functor) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& data = context->input(0);
    const Tensor& segment_ids = context->input(1);
    const Tensor& num_segments = context->input(2);

    OP_REQUIRES(
        context, IsLegacyScalar(num_segments.shape()),
        errors::InvalidArgument("num_segments should be a scalar, not shape ",
                                num_segments.shape().DebugString()));
    OP_REQUIRES(
        context,
        TensorShapeUtils::StartsWith(data.shape(), segment_ids.shape()),
        errors::InvalidArgument("data.shape = ", data.shape().DebugString(),
                                " does not start with segment_ids.shape = ",
                                segment_ids.shape().DebugString()));

    const auto segment_flat = segment_ids.flat<Index>();
    const Index output_rows =
        internal::SubtleMustCopy(num_segments.scalar<int32>()());
    OP_REQUIRES(context, output_rows >= 0,
                errors::InvalidArgument("Input num_segments == ", output_rows,
                                        " must not be negative."));

    TensorShape output_shape;
    output_shape.AddDim(output_rows);
    for (int i = segment_ids.dims(); i < data.dims(); i++) {
      output_shape.AddDim(data.dim_size(i));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    auto output_flat = output->flat_outer_dims<T>();

    auto data_ptr = data.template flat<T>().data();
    reduction_functor_(context, context->template eigen_device<Device>(),
                     output_rows, segment_ids.shape(), segment_flat,
                     data.NumElements(), data_ptr, output_flat);
  }
 private:
  functor::UnsortedSegmentBaseFunctor<Device, T, Index>& reduction_functor_;
};

template <typename Device, class T, class Index>
class UnsortedSegmentSumOp : public UnsortedSegmentBaseOp<Device, T, Index> {
 public:
  explicit UnsortedSegmentSumOp(OpKernelConstruction* context)
      : UnsortedSegmentBaseOp<Device, T, Index>(
            context,
            sum_functor_) {}
 private:
    functor::UnsortedSegmentSumFunctor<Device, T, Index> sum_functor_;
};

template <typename Device, class T, class Index>
class UnsortedSegmentMaxOp : public UnsortedSegmentBaseOp<Device, T, Index> {
 public:
  explicit UnsortedSegmentMaxOp(OpKernelConstruction* context)
      : UnsortedSegmentBaseOp<Device, T, Index>(
            context,
            max_functor_) {}
 private:
    functor::UnsortedSegmentMaxFunctor<Device, T, Index> max_functor_;
};

#define REGISTER_REAL_CPU_UNSORTED_KERNELS(type, index_type)                  \
  REGISTER_KERNEL_BUILDER(Name("UnsortedSegmentSum")                          \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<type>("T")                      \
                              .TypeConstraint<index_type>("Tindices"),        \
                          UnsortedSegmentSumOp<CPUDevice, type, index_type>); \
  REGISTER_KERNEL_BUILDER(Name("UnsortedSegmentMax")                          \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<type>("T")                      \
                              .TypeConstraint<index_type>("Tindices"),        \
                          UnsortedSegmentMaxOp<CPUDevice, type, index_type>);

#define REGISTER_COMPLEX_CPU_UNSORTED_KERNELS(type, index_type)        \
  REGISTER_KERNEL_BUILDER(Name("UnsortedSegmentSum")                   \
                              .Device(DEVICE_CPU)                      \
                              .TypeConstraint<type>("T")               \
                              .TypeConstraint<index_type>("Tindices"), \
                          UnsortedSegmentSumOp<CPUDevice, type, index_type>);

#define REGISTER_REAL_CPU_UNSORTED_KERNELS_ALL(type) \
  REGISTER_REAL_CPU_UNSORTED_KERNELS(type, int32);   \
  REGISTER_REAL_CPU_UNSORTED_KERNELS(type, int64)

#define REGISTER_COMPLEX_CPU_UNSORTED_KERNELS_ALL(type) \
  REGISTER_COMPLEX_CPU_UNSORTED_KERNELS(type, int32);   \
  REGISTER_COMPLEX_CPU_UNSORTED_KERNELS(type, int64)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_REAL_CPU_UNSORTED_KERNELS_ALL);
REGISTER_COMPLEX_CPU_UNSORTED_KERNELS_ALL(complex64);
REGISTER_COMPLEX_CPU_UNSORTED_KERNELS_ALL(complex128);
#undef REGISTER_REAL_CPU_UNSORTED_KERNELS
#undef REGISTER_COMPLEX_CPU_UNSORTED_KERNELS
#undef REGISTER_COMPLEX_CPU_UNSORTED_KERNELS_ALL
#undef REGISTER_REAL_CPU_UNSORTED_KERNELS_ALL

#if GOOGLE_CUDA
#define REGISTER_GPU_UNSORTED_KERNELS(type, index_type)                \
  REGISTER_KERNEL_BUILDER(Name("UnsortedSegmentSum")                   \
                              .Device(DEVICE_GPU)                      \
                              .HostMemory("num_segments")              \
                              .TypeConstraint<type>("T")               \
                              .TypeConstraint<index_type>("Tindices"), \
                          UnsortedSegmentSumOp<GPUDevice, type, index_type>);

#define REGISTER_GPU_UNSORTED_KERNELS_ALL(type) \
  REGISTER_GPU_UNSORTED_KERNELS(type, int32);   \
  REGISTER_GPU_UNSORTED_KERNELS(type, int64);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_UNSORTED_KERNELS_ALL);
#undef REGISTER_GPU_UNSORTED_KERNELS
#undef REGISTER_GPU_UNSORTED_KERNELS_ALL
#endif  // GOOGLE_CUDA

// Same as SegmentReductionOp but takes as input a "sparse" tensor, represented
// by two dense tensors, one containing the data, and the other containing
// indices into the data.
template <typename Device, class T>
class SparseSegmentReductionOpBase : public OpKernel {
 public:
  explicit SparseSegmentReductionOpBase(OpKernelConstruction* context,
                                        bool is_mean, bool is_sqrtn)
      : OpKernel(context), is_mean_(is_mean), is_sqrtn_(is_sqrtn) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& segment_ids = context->input(2);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices should be a vector."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(segment_ids.shape()),
                errors::InvalidArgument("segment_ids should be a vector."));

    const int64 num_indices = indices.NumElements();
    OP_REQUIRES(context, num_indices == segment_ids.NumElements(),
                errors::InvalidArgument(
                    "segment_ids and indices should have same size."));

    auto input_flat = input.flat_outer_dims<T>();
    const auto indices_vec = indices.vec<Index>();
    typedef int32 OutputRow;
    const auto segment_vec = segment_ids.vec<OutputRow>();
    // Note that the current implementation assumes that segment_vec values are
    // sorted.
    const OutputRow output_rows =
        num_indices > 0
            ? internal::SubtleMustCopy(segment_vec(num_indices - 1)) + 1
            : 0;
    OP_REQUIRES(context, output_rows >= 0,
                errors::InvalidArgument("segment ids must be >= 0"));

    TensorShape output_shape = input.shape();
    output_shape.set_dim(0, output_rows);

    // Note that we do not initialize the output buffer with a default value.
    // We require that segment ids be sorted and cover all values (otherwise we
    // return an error).
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    if (num_indices == 0) return;
    OP_REQUIRES(context, output_rows > 0,
                errors::InvalidArgument("segment ids must be >= 0"));
    auto output_flat = output->flat_outer_dims<T>();

    int64 start = 0, end = 1;
    OutputRow out_index = internal::SubtleMustCopy(segment_vec(start));
    OP_REQUIRES(context, out_index == 0,
                errors::InvalidArgument("segment ids do not start at 0"));

    while (true) {
      // We initialize next_index to 0 to avoid "warning: 'next_index' may be
      // used uninitialized in this function" in the Mac build (since the
      // compiler isn't smart enough to realize the code is safe).
      OutputRow next_index = 0;
      if (end < num_indices) {
        next_index = internal::SubtleMustCopy(segment_vec(end));
        if (out_index == next_index) {
          ++end;
          continue;
        }
        // We have a new segment here.  Verify that the segment ids grow by one
        // each time, so that we cover every possible output value.
        OP_REQUIRES(
            context, out_index + 1 == next_index,
            errors::InvalidArgument("segment ids are not increasing by 1"));
      }

      OP_REQUIRES(
          context, FastBoundsCheck(out_index, output_rows),
          errors::InvalidArgument(
              "Segment id ", out_index, " out of range [0, ", output_rows,
              "), probably because 'segment_ids' input is not sorted."));
      auto out = output_flat.template chip<0>(out_index);
      const int bad_offset =
          Reduce(input_flat, indices_vec, start, end - start, out);
      OP_REQUIRES(context, bad_offset < 0,
                  errors::InvalidArgument(
                      "Bad: indices[", start + bad_offset, "] == ",
                      indices_vec(start + bad_offset), " out of range [0, ",
                      input_flat.dimension(0), ")"));

      if (end >= num_indices) break;
      start = end;
      ++end;
      out_index = next_index;
    }
  }

 private:
  typedef int32 Index;

  int64 Reduce(const typename TTypes<T>::ConstMatrix& input_flat,
               const typename TTypes<Index>::ConstVec& indices_vec, int64 start,
               int64 num,
               Eigen::TensorChippingOp<0, typename TTypes<T>::Matrix> out) {
#define INDEX(n, i)                               \
  const auto index##n = indices_vec(start + (i)); \
  if (!FastBoundsCheck(index##n, input_flat.dimension(0))) return (i);

#define L(n) input_flat.template chip<0>(index##n)

    if (num == 1) {
      INDEX(0, 0);
      out = L(0);
    } else {
      int64 r = num % 8;
      T m(1);
      if (is_mean_ && (num < 10)) {
        m = T(num);
      }
      if (is_sqrtn_ && (num < 10)) {
        m = T(sqrt(num));
      }
      switch (r) {
        case 2: {
          INDEX(0, 0);
          INDEX(1, 1);
          out = (L(0) + L(1)) / m;
          break;
        }
        case 3: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          out = (L(0) + L(1) + L(2)) / m;
          break;
        }
        case 4: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          out = (L(0) + L(1) + L(2) + L(3)) / m;
          break;
        }
        case 5: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          INDEX(4, 4);
          out = (L(0) + L(1) + L(2) + L(3) + L(4)) / m;
          break;
        }
        case 6: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          INDEX(4, 4);
          INDEX(5, 5);
          out = (L(0) + L(1) + L(2) + L(3) + L(4) + L(5)) / m;
          break;
        }
        case 7: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          INDEX(4, 4);
          INDEX(5, 5);
          INDEX(6, 6);
          out = (L(0) + L(1) + L(2) + L(3) + L(4) + L(5) + L(6)) / m;
          break;
        }
        case 0: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          INDEX(4, 4);
          INDEX(5, 5);
          INDEX(6, 6);
          INDEX(7, 7);
          out = (L(0) + L(1) + L(2) + L(3) + L(4) + L(5) + L(6) + L(7)) / m;
          r = 8;
          break;
        }
        case 1: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          INDEX(4, 4);
          INDEX(5, 5);
          INDEX(6, 6);
          INDEX(7, 7);
          INDEX(8, 8);
          out = (L(0) + L(1) + L(2) + L(3) + L(4) + L(5) + L(6) + L(7) + L(8)) /
                m;
          r = 9;
          break;
        }
      }
      for (; r < num; r += 8) {
        INDEX(0, r);
        INDEX(1, r + 1);
        INDEX(2, r + 2);
        INDEX(3, r + 3);
        INDEX(4, r + 4);
        INDEX(5, r + 5);
        INDEX(6, r + 6);
        INDEX(7, r + 7);
        out += L(0) + L(1) + L(2) + L(3) + L(4) + L(5) + L(6) + L(7);
      }
      if (is_mean_ && num >= 10) {
        out = out / static_cast<T>(num);
      }
      if (is_sqrtn_ && num >= 10) {
        out = out / static_cast<T>(sqrt(num));
      }
    }

    return -1;
#undef L
#undef INDEX
  }

  const bool is_mean_;
  const bool is_sqrtn_;
};

template <typename Device, class T>
class SparseSegmentReductionMeanOp
    : public SparseSegmentReductionOpBase<Device, T> {
 public:
  explicit SparseSegmentReductionMeanOp(OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T>(context, true /*is_mean*/,
                                                false /*is_sqrtn*/) {}
};

template <typename Device, class T>
class SparseSegmentReductionSqrtNOp
    : public SparseSegmentReductionOpBase<Device, T> {
 public:
  explicit SparseSegmentReductionSqrtNOp(OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T>(context, false /*is_mean*/,
                                                true /*is_sqrtn*/) {}
};

template <typename Device, class T>
class SparseSegmentReductionSumOp
    : public SparseSegmentReductionOpBase<Device, T> {
 public:
  explicit SparseSegmentReductionSumOp(OpKernelConstruction* context)
      : SparseSegmentReductionOpBase<Device, T>(context, false /*is_mean*/,
                                                false /*is_sqrtn*/) {}
};

#define REGISTER_CPU_SPARSE_KERNELS(type)                     \
  REGISTER_KERNEL_BUILDER(Name("SparseSegmentSum")            \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<type>("T")      \
                              .TypeConstraint<int32>("Tidx"), \
                          SparseSegmentReductionSumOp<CPUDevice, type>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_SPARSE_KERNELS);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type)                     \
  REGISTER_KERNEL_BUILDER(Name("SparseSegmentMean")           \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<type>("T")      \
                              .TypeConstraint<int32>("Tidx"), \
                          SparseSegmentReductionMeanOp<CPUDevice, type>);
REGISTER_CPU_SPARSE_KERNELS(float);
REGISTER_CPU_SPARSE_KERNELS(double);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type)                     \
  REGISTER_KERNEL_BUILDER(Name("SparseSegmentSqrtN")          \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<type>("T")      \
                              .TypeConstraint<int32>("Tidx"), \
                          SparseSegmentReductionSqrtNOp<CPUDevice, type>);
REGISTER_CPU_SPARSE_KERNELS(float);
REGISTER_CPU_SPARSE_KERNELS(double);
#undef REGISTER_CPU_SPARSE_KERNELS

template <class T>
class SparseSegmentGradOpBase : public OpKernel {
 public:
  explicit SparseSegmentGradOpBase(OpKernelConstruction* context, bool is_sqrtn)
      : OpKernel(context), is_sqrtn_(is_sqrtn) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& segment_ids = context->input(2);
    const Tensor& output_dim0 = context->input(3);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices should be a vector."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(segment_ids.shape()),
                errors::InvalidArgument("segment_ids should be a vector."));
    OP_REQUIRES(context, IsLegacyScalar(output_dim0.shape()),
                errors::InvalidArgument("output_dim0 should be a scalar."));

    const int64 N = indices.NumElements();
    OP_REQUIRES(context, N == segment_ids.NumElements(),
                errors::InvalidArgument(
                    "segment_ids and indices should have same size."));
    typedef int32 SegmentId;
    const SegmentId M =
        internal::SubtleMustCopy(output_dim0.scalar<SegmentId>()());

    auto input_flat = input.flat_outer_dims<T>();
    typedef int32 Index;
    const auto indices_vec = indices.vec<Index>();
    const auto segment_vec = segment_ids.vec<SegmentId>();

    TensorShape output_shape = input.shape();
    output_shape.set_dim(0, M);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    if (M == 0 || N == 0) return;

    // Note that similar to SparseSegmentMean, we assume that segment_vec is
    // already sorted and has non-negative values.
    const SegmentId num_segments =
        internal::SubtleMustCopy(segment_vec(N - 1)) + 1;
    OP_REQUIRES(context, input.dim_size(0) == num_segments,
                errors::InvalidArgument("Invalid number of segments"));

    // Compute scaling factors for input.
    std::vector<double> scaling(num_segments, 0.0);
    for (int64 i = 0; i < N; ++i) {
      const SegmentId idx = internal::SubtleMustCopy(segment_vec(i));
      OP_REQUIRES(
          context, FastBoundsCheck(idx, num_segments),
          errors::InvalidArgument("Segment id ", idx, " out of range [0, ",
                                  num_segments, ")."));
      scaling[idx] += 1;
    }
    for (size_t i = 0; i < scaling.size(); ++i) {
      if (is_sqrtn_) {
        scaling[i] = 1.0 / sqrt(std::max(scaling[i], 1.0));
      } else {
        scaling[i] = 1.0 / std::max(scaling[i], 1.0);
      }
    }

    auto output_flat = output->flat_outer_dims<T>();
    output_flat.setZero();
    std::vector<bool> is_modified(M, false);

    for (int64 i = 0; i < N; ++i) {
      const Index output_idx = internal::SubtleMustCopy(indices_vec(i));
      OP_REQUIRES(context, FastBoundsCheck(output_idx, M),
                  errors::InvalidArgument("Index ", output_idx,
                                          " out of range [0, ", M, ")."));

      const SegmentId idx = internal::SubtleMustCopy(segment_vec(i));
      OP_REQUIRES(
          context, FastBoundsCheck(idx, num_segments),
          errors::InvalidArgument("Segment id ", idx, " out of range [0, ",
                                  num_segments, ")."));

      const T scale = static_cast<T>(scaling[idx]);
      if (is_modified[output_idx]) {
        if (scale == 1.0) {
          output_flat.template chip<0>(output_idx) +=
              input_flat.template chip<0>(idx);
        } else {
          output_flat.template chip<0>(output_idx) +=
              input_flat.template chip<0>(idx) * scale;
        }
      } else {
        if (scale == 1.0) {
          output_flat.template chip<0>(output_idx) =
              input_flat.template chip<0>(idx);
        } else {
          output_flat.template chip<0>(output_idx) =
              input_flat.template chip<0>(idx) * scale;
        }
      }
      is_modified[output_idx] = true;
    }
  }

 private:
  const bool is_sqrtn_;
};

template <class T>
class SparseSegmentMeanGradOp : public SparseSegmentGradOpBase<T> {
 public:
  explicit SparseSegmentMeanGradOp(OpKernelConstruction* context)
      : SparseSegmentGradOpBase<T>(context, false /*is_sqrtn*/) {}
};

template <class T>
class SparseSegmentSqrtNGradOp : public SparseSegmentGradOpBase<T> {
 public:
  explicit SparseSegmentSqrtNGradOp(OpKernelConstruction* context)
      : SparseSegmentGradOpBase<T>(context, true /*is_sqrtn*/) {}
};

#define REGISTER_CPU_SPARSE_KERNELS(type)                     \
  REGISTER_KERNEL_BUILDER(Name("SparseSegmentMeanGrad")       \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<type>("T")      \
                              .TypeConstraint<int32>("Tidx"), \
                          SparseSegmentMeanGradOp<type>);
REGISTER_CPU_SPARSE_KERNELS(float);
REGISTER_CPU_SPARSE_KERNELS(double);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type)                     \
  REGISTER_KERNEL_BUILDER(Name("SparseSegmentSqrtNGrad")      \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<type>("T")      \
                              .TypeConstraint<int32>("Tidx"), \
                          SparseSegmentSqrtNGradOp<type>);
REGISTER_CPU_SPARSE_KERNELS(float);
REGISTER_CPU_SPARSE_KERNELS(double);
#undef REGISTER_CPU_SPARSE_KERNELS
}  // namespace tensorflow
