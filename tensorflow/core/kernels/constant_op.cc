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

// See docs in ../ops/array_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/constant_op.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

ConstantOp::ConstantOp(OpKernelConstruction* ctx)
    : OpKernel(ctx), tensor_(ctx->output_type(0)) {
  const TensorProto* proto = nullptr;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("value", &proto));
  OP_REQUIRES_OK(ctx, ctx->device()->MakeTensorFromProto(
                          *proto, AllocatorAttributes(), &tensor_));
  OP_REQUIRES(
      ctx, ctx->output_type(0) == tensor_.dtype(),
      errors::InvalidArgument("Type mismatch between value (",
                              DataTypeString(tensor_.dtype()), ") and dtype (",
                              DataTypeString(ctx->output_type(0)), ")"));
}

void ConstantOp::Compute(OpKernelContext* ctx) { ctx->set_output(0, tensor_); }

ConstantOp::~ConstantOp() {}

REGISTER_KERNEL_BUILDER(Name("Const").Device(DEVICE_CPU), ConstantOp);

#if TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(TYPE)                                     \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("Const").Device(DEVICE_SYCL).TypeConstraint<TYPE>("dtype"), \
      ConstantOp);
TF_CALL_NUMBER_TYPES(REGISTER_SYCL_KERNEL);
#undef REGISTER_SYCL_KERNEL
#endif

#if GOOGLE_CUDA
#define REGISTER_KERNEL(D, TYPE)                                      \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Const").Device(DEVICE_##D).TypeConstraint<TYPE>("dtype"), \
      ConstantOp);
REGISTER_KERNEL(GPU, Eigen::half);
REGISTER_KERNEL(GPU, bfloat16);
REGISTER_KERNEL(GPU, float);
REGISTER_KERNEL(GPU, double);
REGISTER_KERNEL(GPU, uint8);
REGISTER_KERNEL(GPU, int8);
REGISTER_KERNEL(GPU, uint16);
REGISTER_KERNEL(GPU, int16);
REGISTER_KERNEL(GPU, int64);
REGISTER_KERNEL(GPU, complex64);
REGISTER_KERNEL(GPU, complex128);
REGISTER_KERNEL(GPU, bool);
// Currently we do not support string constants on GPU
#undef REGISTER_KERNEL
#endif

HostConstantOp::HostConstantOp(OpKernelConstruction* ctx)
    : OpKernel(ctx), tensor_(ctx->output_type(0)) {
  const TensorProto* proto = nullptr;
  AllocatorAttributes alloc_attr;
  alloc_attr.set_on_host(true);
  OP_REQUIRES_OK(ctx, ctx->GetAttr("value", &proto));
  OP_REQUIRES_OK(
      ctx, ctx->device()->MakeTensorFromProto(*proto, alloc_attr, &tensor_));
  OP_REQUIRES(
      ctx, ctx->output_type(0) == tensor_.dtype(),
      errors::InvalidArgument("Type mismatch between value (",
                              DataTypeString(tensor_.dtype()), ") and dtype (",
                              DataTypeString(ctx->output_type(0)), ")"));
}

void HostConstantOp::Compute(OpKernelContext* ctx) {
  ctx->set_output(0, tensor_);
}

#if GOOGLE_CUDA
// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Const")
                            .Device(DEVICE_GPU)
                            .HostMemory("output")
                            .TypeConstraint<int32>("dtype"),
                        HostConstantOp);
#endif

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif //TENSORFLOW_USE_SYCL

namespace functor {

// Partial specialization of FillFunctor<Device=CPUDevice, T>.
template <typename T>
struct FillFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstScalar in) {
    out.device(d) = out.constant(in());
  }
};

#ifdef TENSORFLOW_USE_SYCL
// Partial specialization of FillFunctor<Device=SYCLDevice, T>.
template <typename T>
struct FillFunctor<SYCLDevice, T> {
  void operator()(const SYCLDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstScalar in) {
    To32Bit(out).device(d) = To32Bit(out).constant(in());
  }
};
#endif // TENSORFLOW_USE_SYCL

}  // end namespace functor

template <typename Device, typename T>
class FillOp : public OpKernel {
 public:
  explicit FillOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& Tdims = context->input(0);
    OP_REQUIRES(
        context, IsLegacyVector(Tdims.shape()),
        errors::InvalidArgument("dims must be a vector of int32, got shape ",
                                Tdims.shape().DebugString()));
    const Tensor& Tvalue = context->input(1);
    OP_REQUIRES(context, IsLegacyScalar(Tvalue.shape()),
                errors::InvalidArgument("value must be a scalar, got shape ",
                                        Tvalue.shape().DebugString()));
    auto dims = Tdims.flat<int32>();
    TensorShape shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                reinterpret_cast<const int32*>(dims.data()),
                                dims.size(), &shape));
    Tensor* out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &out));
    functor::FillFunctor<Device, T> functor;
    functor(context->eigen_device<Device>(), out->flat<T>(),
            Tvalue.scalar<T>());
  }
};

#define REGISTER_KERNEL(D, TYPE)                         \
  REGISTER_KERNEL_BUILDER(Name("Fill")                   \
                              .Device(DEVICE_##D)        \
                              .TypeConstraint<TYPE>("T") \
                              .HostMemory("dims"),       \
                          FillOp<D##Device, TYPE>);

#define REGISTER_CPU_KERNEL(TYPE) REGISTER_KERNEL(CPU, TYPE)
TF_CALL_ALL_TYPES(REGISTER_CPU_KERNEL);
// TODO(b/28917570): Add a test for this. Currently python 3 is not happy about
// the conversion from uint8 to quint8.
REGISTER_KERNEL(CPU, quint8);
#undef REGISTER_CPU_KERNEL

#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL(SYCL, float)
REGISTER_KERNEL_BUILDER(Name("Fill")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int32>("T")
                            .HostMemory("dims")
                            .HostMemory("value")
                            .HostMemory("output"),
                        FillOp<CPUDevice, int32>);
#endif // TENSORFLOW_USE_SYCL

#if GOOGLE_CUDA
REGISTER_KERNEL(GPU, Eigen::half);
REGISTER_KERNEL(GPU, float);
REGISTER_KERNEL(GPU, double);
REGISTER_KERNEL(GPU, uint8);
REGISTER_KERNEL(GPU, int8);
REGISTER_KERNEL(GPU, uint16);
REGISTER_KERNEL(GPU, int16);
REGISTER_KERNEL(GPU, int64);
// Currently we do not support filling strings and complex64 on GPU

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Fill")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("dims")
                            .HostMemory("value")
                            .HostMemory("output"),
                        FillOp<CPUDevice, int32>);
#endif

#undef REGISTER_KERNEL

template <typename Device, typename T>
class ZerosLikeOp : public OpKernel {
 public:
  explicit ZerosLikeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &out));
    functor::SetZeroFunctor<Device, T> f;
    f(ctx->eigen_device<Device>(), out->flat<T>());
  }
};

#define REGISTER_KERNEL(type, dev)                                      \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("ZerosLike").Device(DEVICE_##dev).TypeConstraint<type>("T"), \
      ZerosLikeOp<dev##Device, type>)

#define REGISTER_CPU(type) REGISTER_KERNEL(type, CPU)
TF_CALL_ALL_TYPES(REGISTER_CPU);
#undef REGISTER_CPU

#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL(float, SYCL);
REGISTER_KERNEL_BUILDER(Name("ZerosLike")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int32>("T")
                            .HostMemory("y"),
                        ZerosLikeOp<CPUDevice, int32>);
#endif  // TENSORFLOW_USE_SYCL

#if GOOGLE_CUDA
REGISTER_KERNEL(bool, GPU);
REGISTER_KERNEL(Eigen::half, GPU);
REGISTER_KERNEL(float, GPU);
REGISTER_KERNEL(double, GPU);
REGISTER_KERNEL(complex64, GPU);
REGISTER_KERNEL(complex128, GPU);
REGISTER_KERNEL(int64, GPU);
REGISTER_KERNEL_BUILDER(Name("ZerosLike")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("y"),
                        ZerosLikeOp<CPUDevice, int32>);
#endif  // GOOGLE_CUDA

#undef REGISTER_KERNEL

class PlaceholderOp : public OpKernel {
 public:
  explicit PlaceholderOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &expected_shape_));
  }

  void Compute(OpKernelContext* ctx) override {
    if (expected_shape_.dims() > 0) {
      OP_REQUIRES(ctx, false,
                  errors::InvalidArgument(
                      "You must feed a value for placeholder tensor '", name(),
                      "' with dtype ", DataTypeString(output_type(0)),
                      " and shape ", expected_shape_.DebugString()));
    } else {
      OP_REQUIRES(ctx, false,
                  errors::InvalidArgument(
                      "You must feed a value for placeholder tensor '", name(),
                      "' with dtype ", DataTypeString(output_type(0))));
    }
  }

 private:
  TensorShape expected_shape_;
};

REGISTER_KERNEL_BUILDER(Name("Placeholder").Device(DEVICE_CPU), PlaceholderOp);
REGISTER_KERNEL_BUILDER(Name("PlaceholderV2").Device(DEVICE_CPU),
                        PlaceholderOp);
// The following GPU kernel registration is used to address the situation that
// a placeholder is added in a GPU device context and soft placement is false.
// Since a placeholder should never be executed, adding these GPU kernels has
// no effect on graph execution.
REGISTER_KERNEL_BUILDER(Name("Placeholder").Device(DEVICE_GPU), PlaceholderOp);
REGISTER_KERNEL_BUILDER(Name("PlaceholderV2").Device(DEVICE_GPU),
                        PlaceholderOp);

}  // namespace tensorflow
