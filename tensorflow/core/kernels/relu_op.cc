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

// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/relu_op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif // TENSORFLOW_USE_SYCL

#define REGISTER_RELU_KERNELS(type)                                   \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Relu").Device(DEVICE_CPU).TypeConstraint<type>("T"),      \
      ReluOp<CPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("ReluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"),  \
      ReluGradOp<CPUDevice, type>);                                   \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Relu6").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      Relu6Op<CPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Relu6Grad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      Relu6GradOp<CPUDevice, type>)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_RELU_KERNELS);
#undef REGISTER_RELU_KERNELS

#define REGISTER_ELU_KERNELS(type)                                  \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Elu").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      EluOp<CPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("EluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      EluGradOp<CPUDevice, type>)

// Elu only makes sense with float or double.
TF_CALL_GPU_NUMBER_TYPES(REGISTER_ELU_KERNELS);
#undef REGISTER_ELU_KERNELS

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                    \
  template <>                                                                  \
  void Relu<GPUDevice, T>::operator()(                                         \
      const GPUDevice& d, typename TTypes<T>::ConstTensor features,            \
      typename TTypes<T>::Tensor activations);                                 \
  extern template struct Relu<GPUDevice, T>;                                   \
                                                                               \
  template <>                                                                  \
  void ReluGrad<GPUDevice, T>::operator()(                                     \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients,           \
      typename TTypes<T>::ConstTensor features,                                \
      typename TTypes<T>::Tensor backprops);                                   \
  extern template struct ReluGrad<GPUDevice, T>;                               \
                                                                               \
  template <>                                                                  \
  void Relu6<GPUDevice, T>::operator()(                                        \
      const GPUDevice& d, typename TTypes<T>::ConstTensor features,            \
      typename TTypes<T>::Tensor activations);                                 \
  extern template struct Relu6<GPUDevice, T>;                                  \
                                                                               \
  template <>                                                                  \
  void Relu6Grad<GPUDevice, T>::operator()(                                    \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients,           \
      typename TTypes<T>::ConstTensor features,                                \
      typename TTypes<T>::Tensor backprops);                                   \
  extern template struct Relu6Grad<GPUDevice, T>;                              \
                                                                               \
  template <>                                                                  \
  void Elu<GPUDevice, T>::operator()(const GPUDevice& d,                       \
                                     typename TTypes<T>::ConstTensor features, \
                                     typename TTypes<T>::Tensor activations);  \
  extern template struct Elu<GPUDevice, T>;                                    \
                                                                               \
  template <>                                                                  \
  void EluGrad<GPUDevice, T>::operator()(                                      \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients,           \
      typename TTypes<T>::ConstTensor activations,                             \
      typename TTypes<T>::Tensor backprops);                                   \
  extern template struct EluGrad<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNELS(type)                                    \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Relu").Device(DEVICE_GPU).TypeConstraint<type>("T"),      \
      ReluOp<GPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("ReluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),  \
      ReluGradOp<GPUDevice, type>);                                   \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Relu6").Device(DEVICE_GPU).TypeConstraint<type>("T"),     \
      Relu6Op<GPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Relu6Grad").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      Relu6GradOp<GPUDevice, type>);                                  \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Elu").Device(DEVICE_GPU).TypeConstraint<type>("T"),       \
      EluOp<GPUDevice, type>);                                        \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("EluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),   \
      EluGradOp<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS

#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL
// Registration of the GPU implementations.
#define REGISTER_SYCL_KERNELS(type)                                    \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("Relu").Device(DEVICE_SYCL).TypeConstraint<type>("T"),      \
      ReluOp<SYCLDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("ReluGrad").Device(DEVICE_SYCL).TypeConstraint<type>("T"),  \
      ReluGradOp<SYCLDevice, type>);                                   \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("Relu6").Device(DEVICE_SYCL).TypeConstraint<type>("T"),     \
      Relu6Op<SYCLDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("Relu6Grad").Device(DEVICE_SYCL).TypeConstraint<type>("T"), \
      Relu6GradOp<SYCLDevice, type>);                                  \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("Elu").Device(DEVICE_SYCL).TypeConstraint<type>("T"),       \
      EluOp<SYCLDevice, type>);                                        \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("EluGrad").Device(DEVICE_SYCL).TypeConstraint<type>("T"),   \
      EluGradOp<SYCLDevice, type>)

REGISTER_SYCL_KERNELS(float);
#undef REGISTER_SYCL_KERNELS
#endif // TENSORFLOW_USE_SYCL

}  // namespace tensorflow
