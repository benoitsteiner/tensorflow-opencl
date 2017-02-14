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

#include "tensorflow/core/kernels/shape_ops.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

// Shape ----------------------------------------
REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int32>("out_type"),
                        ShapeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int64>("out_type"),
                        ShapeOp<int64>);

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(type)                               \
  REGISTER_KERNEL_BUILDER(Name("Shape")                          \
                              .Device(DEVICE_SYCL)               \
                              .HostMemory("output")              \
                              .TypeConstraint<int32>("out_type") \
                              .TypeConstraint<type>("T"),        \
                          ShapeOp<int32>);                       \
  REGISTER_KERNEL_BUILDER(Name("Shape")                          \
                              .Device(DEVICE_SYCL)               \
                              .HostMemory("output")              \
                              .TypeConstraint<int64>("out_type") \
                              .TypeConstraint<type>("T"),        \
                          ShapeOp<int64>);

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_KERNEL);
#undef REGISTER_SYCL_KERNEL

REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_SYCL)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_type"),
                        ShapeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_SYCL)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64>("out_type"),
                        ShapeOp<int64>);
#endif  // TENSORFLOW_USE_SYCL

#if GOOGLE_CUDA
#define REGISTER_GPU_KERNEL(type)                                \
  REGISTER_KERNEL_BUILDER(Name("Shape")                          \
                              .Device(DEVICE_GPU)                \
                              .HostMemory("output")              \
                              .TypeConstraint<int32>("out_type") \
                              .TypeConstraint<type>("T"),        \
                          ShapeOp<int32>);                       \
  REGISTER_KERNEL_BUILDER(Name("Shape")                          \
                              .Device(DEVICE_GPU)                \
                              .HostMemory("output")              \
                              .TypeConstraint<int64>("out_type") \
                              .TypeConstraint<type>("T"),        \
                          ShapeOp<int64>);

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_type"),
                        ShapeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64>("out_type"),
                        ShapeOp<int64>);
#endif

// ShapeN ---------------------------------------
REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int32>("out_type"),
                        ShapeNOp<int32>);
REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int64>("out_type"),
                        ShapeNOp<int64>);

#if GOOGLE_CUDA
#define REGISTER_GPU_KERNEL(type)                                \
  REGISTER_KERNEL_BUILDER(Name("ShapeN")                         \
                              .Device(DEVICE_GPU)                \
                              .HostMemory("output")              \
                              .TypeConstraint<int32>("out_type") \
                              .TypeConstraint<type>("T"),        \
                          ShapeNOp<int32>);                      \
  REGISTER_KERNEL_BUILDER(Name("ShapeN")                         \
                              .Device(DEVICE_GPU)                \
                              .HostMemory("output")              \
                              .TypeConstraint<int64>("out_type") \
                              .TypeConstraint<type>("T"),        \
                          ShapeNOp<int64>)

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_type"),
                        ShapeNOp<int32>);
REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64>("out_type"),
                        ShapeNOp<int64>);
#endif

#if TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(type)                               \
  REGISTER_KERNEL_BUILDER(Name("ShapeN")                         \
                              .Device(DEVICE_SYCL)               \
                              .HostMemory("output")              \
                              .TypeConstraint<int32>("out_type") \
                              .TypeConstraint<type>("T"),        \
                          ShapeNOp<int32>);                      \
  REGISTER_KERNEL_BUILDER(Name("ShapeN")                         \
                              .Device(DEVICE_SYCL)               \
                              .HostMemory("output")              \
                              .TypeConstraint<int64>("out_type") \
                              .TypeConstraint<type>("T"),        \
                          ShapeNOp<int64>)

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_KERNEL);
#undef REGISTER_SYCL_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(DEVICE_SYCL)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_type"),
                        ShapeNOp<int32>);
REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(DEVICE_SYCL)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64>("out_type"),
                        ShapeNOp<int64>);
#endif  // TENSORFLOW_USE_SYCL

// Rank ------------------------------------------
REGISTER_KERNEL_BUILDER(Name("Rank").Device(DEVICE_CPU).HostMemory("output"),
                        RankOp);

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(type)                       \
  REGISTER_KERNEL_BUILDER(Name("Rank")                   \
                              .Device(DEVICE_SYCL)       \
                              .TypeConstraint<type>("T") \
                              .HostMemory("output"),     \
                          RankOp);
REGISTER_SYCL_KERNEL(float);
REGISTER_SYCL_KERNEL(double);
#undef REGISTER_SYCL_KERNEL

// A special GPU kernel for int32 and bool.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Rank")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("output"),
                        RankOp);

REGISTER_KERNEL_BUILDER(Name("Rank")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<bool>("T")
                            .HostMemory("input")
                            .HostMemory("output"),
                        RankOp);
#endif  // TENSORFLOW_USE_SYCL

#if GOOGLE_CUDA
#define REGISTER_GPU_KERNEL(type)                        \
  REGISTER_KERNEL_BUILDER(Name("Rank")                   \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("output"),     \
                          RankOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

// A special GPU kernel for int32 and bool.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Rank")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("output"),
                        RankOp);

REGISTER_KERNEL_BUILDER(Name("Rank")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<bool>("T")
                            .HostMemory("input")
                            .HostMemory("output"),
                        RankOp);
#endif

// Size ------------------------------------------
REGISTER_KERNEL_BUILDER(Name("Size")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int32>("out_type"),
                        SizeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Size")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int64>("out_type"),
                        SizeOp<int64>);

#if GOOGLE_CUDA
#define REGISTER_GPU_KERNEL(type)                                \
  REGISTER_KERNEL_BUILDER(Name("Size")                           \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("out_type") \
                              .HostMemory("output"),             \
                          SizeOp<int32>);                        \
  REGISTER_KERNEL_BUILDER(Name("Size")                           \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("out_type") \
                              .HostMemory("output"),             \
                          SizeOp<int64>);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Size")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_type")
                            .HostMemory("input")
                            .HostMemory("output"),
                        SizeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Size")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64>("out_type")
                            .HostMemory("input")
                            .HostMemory("output"),
                        SizeOp<int64>);
#endif

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(type)                               \
  REGISTER_KERNEL_BUILDER(Name("Size")                           \
                              .Device(DEVICE_SYCL)               \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("out_type") \
                              .HostMemory("output"),             \
                          SizeOp<int32>);                        \
  REGISTER_KERNEL_BUILDER(Name("Size")                           \
                              .Device(DEVICE_SYCL)               \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("out_type") \
                              .HostMemory("output"),             \
                          SizeOp<int64>);
REGISTER_SYCL_KERNEL(float);
REGISTER_SYCL_KERNEL(double);
#undef REGISTER_SYCL_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Size")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_type")
                            .HostMemory("input")
                            .HostMemory("output"),
                        SizeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Size")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64>("out_type")
                            .HostMemory("input")
                            .HostMemory("output"),
                        SizeOp<int64>);
#endif // TENSORFLOW_USE_SYCL

// ExpandDims ------------------------------------
REGISTER_KERNEL_BUILDER(Name("ExpandDims")
                            .Device(DEVICE_CPU)
                            .HostMemory("dim")
                            .TypeConstraint<int32>("Tdim"),
                        ExpandDimsOp);

#if GOOGLE_CUDA
#define REGISTER_GPU_KERNEL(type)                            \
  REGISTER_KERNEL_BUILDER(Name("ExpandDims")                 \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<type>("T")     \
                              .TypeConstraint<int32>("Tdim") \
                              .HostMemory("dim"),            \
                          ExpandDimsOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

REGISTER_KERNEL_BUILDER(Name("ExpandDims")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("Tdim")
                            .HostMemory("input")
                            .HostMemory("dim")
                            .HostMemory("output"),
                        ExpandDimsOp);
#endif // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(type)                           \
  REGISTER_KERNEL_BUILDER(Name("ExpandDims")                 \
                              .Device(DEVICE_SYCL)           \
                              .TypeConstraint<type>("T")     \
                              .TypeConstraint<int32>("Tdim") \
                              .HostMemory("dim"),            \
                          ExpandDimsOp);
REGISTER_SYCL_KERNEL(float)
REGISTER_SYCL_KERNEL(double)

#undef REGISTER_SYCL_KERNEL

REGISTER_KERNEL_BUILDER(Name("ExpandDims")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("Tdim")
                            .HostMemory("input")
                            .HostMemory("dim")
                            .HostMemory("output"),
                        ExpandDimsOp);
#endif // TENSORFLOW_USE_SYCL

// Squeeze ---------------------------------------
REGISTER_KERNEL_BUILDER(Name("Squeeze").Device(DEVICE_CPU), SqueezeOp);

#if GOOGLE_CUDA
#define REGISTER_GPU_KERNEL(type)                                   \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Squeeze").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SqueezeOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Squeeze")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("output"),
                        SqueezeOp);
#endif

#if TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(type)                                  \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Squeeze").Device(DEVICE_SYCL).TypeConstraint<type>("T"),\
      SqueezeOp);
REGISTER_SYCL_KERNEL(float);
REGISTER_SYCL_KERNEL(double);
#undef REGISTER_SYCL_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Squeeze")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("output"),
                        SqueezeOp);
#endif // TENSORFLOW_USE_SYCL

}  // namespace tensorflow
