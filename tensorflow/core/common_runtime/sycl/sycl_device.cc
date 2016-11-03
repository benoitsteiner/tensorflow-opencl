/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#if TENSORFLOW_USE_SYCL

#include "tensorflow/core/common_runtime/sycl/sycl_device.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/tensor.pb_text.h"
#include "tensorflow/core/platform/tracing.h"

namespace tensorflow {

cl::sycl::gpu_selector s;
cl::sycl::queue q(s);

SYCLDevice::SYCLDevice(const SessionOptions& options, const string& name,
                       Bytes memory_limit, const DeviceLocality& locality,
                       const string& physical_device_desc, Allocator* allocator)
    : LocalDevice(options,
                  Device::BuildDeviceAttributes(name, DEVICE_SYCL, memory_limit,
                                                locality, physical_device_desc),
                  allocator),
      allocator_(allocator),
      device_context_(new SYCLDeviceContext()),
      device_(q) {
  set_eigen_sycl_device(&device_);
}

SYCLDevice::~SYCLDevice() {
  device_context_->Unref();
}

void SYCLDevice::Compute(OpKernel* op_kernel, OpKernelContext* context) {
  assert(context);
  if (port::Tracing::IsActive()) {
    // TODO(pbar) We really need a useful identifier of the graph node.
    const uint64 id = Hash64(op_kernel->name());
    port::Tracing::ScopedActivity region(port::Tracing::EventCategory::kCompute,
                                         id);
  }
  op_kernel->Compute(context);
}

Allocator* SYCLDevice::GetAllocator(AllocatorAttributes attr) {
  return allocator_;
}

Status SYCLDevice::MakeTensorFromProto(const TensorProto& tensor_proto,
                                       const AllocatorAttributes alloc_attrs,
                                       Tensor* tensor) {
  Tensor parsed(tensor_proto.dtype());
  if (!parsed.FromProto(cpu_allocator(), tensor_proto)) {
    return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                   ProtoDebugString(tensor_proto));
  }
  *tensor = std::move(parsed);
  return Status::OK();
}

Status SYCLDevice::FillContextMap(const Graph* graph,
				  DeviceContextMap* device_context_map) {
  // Fill in the context map.  It is OK for this map to contain
  // duplicate DeviceContexts so long as we increment the refcount.
  device_context_map->resize(graph->num_node_ids());
  for (Node* n : graph->nodes()) {
    device_context_->Ref();
    (*device_context_map)[n->id()] = device_context_;
  }

  return Status::OK();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_SYCL
