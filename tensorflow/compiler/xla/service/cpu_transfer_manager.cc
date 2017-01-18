/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu_transfer_manager.h"

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/service/cpu/infeed_manager.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace se = ::perftools::gputools;

namespace xla {

namespace {

class CpuInfeedBuffer : public cpu::runtime::InfeedBuffer {
 public:
  explicit CpuInfeedBuffer(int32 length)
      : length_(length),
        buffer_(new char[length]),
        device_memory_(buffer_, length_) {}
  ~CpuInfeedBuffer() override { delete[] buffer_; }

  int32 length() override { return length_; }
  void* data() override { return buffer_; }
  void Done() override { delete this; }

  se::DeviceMemoryBase* device_memory() { return &device_memory_; }

 private:
  int32 length_;
  char* buffer_;
  se::DeviceMemoryBase device_memory_;
};

}  // namespace

CpuTransferManager::CpuTransferManager()
    : GenericTransferManager(se::host::kHostPlatformId) {}

Status CpuTransferManager::TransferLiteralToInfeed(se::StreamExecutor* executor,
                                                   const Literal& literal) {
  const Shape& shape = literal.shape();
  VLOG(2) << "transferring literal shape to infeed: "
          << ShapeUtil::HumanString(shape);

  // TODO(b/31381668) handle tuples.
  if (ShapeUtil::IsTuple(shape)) {
    return Unimplemented("Infeed with a tuple shape is not supported: %s",
                         ShapeUtil::HumanString(literal.shape()).c_str());
  }

  cpu::runtime::InfeedManager* infeed_manager =
      cpu::runtime::GetInfeedManager();

  int64 size = GetByteSizeRequirement(shape);
  if (size > std::numeric_limits<int32>::max()) {
    return Unimplemented("Infeed shape is too large: %s needs %lld bytes",
                         ShapeUtil::HumanString(literal.shape()).c_str(), size);
  }
  int32 size_32 = static_cast<int32>(size);
  CpuInfeedBuffer* queued_buffer = new CpuInfeedBuffer(size_32);
  TF_RETURN_IF_ERROR(TransferBufferToDevice(
      executor, /*size=*/size, /*source=*/LiteralUtil::InternalData(literal),
      queued_buffer->device_memory()));

  infeed_manager->EnqueueBuffer(queued_buffer);

  return Status::OK();
}

}  // namespace xla

static xla::TransferManager* CreateCpuTransferManager() {
  return new xla::CpuTransferManager();
}

static bool InitModule() {
  xla::TransferManager::RegisterTransferManager(se::host::kHostPlatformId,
                                                &CreateCpuTransferManager);
  return true;
}
static bool module_initialized = InitModule();
