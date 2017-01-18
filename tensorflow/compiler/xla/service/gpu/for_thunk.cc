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

#include "tensorflow/compiler/xla/service/gpu/for_thunk.h"

#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace gpu {

ForThunk::ForThunk(const int64 loop_limit,
                   std::unique_ptr<ThunkSequence> body_thunk_sequence,
                   const HloInstruction* hlo)
    : Thunk(Kind::kWhile, hlo),
      loop_limit_(loop_limit),
      body_thunk_sequence_(
          MakeUnique<SequentialThunk>(std::move(*body_thunk_sequence), hlo)) {}

tensorflow::Status ForThunk::Initialize(const GpuExecutable& executable) {
  TF_RETURN_IF_ERROR(body_thunk_sequence_->Initialize(executable));
  return tensorflow::Status::OK();
}

tensorflow::Status ForThunk::ExecuteOnStream(
    const BufferAllocations& buffer_allocations,
    perftools::gputools::Stream* stream) {
  for (int64 i = 0; i < loop_limit_; ++i) {
    // Invoke loop body thunk sequence.
    TF_RETURN_IF_ERROR(
        body_thunk_sequence_->ExecuteOnStream(buffer_allocations, stream));
  }
  return tensorflow::Status::OK();
}

}  // namespace gpu
}  // namespace xla
