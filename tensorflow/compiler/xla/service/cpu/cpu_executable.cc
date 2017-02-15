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

#include "tensorflow/compiler/xla/service/cpu/cpu_executable.h"

#include <stdint.h>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

#include "external/llvm/include/llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace se = ::perftools::gputools;

namespace xla {
namespace cpu {

CpuExecutable::CpuExecutable(
    std::unique_ptr<SimpleOrcJIT> jit,
    std::unique_ptr<BufferAssignment> assignment,
    std::unique_ptr<HloModule> hlo_module,
    std::unique_ptr<HloModuleConfig> module_config,
    const string& entry_function_name,
    std::unordered_map<const HloInstruction*, size_t> hlo_to_profile_idx)
    : Executable(std::move(hlo_module), std::move(module_config)),
      jit_(std::move(jit)),
      assignment_(std::move(assignment)),
      hlo_to_profile_idx_(std::move(hlo_to_profile_idx)) {
  // Resolve symbols in the constructor rather than at execution time to avoid
  // races because FindSymbol is not thread safe.
  llvm::JITSymbol sym = jit_->FindSymbol(entry_function_name);
  // We expect to find the symbol provided with entry_function_name; otherwise
  // this is an internal error.
  CHECK(sym) << "Symbol " << entry_function_name << " not found.";
  // getAddress can do work under the hood in the jit, so it needs to be
  // guarded by the mutex.
  compute_function_ = reinterpret_cast<ComputeFunctionType>(sym.getAddress());
}

// Given a pointer to an output buffer (following the CPU JIT calling
// conventions), mark addresses that are "live". The initial pointer itself is
// trivially live. If the shape of the buffer is a tuple, this analysis looks
// into the tuple's elements and marks them live as well (since tuples keep
// pointers to buffers) and also works recursively.  address is an in-memory
// buffer address that contains some runtime XLA object.  shape is its
// shape. marked_addresses is the set of live addresses to populate.
static void MarkLiveAddressesInOutput(
    const void* address, const Shape& shape,
    std::unordered_set<const void*>* marked_addresses) {
  marked_addresses->insert(address);
  const uintptr_t* address_buffer = static_cast<const uintptr_t*>(address);
  if (ShapeUtil::IsTuple(shape)) {
    for (int i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
      const uintptr_t* element_address = address_buffer + i;
      const void* element = reinterpret_cast<const void*>(*element_address);
      MarkLiveAddressesInOutput(
          element, ShapeUtil::GetTupleElementShape(shape, i), marked_addresses);
    }
  }
}

Status CpuExecutable::AllocateBuffers(
    DeviceMemoryAllocator* memory_allocator, int device_ordinal,
    std::vector<perftools::gputools::DeviceMemoryBase>* buffers) {
  CHECK_EQ(buffers->size(), assignment_->Allocations().size());
  VLOG(3) << "Allocating " << assignment_->Allocations().size()
          << " allocations for module " << module().name();
  for (BufferAllocation::Index i = 0; i < assignment_->Allocations().size();
       ++i) {
    auto& allocation = assignment_->GetAllocation(i);

    VLOG(3) << allocation.ToString();

    if (allocation.is_entry_computation_parameter()) {
      VLOG(3) << "allocation #" << i << " is a parameter";
      continue;
    }

    if (allocation.is_thread_local()) {
      VLOG(3) << "buffer #" << i << " is thread-local";
      continue;
    }

    int64 buffer_size = allocation.size();
    if (!(*buffers)[i].is_null()) {
      VLOG(3) << "buffer #" << i
              << " is in the preallocated result ShapedBuffer";
    } else {
      TF_ASSIGN_OR_RETURN((*buffers)[i], memory_allocator->Allocate(
                                             device_ordinal, buffer_size));

      VLOG(3) << "buffer #" << i << " allocated " << buffer_size << " bytes ["
              << (*buffers)[i].opaque() << "]";
    }

    // Since the output buffer and all the temporary buffers were written into
    // by the JITed code, msan has no way of knowing their memory was
    // initialized. Mark them initialized so that msan doesn't flag loads from
    // these buffers.
    TF_ANNOTATE_MEMORY_IS_INITIALIZED((*buffers)[i].opaque(), buffer_size);
  }

  TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice result_slice,
                      assignment_->GetUniqueTopLevelOutputSlice());
  VLOG(3) << "result index: " << result_slice.index();

  return Status::OK();
}

Status CpuExecutable::ExecuteComputeFunction(
    const ExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    tensorflow::gtl::ArraySlice<se::DeviceMemoryBase> buffers,
    HloExecutionProfile* hlo_execution_profile) {
  std::vector<se::DeviceMemoryBase> argument_buffers;
  for (int i = 0; i < arguments.size(); ++i) {
    TF_RET_CHECK(!ShapeUtil::IsTuple(arguments[i]->shape()));
    argument_buffers.push_back(arguments[i]->buffer(/*index=*/{}));
  }
  return ExecuteComputeFunction(run_options, argument_buffers, buffers,
                                hlo_execution_profile);
}

Status CpuExecutable::ExecuteComputeFunction(
    const ExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<se::DeviceMemoryBase> arguments,
    tensorflow::gtl::ArraySlice<se::DeviceMemoryBase> buffers,
    HloExecutionProfile* hlo_execution_profile) {
  // The calling convention for JITed functions is:
  //
  //  void function(void* result, const void* run_options, void** args_array,
  //                void** temps_array)
  //
  // result: Points at the result.
  // run_options: the ExecutableRunOptions object.
  // args_array: An array of pointers, each of which points to a parameter.
  //               The size of this array is determined by the function's arity
  //               (ProgramShape).
  // temps_array:  An array of pointers, each of which points to a temporary
  //               buffer the computation needs. The size of this array is
  //               determined by buffer analysis.
  //
  std::vector<const void*> args_array;
  for (se::DeviceMemoryBase arg_mem : arguments) {
    args_array.push_back(arg_mem.opaque());
  }

  uint64 start_micros = tensorflow::Env::Default()->NowMicros();

  // Allocate profiling counters for each hlo instruction that we would like to
  // profile.  Allocate an additional profile counter for the entire
  // computation.
  std::vector<uint64> profile_counters(hlo_to_profile_idx_.size() + 1);

  // Call the computation function following the calling convention.
  std::vector<void*> buffer_pointers;
  for (auto& buffer : buffers) {
    buffer_pointers.push_back(const_cast<void*>(buffer.opaque()));
  }
  TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice result_slice,
                      assignment_->GetUniqueTopLevelOutputSlice());
  void* result_buffer = buffer_pointers[result_slice.index()];
  if (VLOG_IS_ON(3)) {
    VLOG(3) << "Executing compute function:";
    VLOG(3) << tensorflow::strings::Printf(
        "  func(void* result, void* params[%zu], void* temps[%zu], "
        "uint64 profile_counters[%zu])",
        args_array.size(), buffer_pointers.size(), profile_counters.size());
    VLOG(3) << tensorflow::strings::Printf("    result = %p", result_buffer);
    auto ptr_printer = [](string* out, const void* p) {
      tensorflow::strings::StrAppend(out, tensorflow::strings::Printf("%p", p));
    };
    VLOG(3) << tensorflow::strings::Printf(
        "    params = [%s]",
        tensorflow::str_util::Join(args_array, ", ", ptr_printer).c_str());
    VLOG(3) << tensorflow::strings::Printf(
        "    temps = [%s]",
        tensorflow::str_util::Join(buffer_pointers, ", ", ptr_printer).c_str());
    VLOG(3) << tensorflow::strings::Printf("    profile_counters = %p",
                                           profile_counters.data());
  }

  compute_function_(result_buffer, run_options, args_array.data(),
                    buffer_pointers.data(), profile_counters.data());

  uint64 end_micros = tensorflow::Env::Default()->NowMicros();

  {
    tensorflow::mutex_lock lock(mutex_);
    const double nanoseconds = (end_micros - start_micros) * 1000.0;
    execution_profile_.set_compute_time_ns(std::max(nanoseconds, 1.0));

    // The last profile counter is used for the computation as a whole.
    execution_profile_.set_compute_cycle_count(profile_counters.back());
  }

  if (hlo_execution_profile != nullptr) {
    hlo_execution_profile->set_total_cycles_executed(profile_counters.back());

    for (auto hlo_prof_idx : hlo_to_profile_idx_) {
      const HloInstruction* hlo = hlo_prof_idx.first;
      uint64 cycles_taken = profile_counters[hlo_prof_idx.second];
      hlo_execution_profile->AddProfileResult(hlo, cycles_taken);
    }
  }
  return Status::OK();
}

StatusOr<perftools::gputools::DeviceMemoryBase> CpuExecutable::ExecuteOnStream(
    const ExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<se::DeviceMemoryBase> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  se::Stream* stream = run_options->stream();
  DeviceMemoryAllocator* memory_allocator = run_options->allocator();
  std::vector<se::DeviceMemoryBase> buffers(assignment_->Allocations().size());

  TF_RETURN_IF_ERROR(AllocateBuffers(
      memory_allocator, stream->parent()->device_ordinal(), &buffers));
  TF_RETURN_IF_ERROR(ExecuteComputeFunction(run_options, arguments, buffers,
                                            hlo_execution_profile));

  // Mark the buffers that are actually live (used in the output) when the
  // computation finishes executing.
  std::unordered_set<const void*> marked_addresses;
  TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice result_slice,
                      assignment_->GetUniqueTopLevelOutputSlice());
  se::DeviceMemoryBase top_level_output = buffers[result_slice.index()];
  MarkLiveAddressesInOutput(top_level_output.opaque(), result_shape(),
                            &marked_addresses);

  VLOG(3) << "Live addresses in output marking found "
          << marked_addresses.size() << " addresses:\n"
          << tensorflow::str_util::Join(
                 marked_addresses, ", ", [](string* out, const void* address) {
                   tensorflow::strings::StrAppend(
                       out, tensorflow::strings::Printf("%p", address));
                 });

  // Computation is done - deallocate temp buffers. Keep those marked live
  // because they are referenced by the output of the computation and are needed
  // by the service. They will be deallocated by the service.
  for (size_t i = 0; i < buffers.size(); ++i) {
    se::DeviceMemoryBase alloc = buffers[i];
    if (marked_addresses.count(alloc.opaque()) == 0 && !alloc.is_null()) {
      VLOG(3) << "CpuExecutable deallocating buffer #" << i << " ["
              << alloc.opaque() << "]";
      TF_RETURN_IF_ERROR(memory_allocator->Deallocate(
          stream->parent()->device_ordinal(), &alloc));
    }
  }

  return top_level_output;
}

StatusOr<std::unique_ptr<ShapedBuffer>> CpuExecutable::ExecuteOnStream(
    const ExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  if (GetRootPointsToSet().IsAmbiguous()) {
    return Unimplemented("Points-to set of root instruction is ambiguous");
  }

  se::Stream* stream = run_options->stream();
  DeviceMemoryAllocator* memory_allocator = run_options->allocator();
  std::vector<se::DeviceMemoryBase> buffers(assignment_->Allocations().size());

  TF_ASSIGN_OR_RETURN(std::unique_ptr<ShapedBuffer> result_buffer,
                      ShapedBuffer::MakeShapedBuffer(
                          result_shape(), stream->parent()->platform(),
                          stream->parent()->device_ordinal()));
  TF_RETURN_IF_ERROR(AllocateBuffers(
      memory_allocator, stream->parent()->device_ordinal(), &buffers));
  TF_RETURN_IF_ERROR(ExecuteComputeFunction(run_options, arguments, buffers,
                                            hlo_execution_profile));

  // Copy DeviceMemoryBase values which contain the array(s) of the result into
  // the respective location in ShapedBuffer which is returned to the caller.
  std::vector<bool> buffers_in_result(assignment_->Allocations().size(), false);
  TF_RETURN_IF_ERROR(
      result_buffer->mutable_shape_index_to_buffer_entry()
          ->ForEachMutableElement(
              [&buffers, &buffers_in_result, &result_buffer, this](
                  const ShapeIndex& index, bool is_leaf, size_t* buffer_entry) {
                if (is_leaf) {
                  const std::vector<const LogicalBuffer*>& sources =
                      this->GetRootPointsToSet().element(index);
                  // The points to set is unambiguous so the set should be a
                  // singleton.
                  CHECK_EQ(1, sources.size());
                  const LogicalBuffer* buffer_source = sources[0];
                  HloInstruction* src = buffer_source->instruction();

                  // The source for this result buffer can be a nested buffer
                  // such as a tuple element.

                  // The source instruction should have a non-parameter buffer
                  // assigned.
                  TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice slice,
                                      this->assignment_->GetUniqueSlice(
                                          src, buffer_source->index()));
                  CHECK(!slice.allocation()->is_entry_computation_parameter());

                  const BufferAllocation::Index buffer_index = slice.index();
                  const se::DeviceMemoryBase& buffer = buffers[buffer_index];
                  CHECK(!buffer.is_null() || buffer.size() == 0);
                  *buffer_entry = result_buffer->mutable_buffers()->size();
                  result_buffer->mutable_buffers()->push_back(buffer);
                  buffers_in_result[buffer_index] = true;
                }
                return Status::OK();
              }));

  // Free all buffers not in the result.
  for (size_t i = 0; i < buffers.size(); ++i) {
    se::DeviceMemoryBase alloc = buffers[i];
    if (!buffers_in_result[i] && !alloc.is_null()) {
      VLOG(3) << "CpuExecutable deallocating buffer #" << i << " ["
              << alloc.opaque() << "]";
      TF_RETURN_IF_ERROR(memory_allocator->Deallocate(
          stream->parent()->device_ordinal(), &alloc));
    }
  }

  return std::move(result_buffer);
}

Status CpuExecutable::ExecuteOnStream(
    const ExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    ShapedBuffer* result_buffer, HloExecutionProfile* hlo_execution_profile) {
  // Every array element in the result of the computation must be unambiguously
  // produced by a single instruction.
  // This ensures that the buffers inside result_buffer can be assigned without
  // conflict to the respective instructions because there is a one-to-one
  // correspondence between hlo instructions and array buffers in the result.
  if (GetRootPointsToSet().IsAmbiguous()) {
    return Unimplemented(
        "Points-to set of root instruction is ambiguous or not distinct");
  }
  if (!ShapeUtil::Compatible(result_buffer->shape(), result_shape())) {
    return InvalidArgument(
        "Result buffer shape %s is incompatible with result shape %s",
        ShapeUtil::HumanString(result_buffer->shape()).c_str(),
        ShapeUtil::HumanString(result_shape()).c_str());
  }

  se::Stream* stream = run_options->stream();
  DeviceMemoryAllocator* memory_allocator = run_options->allocator();
  std::vector<se::DeviceMemoryBase> buffers(assignment_->Allocations().size());

  // If two tuple elements point to the same buffer, one of the results in the
  // result buffer is considered the canonical location while the other result
  // points to it (instead of, say, making a copy of the result).
  // buffer_index_to_shape_index maps a buffer index to its canonical location
  // in the result buffer.
  std::unordered_map<BufferAllocation::Index, size_t>
      buffer_index_to_shape_index;

  // Copy values from result_buffer to the index in "buffers". These buffers
  // will not be allocated in the call to AllocateBuffers.
  std::vector<bool> buffers_in_result(assignment_->Allocations().size(), false);
  TF_RETURN_IF_ERROR(
      result_buffer->mutable_shape_index_to_buffer_entry()
          ->ForEachMutableElement(
              [&buffers, &buffers_in_result, &buffer_index_to_shape_index,
               result_buffer, this](const ShapeIndex& index, bool is_leaf,
                                    size_t* buffer_entry) {
                if (is_leaf) {
                  const std::vector<const LogicalBuffer*>& sources =
                      this->GetRootPointsToSet().element(index);
                  // The points to set is unambiguous so the set should be a
                  // singleton.
                  CHECK_EQ(1, sources.size());
                  const LogicalBuffer* buffer_source = sources[0];
                  HloInstruction* src = buffer_source->instruction();

                  // The source for this result buffer can be a nested buffer
                  // such as a tuple element.

                  // The source instruction should have a non-parameter buffer
                  // assigned.
                  TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice slice,
                                      this->assignment_->GetUniqueSlice(
                                          src, buffer_source->index()));
                  CHECK(!slice.allocation()->is_entry_computation_parameter());

                  const BufferAllocation::Index buffer_index = slice.index();
                  auto insert_result = buffer_index_to_shape_index.emplace(
                      buffer_index, *buffer_entry);
                  if (insert_result.second) {
                    // The points-to set is distinct so this buffer should not
                    // have
                    // been assigned in a previous invocation of this lambda.
                    perftools::gputools::DeviceMemoryBase memory_base =
                        result_buffer->buffer(index);
                    CHECK(!memory_base.is_null());
                    CHECK(buffers[buffer_index].is_null());
                    buffers[buffer_index] = memory_base;
                    buffers_in_result[buffer_index] = true;
                  } else {
                    // Record the fact that this tuple element is identical to
                    // some
                    // prior result.
                    *buffer_entry = insert_result.first->second;
                  }
                }
                return Status::OK();
              }));

  TF_RETURN_IF_ERROR(AllocateBuffers(
      memory_allocator, stream->parent()->device_ordinal(), &buffers));
  TF_RETURN_IF_ERROR(ExecuteComputeFunction(run_options, arguments, buffers,
                                            hlo_execution_profile));

  // Free all buffers not in the result.
  for (size_t i = 0; i < buffers.size(); ++i) {
    se::DeviceMemoryBase alloc = buffers[i];
    if (!buffers_in_result[i] && !alloc.is_null()) {
      VLOG(3) << "CpuExecutable deallocating buffer #" << i << " ["
              << alloc.opaque() << "]";
      TF_RETURN_IF_ERROR(memory_allocator->Deallocate(
          stream->parent()->device_ordinal(), &alloc));
    }
  }

  return Status::OK();
}

StatusOr<perftools::gputools::DeviceMemoryBase>
CpuExecutable::ExecuteAsyncOnStream(
    const ExecutableRunOptions* run_options,
    tensorflow::gtl::ArraySlice<se::DeviceMemoryBase> arguments) {
  // TODO(b/30671675): Implement asynchronous execution mode.
  return Unimplemented(
      "Asynchronous execution on stream is not yet supported on CPU.");
}

const PointsToSet& CpuExecutable::GetRootPointsToSet() const {
  return assignment_->points_to_analysis().GetPointsToSet(
      module().entry_computation()->root_instruction());
}

}  // namespace cpu
}  // namespace xla
