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

#include "tensorflow/compiler/xla/tests/local_client_test_base.h"

#include <vector>

#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

/* static */ TestAllocator* LocalClientTestBase::allocator_;

StatusOr<perftools::gputools::DeviceMemoryBase> TestAllocator::Allocate(
    int device_ordinal, uint64 size, bool retry_on_failure) {
  VLOG(2) << "Allocate(" << device_ordinal << ", " << size << ")";
  {
    tensorflow::mutex_lock lock(count_mutex_);
    allocation_count_++;
    device_allocation_count_[device_ordinal]++;
  }
  return StreamExecutorMemoryAllocator::Allocate(device_ordinal, size);
}

tensorflow::Status TestAllocator::Deallocate(
    int device_ordinal, perftools::gputools::DeviceMemoryBase* mem) {
  VLOG(2) << "Deallocate(" << device_ordinal << ")";
  {
    tensorflow::mutex_lock lock(count_mutex_);
    deallocation_count_++;
    device_deallocation_count_[device_ordinal]++;
  }
  return StreamExecutorMemoryAllocator::Deallocate(device_ordinal, mem);
}

int64 TestAllocator::allocation_count() const {
  tensorflow::mutex_lock lock(count_mutex_);
  return allocation_count_;
}

int64 TestAllocator::allocation_count(int device_ordinal) const {
  tensorflow::mutex_lock lock(count_mutex_);
  auto it = device_allocation_count_.find(device_ordinal);
  if (it == device_allocation_count_.end()) {
    return 0;
  } else {
    return it->second;
  }
}

int64 TestAllocator::deallocation_count() const {
  tensorflow::mutex_lock lock(count_mutex_);
  return deallocation_count_;
}

int64 TestAllocator::deallocation_count(int device_ordinal) const {
  tensorflow::mutex_lock lock(count_mutex_);
  auto it = device_deallocation_count_.find(device_ordinal);
  if (it == device_deallocation_count_.end()) {
    return 0;
  } else {
    return it->second;
  }
}

/* static */ TestAllocator* LocalClientTestBase::GetOrCreateAllocator(
    perftools::gputools::Platform* platform) {
  if (allocator_ == nullptr) {
    allocator_ = new TestAllocator(
        platform == nullptr ? PlatformUtil::GetDefaultPlatform().ValueOrDie()
                            : platform);
  }
  return allocator_;
}

LocalClientTestBase::LocalClientTestBase(
    perftools::gputools::Platform* platform)
    : local_client_(
          ClientLibrary::GetOrCreateLocalClient(platform).ValueOrDie()) {
  stream_executor_ = PlatformUtil::GetStreamExecutors(local_client_->platform())
                         .ValueOrDie()[local_client_->default_device_ordinal()];
  transfer_manager_ =
      TransferManager::GetForPlatform(local_client_->platform()).ValueOrDie();
}

std::unique_ptr<ScopedShapedBuffer>
LocalClientTestBase::LiteralToScopedShapedBuffer(const Literal& literal) {
  return LiteralToScopedShapedBuffer(literal,
                                     local_client_->default_device_ordinal());
}

std::unique_ptr<ScopedShapedBuffer>
LocalClientTestBase::LiteralToScopedShapedBuffer(const Literal& literal,
                                                 int device_ordinal) {
  CHECK(!ShapeUtil::IsTuple(literal.shape()));
  auto scoped_buffer =
      ScopedShapedBuffer::MakeScopedShapedBuffer(
          literal.shape(), GetOrCreateAllocator(local_client_->platform()),
          device_ordinal)
          .ConsumeValueOrDie();
  // The creation of the scoped shaped buffer should allocate the buffer.
  CHECK(!scoped_buffer->buffer(/*index=*/{}).is_null() ||
        ShapeUtil::HasZeroElements(literal.shape()));
  TF_CHECK_OK(transfer_manager_->TransferLiteralToDevice(
      stream_executor_, literal, scoped_buffer->mutable_buffer(/*index=*/{})));
  return scoped_buffer;
}

void LocalClientTestBase::CopyShapedBufferToLiteral(
    const ShapedBuffer& shaped_buffer, ShapeIndex* index, Literal* literal) {
  const Shape& shape = ShapeUtil::GetSubshape(shaped_buffer.shape(), *index);
  if (ShapeUtil::IsTuple(shape)) {
    *literal->mutable_shape() = shape;
    for (int i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
      Literal* element_literal = literal->add_tuple_literals();
      index->push_back(i);
      CopyShapedBufferToLiteral(shaped_buffer, index, element_literal);
      index->pop_back();
    }
  } else {
    ASSERT_IS_OK(transfer_manager_->TransferLiteralFromDevice(
        stream_executor_, shaped_buffer.buffer(*index), shape, shape, literal));
  }
}

std::unique_ptr<Literal> LocalClientTestBase::ShapedBufferToLiteral(
    const ShapedBuffer& shaped_buffer) {
  auto literal = MakeUnique<Literal>();
  ShapeIndex index;
  CopyShapedBufferToLiteral(shaped_buffer, &index, literal.get());
  return literal;
}

std::unique_ptr<ScopedShapedBuffer>
LocalClientTestBase::ShapedBufferToScopedShapedBuffer(
    std::unique_ptr<ShapedBuffer> shaped_buffer,
    DeviceMemoryAllocator* allocator) {
  std::unique_ptr<ScopedShapedBuffer> scoped_buffer =
      ScopedShapedBuffer::MakeScopedShapedBuffer(
          shaped_buffer->shape(), allocator, shaped_buffer->device_ordinal())
          .ConsumeValueOrDie();
  // Deallocate the existing DeviceMemoryBase values in the newly created scoped
  // buffer and replace them with the values from the shaped buffer.
  for (perftools::gputools::DeviceMemoryBase& memory_base :
       *scoped_buffer->mutable_buffers()) {
    TF_CHECK_OK(
        allocator->Deallocate(shaped_buffer->device_ordinal(), &memory_base));
  }
  *scoped_buffer->mutable_buffers() = shaped_buffer->buffers();

  TF_CHECK_OK(
      scoped_buffer->mutable_shape_index_to_buffer_entry()
          ->ForEachMutableElement(
              [&shaped_buffer](const ShapeIndex& index, bool is_leaf,
                               size_t* buffer_entry) -> ::tensorflow::Status {
                if (is_leaf) {
                  *buffer_entry =
                      shaped_buffer->shape_index_to_buffer_entry().element(
                          index);
                }
                return tensorflow::Status::OK();
              }));
  return scoped_buffer;
}

LocalExecuteOptions LocalClientTestBase::DefaultLocalExecuteOptions() const {
  return LocalExecuteOptions().set_allocator(
      GetOrCreateAllocator(local_client_->platform()));
}

std::unique_ptr<ScopedShapedBuffer> LocalClientTestBase::ExecuteLocally(
    const Computation& computation,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments) {
  return ExecuteLocally(computation, arguments, DefaultLocalExecuteOptions());
}

std::unique_ptr<ScopedShapedBuffer> LocalClientTestBase::ExecuteLocally(
    const Computation& computation,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    const LocalExecuteOptions& options) {
  return ShapedBufferToScopedShapedBuffer(
      local_client_->ExecuteLocally(computation, arguments, options)
          .ConsumeValueOrDie(),
      options.allocator());
}

void LocalClientTestBase::ExecuteLocally(
    const Computation& computation,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    ShapedBuffer* result) {
  ExecuteLocally(computation, arguments, DefaultLocalExecuteOptions(), result);
}

void LocalClientTestBase::ExecuteLocally(
    const Computation& computation,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    const LocalExecuteOptions& options, ShapedBuffer* result) {
  ASSERT_IS_OK(
      local_client_->ExecuteLocally(computation, arguments, options, result));
}

}  // namespace xla
