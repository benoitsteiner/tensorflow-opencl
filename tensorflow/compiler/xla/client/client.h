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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_CLIENT_H_

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/service/session.pb.h"
#include "tensorflow/compiler/xla/service_interface.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {

// XLA service's client object -- wraps the service with convenience and
// lifetime-oriented methods.
class Client {
 public:
  explicit Client(ServiceInterface* stub);
  virtual ~Client();

  // Executes the computation with the given arguments and returns the global
  // data that was produced from the execution.
  // * If shape_with_output_layout is not nullptr this points to a shape with a
  //   layout to use as a hint when storing the output of the computation.
  //   Subsequent transfers of this output array to the client may be faster
  //   when using this layout.
  // * If execution_profile is not nullptr then the pointed-to ExecutionProfile
  //   will be filled with profile data from the execution.
  // * If seed is set then that seed is used for the graph execution.
  StatusOr<std::unique_ptr<GlobalData>> Execute(
      const Computation& computation,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments,
      const Shape* shape_with_output_layout = nullptr,
      ExecutionProfile* execution_profile = nullptr, uint64 seed = 0);

  // A struct to represent a computation instance to be executed.
  // * If device_handle is not nullptr, the computation is executed on a device
  //   associated with the handle. Otherwise, a device is chosen by the service.
  // * If shapes_with_output_layout is not nullptr, the given shape and its
  //   layout is used as a hint when storing the output of the computation.
  // * If execution_profile is not nullptr, the pointed-to ExecutionProfile will
  //   be filled with profile data from the execution of the computation.
  // * seed is for the random number generator used in the computation.
  struct ComputationInstance {
    const Computation& computation;
    std::vector<GlobalData*> arguments;
    const DeviceHandle* device_handle;
    const Shape* shape_with_output_layout;
    ExecutionProfile* execution_profile;
    uint64 seed;
  };

  // Executes a list ComputationInstances and returns global data produced from
  // each computation.
  StatusOr<std::vector<std::unique_ptr<GlobalData>>> ExecuteParallel(
      tensorflow::gtl::ArraySlice<ComputationInstance> computations);

  // Requests device_count device handles available on the target. The returned
  // device handles are used to specify the devices to execute the computations
  // (see ExecuteParallel) or to transfer data (see TransferToServer or
  // TransferToInfeed).
  StatusOr<std::vector<DeviceHandle>> GetDeviceHandles(int64 device_count);

  // Executes the given computation as above Execute(), but launches the
  // computation asynchronously and returns before the execution is complete.
  // Returns an ExecutionHandle that represents the launched execution, which is
  // used to call WaitForExecution() to wait for the execution's completion.
  StatusOr<ExecutionHandle> ExecuteAsync(
      const Computation& computation,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments,
      const Shape* shape_with_output_layout = nullptr, uint64 seed = 0);

  // Waits until the given asynchronously launched execution of the computation
  // is complete and returns the execution result. Once this is called, the
  // given execution handle is no longer valid. If execution_profile is not
  // nullptr then the pointed-to ExecutionProfile will be filled with profile
  // data from the execution.
  StatusOr<std::unique_ptr<GlobalData>> WaitForExecution(
      const Computation& computation, const ExecutionHandle& execution,
      ExecutionProfile* execution_profile = nullptr);

  // Transfer the global data provided to this client process, which is
  // returned in the provided literal. Use sparingly to avoid transfer
  // overheads.
  //
  // If shape_with_layout is not nullptr, it points to a shape whose layout will
  // be the layout of the returned literal.
  StatusOr<std::unique_ptr<Literal>> Transfer(
      const GlobalData& data, const Shape* shape_with_layout = nullptr);

  // Transfer the given literal to the server. This allocates memory on the
  // device and copies the literal's contents over. Returns a global data handle
  // that can be used to refer to this value from the client.
  //
  // If device_handle is not nullptr, data is transferred to the associated
  // device (and its replicas if replication is enabled). Otherwise, data is
  // transferred to the default device (and its replicas).
  StatusOr<std::unique_ptr<GlobalData>> TransferToServer(
      const Literal& literal, const DeviceHandle* device_handle = nullptr);

  // Transfer the given literal to the Infeed interface of the device.
  //
  // device_handle and replica_id together specify a particular device; a device
  // assigned for the given replica_id among the replicas that the given device
  // handle belongs to.
  Status TransferToInfeed(const Literal& literal, int64 replica_id = 0,
                          const DeviceHandle* device_handle = nullptr);

  // Resets the device, clearing all existing state on the device.
  Status ResetDevice();

  // Executes the computation with the given arguments and transfers the result
  // to the client as a literal. Parameters are defined the same as for
  // Execute() and Transfer().
  StatusOr<std::unique_ptr<Literal>> ExecuteAndTransfer(
      const Computation& computation,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments,
      const Shape* shape_with_output_layout = nullptr,
      ExecutionProfile* execution_profile = nullptr, uint64 seed = 0);

  // Unregister the memory for the given GlobalData on the device.
  Status Unregister(const GlobalData& data);

  // Returns a vector of global data handles that point to the tuple elements.
  StatusOr<std::vector<std::unique_ptr<GlobalData>>> DeconstructTuple(
      const GlobalData& computation);

  // Retrieves the statistics of the given computation.
  StatusOr<ComputationStats> GetComputationStats(
      const Computation& computation) const;

  // Returns the Shape of the given array specified by 'data'. The shape
  // includes the Layout of the array as it is stored on the service. The layout
  // information is useful for calling TransferInProcess.
  StatusOr<Shape> GetShape(const GlobalData& data);

  // As above, but returns the shape of the provided computation (parameter
  // types/names and return type).
  StatusOr<std::unique_ptr<ProgramShape>> GetComputationShape(
      const Computation& computation);

  // Creates a channel handle that can be used to transfer data between
  // two computations via a pair of Send and Recv instructions.
  StatusOr<ChannelHandle> CreateChannelHandle();

  // If the service is running in the same process as the client then the
  // following "InProcess" transfer methods may be used. These methods enable
  // more efficient transfer of arrays to and from the service.

  // Transfer array from the service into the given buffer. The buffer must be
  // large enough to hold the array. The array is copied verbatim (memcpy) from
  // the service. The method GetShape should be called ahead of time
  // to get the shape and layout of the array as it is stored in the
  // service. The shape and layout can be used to determine how large the buffer
  // needs to be.
  Status TransferInProcess(const GlobalData& data, void* destination);

  // Transfer array to the service from the given buffer with the given shape
  // and layout. The service creates an internal copy of the data so the client
  // can free the buffer when this method returns.
  StatusOr<std::unique_ptr<GlobalData>> TransferToServerInProcess(
      const Shape& shape, const void* buffer);

  StatusOr<Computation> LoadSnapshot(const SessionModule& module);

  ServiceInterface* stub() { return stub_; }

 private:
  // Returns the execution statistics (e.g., gflop/s) as a string from the
  // ExecutionProfile returned from an execution of the computation.
  StatusOr<string> ExecutionStatsAsString(const Computation& computation,
                                          const ExecutionProfile& profile);

  ServiceInterface* stub_;  // Stub that this client is connected on.

  TF_DISALLOW_COPY_AND_ASSIGN(Client);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_CLIENT_H_
