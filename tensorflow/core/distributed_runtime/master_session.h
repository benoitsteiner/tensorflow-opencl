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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_SESSION_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_SESSION_H_

#include <atomic>
#include <vector>

#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/simple_graph_execution_state.h"
#include "tensorflow/core/common_runtime/stats_publisher_interface.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/master_env.h"
#include "tensorflow/core/distributed_runtime/message_wrappers.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/master.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class Device;
struct MasterEnv;

// A session encapsulates a graph computation (resource allocation,
// placement, execution, etc.).
class MasterSession : public core::RefCounted {
 public:
  // This session encapsulates the graph computation for a graph.
  //
  // The session places nodes on devices in "remote_devs" and executes
  // operations on these devices.
  //
  // The caller takes ownership of all remote devices.
  MasterSession(const SessionOptions& options, const MasterEnv* env,
                std::vector<Device*>* remote_devs,
                StatsPublisherFactory stats_publisher_factory);

  // Initialize the MasterSession for "def".  Must be called before Extend(),
  // Run(), or Close().
  //
  // After this method returns, `def` will no longer be valid.
  Status Create(GraphDef* def);

  // Returns the session handle.
  const string& handle() const { return handle_; }

  // Returns the last access time (the number of micro-seconds since
  // some fixed point in time) of this session.
  uint64 last_access_time_usec() const { return last_access_time_usec_.load(); }

  // Attempt to extend the graph according to the given "req".
  // (See master.proto for details of valid extensions.)
  //
  // PRECONDITION: The current version of this session's graph
  //   is "req->current_graph_version".
  //
  // POSTCONDITION: The current version of this session's graph
  //   is "resp->new_graph_version".
  //
  // Extend() may block the caller thread for a long time.
  Status Extend(const ExtendSessionRequest* req, ExtendSessionResponse* resp);

  // Setup a partial run call.
  Status PartialRunSetup(const PartialRunSetupRequest* req,
                         PartialRunSetupResponse* resp);

  // Run one step.
  Status Run(CallOptions* opts, const RunStepRequestWrapper& req,
             MutableRunStepResponseWrapper* resp);

  // Close this session and delete "*this". Returns OK if all known
  // states are cleanup successfully.
  //
  // Close() may block the caller thread for a long time.
  Status Close();

 private:
  SessionOptions session_opts_;

  // Not owned.
  const MasterEnv* env_;

  // The opaque session handle.
  const string handle_;

  // Owned.
  std::vector<Device*> remote_devs_;

  // The device set used by this session.
  DeviceSet devices_;

  StatsPublisherFactory stats_publisher_factory_;

  std::atomic_ulong last_access_time_usec_;

  std::atomic<int64> partial_run_handle_counter_ = {0};

  mutex mu_;
  std::unique_ptr<SimpleGraphExecutionState> execution_state_;
  int64 graph_version_;

  // We keep a map from a signature of a run request to the
  // ReffedClientGraph the can execute it.  We keep up to one old copy
  // of each ReffedClientGraph around because if it gets deallocated
  // before a new substitute has been created, Variables can go out of
  // scope and lose their state.
  class ReffedClientGraph;
  typedef std::unordered_map<uint64, ReffedClientGraph*> RCGMap;
  RCGMap run_graphs_ GUARDED_BY(mu_);
  RCGMap partial_run_graphs_ GUARDED_BY(mu_);

  struct PerStepState {
    bool collect_costs = false;
    bool collect_timeline = false;
    bool collect_rpcs = false;
    Microseconds start_micros = Microseconds(0);
    Microseconds end_micros = Microseconds(0);
    std::vector<StepStats> step_stats;  // per partition
    StepStats rpc_stats;                // for RPC layer
    CostGraphDef cost_graph;
  };

  struct RunState {
    std::unordered_set<string> pending_inputs;
    std::unordered_set<string> pending_outputs;
    ReffedClientGraph* rcg = nullptr;
    uint64 step_id;
    int64 count = 0;
    PerStepState pss;
    std::unique_ptr<ProfileHandler> ph;
    bool step_started = false;

    RunState(const std::vector<string>& input_names,
             const std::vector<string>& output_names, ReffedClientGraph* rcg,
             const uint64 step_id, const int64 count);

    ~RunState();
  };
  std::unordered_map<string, std::unique_ptr<RunState>> partial_runs_
      GUARDED_BY(mu_);

  // Active RunStep calls.
  condition_variable num_running_is_zero_;
  int32 num_running_ GUARDED_BY(mu_) = 0;

  bool closed_ GUARDED_BY(mu_) = false;

  std::unordered_map<uint64, int64> subgraph_execution_counts_ GUARDED_BY(mu_);

  // We need to ensure that certain nodes added (e.g., send and recv
  // nodes) are unique across all sub-graphs within this session.
  int64 next_node_id_ GUARDED_BY(mu_) = 0;

  // Used to cancel running steps on Close().
  CancellationManager* cancellation_manager_;

  // Private dtor. The client must call Close().
  virtual ~MasterSession();

  Status StartStep(const BuildGraphOptions& opts, int64* count,
                   ReffedClientGraph** graph, bool is_partial);
  void ClearRunsTable(std::vector<ReffedClientGraph*>* to_unref,
                      RCGMap* rcg_map) EXCLUSIVE_LOCKS_REQUIRED(mu_);
  Status DoRunWithLocalExecution(CallOptions* opts,
                                 const RunStepRequestWrapper& req,
                                 MutableRunStepResponseWrapper* resp);
  Status DoPartialRun(CallOptions* opts, const RunStepRequestWrapper& req,
                      MutableRunStepResponseWrapper* resp);
  void UpdateLastAccessTime();

  Status BuildAndRegisterPartitions(ReffedClientGraph* rcg);

  TF_DISALLOW_COPY_AND_ASSIGN(MasterSession);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_SESSION_H_
