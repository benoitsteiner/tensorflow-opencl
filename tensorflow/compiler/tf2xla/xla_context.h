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

// This file defines the contexts used during XLA compilation.

#ifndef TENSORFLOW_COMPILER_TF2XLA_XLA_CONTEXT_H_
#define TENSORFLOW_COMPILER_TF2XLA_XLA_CONTEXT_H_

#include <vector>

#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

// A XlaExpression wraps an XLA computation. Each Tensor sent
// along an edge during XLA compilation represents a
// XlaExpression, and the shape of the Tensor matches the shape of
// the subcomputation in the ComputationDataHandle. Each
// expression is either a constant, or a function of previously-compiled
// expressions.
class XlaExpression {
 public:
  XlaExpression();

  // handle() stores the XLA handle of the computation that the
  // expression represents.
  void set_handle(const xla::ComputationDataHandle& h);
  const xla::ComputationDataHandle& handle() const;

  void set_constant_value(Tensor value);
  bool has_constant_value() const { return has_constant_value_; }
  const Tensor& constant_value() const { return constant_value_; }

 private:
  // The XLA handle of the expression's computation.
  xla::ComputationDataHandle handle_;

  // If this expression is a constant with a known value, 'constant_value' is a
  // host-memory Tensor containing the value. Used to avoid invoking XLA for
  // expressions that are trivially constant.
  bool has_constant_value_;
  Tensor constant_value_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaExpression);
};

// The XlaContext is the data structure that holds the state of an XLA
// compilation, that is accessible from OpKernelContexts when compiling a
// subgraph of Ops using XLA.
class XlaContext : public ResourceBase {
 public:
  // A struct that represents either a compile-time constant, or an XLA
  // computation handle. Used to represent arguments and return values.
  struct HandleOrConstant {
    // Is this a compile-time constant? If so, what is its value?
    bool is_constant;
    Tensor constant_value;  // Must be in host memory.

    // If this is not a constant, a computation handle.
    xla::ComputationDataHandle handle;
  };

  // Retrieves the XlaContext of the current compilation.
  static XlaContext& Get(const OpKernelContext* ctx);
  static XlaContext& Get(const XlaOpKernelContext* ctx) {
    return Get(ctx->op_kernel_context());
  }

  // Creates a new XlaContext.
  XlaContext(XlaCompiler* compiler, xla::ComputationBuilder* builder,
             bool allow_cpu_custom_calls, bool resolve_compile_time_constants);

  // Virtual method defined by ResourceBase.
  string DebugString() override;

  XlaCompiler* compiler() const { return compiler_; }

  // Returns the ComputationBuilder that Ops use for compiling new
  // expressions.
  xla::ComputationBuilder* builder();

  bool allow_cpu_custom_calls() const { return allow_cpu_custom_calls_; }
  bool has_context_parameter() const { return has_context_parameter_; }

  const std::vector<HandleOrConstant>& args() const { return args_; }
  void set_args(std::vector<HandleOrConstant> args);

  // Get the runtime context parameter, adding one if it does not already exist.
  // Dies if not compiling a local executable.
  const xla::ComputationDataHandle& GetOrCreateRuntimeContextParameter();

  const std::vector<HandleOrConstant>& retvals() { return retvals_; }

  // This is called by the Retval Op to associate a computed value
  // with a specific return value of the subgraph.
  void AddRetval(int retval_index, const xla::ComputationDataHandle& handle);

  // As for Retval, but for return values that are compile-time constants.
  Status AddConstRetval(int retval_index, DataType dtype,
                        const xla::Literal& literal);

  // Mark the computation as having side effects (e.g., Send operators).
  void AddSideEffects();

  bool has_side_effects() const { return has_side_effects_; }

  // Get an XLA lambda to compute Max. This is cached in the
  // XlaContext since it may be used by multiple Ops. There is a
  // separate specialization of the computation for each DataType.
  const xla::Computation* GetOrCreateMax(const DataType type);

  // Get an XLA lambda to compute Add. This is cached in the
  // XlaContext since it may be used by multiple Ops. There is a
  // separate specialization of the computation for each DataType.
  const xla::Computation* GetOrCreateAdd(const DataType type);

  // Get an XLA lambda to compute Sigmoid. This is cached in the
  // XlaContext since it may be used by multiple Ops. There is a
  // separate specialization of the computation for each DataType.
  const xla::Computation* GetOrCreateSigmoid(const DataType type);

  // The name of the XlaContext resource during symbolic graph execution.
  static const char kXlaContextResourceName[];

 private:
  XlaCompiler* const compiler_;

  // The ComputationBuilder used to construct the subgraph's compiled
  // representation.
  xla::ComputationBuilder* builder_;

  // Allow ops to emit CustomCall operations for CPU.
  const bool allow_cpu_custom_calls_;

  // If true, constant return values are returned as Tensors instead of
  // run-time computation outptus.
  const bool resolve_compile_time_constants_;

  // When 'has_context_parameter_' is true, this is the computation handle
  // for an additional final parameter to the computation, through which will be
  // passed a XlaLocalRuntimeContext* at runtime. Created on demand by
  // GetOrCreateRuntimeContextParameter().
  bool has_context_parameter_ = false;
  xla::ComputationDataHandle context_parameter_;

  // Arguments to the Tensorflow graph, indexed by _Arg index.
  // Includes both compile-time constant arguments and runtime parameters.
  std::vector<HandleOrConstant> args_;

  // Return values of the Tensorflow graph, indexed by _Retval index.
  std::vector<HandleOrConstant> retvals_;

  // Does the computation have side effects, i.e., Send() calls?
  bool has_side_effects_ = false;

  // Cache of prebuilt computations indexed by their type.
  using ComputationMap = std::map<DataType, xla::Computation>;

  // Finds the value for the given type in out map if it already
  // exists or makes a new value with create function and keeps it the
  // map. The returned value != nullptr and is owned by the map.
  const xla::Computation* LookupOrCreate(
      DataType type, ComputationMap* out,
      const std::function<xla::Computation()>& create);

  // Cached computation to compute Max of two elements, specialized by type.
  ComputationMap max_func_;

  // Cached computation to compute Sum of two elements, specialized by type.
  ComputationMap add_func_;

  // Cached computation to compute Sigmoid of an element, specialized by type.
  ComputationMap sigmoid_func_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaContext);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_XLA_CONTEXT_H_
