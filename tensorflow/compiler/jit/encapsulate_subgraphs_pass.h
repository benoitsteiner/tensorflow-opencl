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

// An optimization pass that groups nodes marked with a common
// kXlaClusterAttr into functions, and replaces the original nodes by
// calls. The calls are annotated with kXlaCompiledKernelAttr.

#ifndef TENSORFLOW_COMPILER_JIT_ENCAPSULATE_SUBGRAPHS_PASS_H_
#define TENSORFLOW_COMPILER_JIT_ENCAPSULATE_SUBGRAPHS_PASS_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// A rewriting function to apply to each subgraph during encapsulation.
// 'graph' is the subgraph. The rewriting may renumber the inputs and outputs;
// 'input_permutation' is a mapping from old argument numbers to new argument
// numbers, whereas 'output_permutation' is the same for outputs. Both
// 'input_permutation' and 'output_permutation' are initialized to the identity
// permutation. 'nodedef' is the NodeDef for the call to the function under
// construction, provided to allow additional attributes to be set.
typedef std::function<Status(
    std::unique_ptr<Graph>* graph, std::vector<int>* input_permutation,
    std::vector<int>* output_permutation, NodeDef* node_def)>
    RewriteSubgraphFn;

// Transformation that finds subgraphs whose nodes are marked with
// 'group_attribute', splits those subgraphs into functions, and replaces
// the originals with function calls.
//
// 'group_attribute' must be a string valued-attribute that names the new
// functions to introduce.
//
// If 'rewrite_subgraph_fn' is set, it is applied to each subgraph before
// function conversion.
//
// If 'parallel_checking' is true, the unencapsulated operators are added to the
// output graph, together with a "ParallelCheck" operator, that verifies that
// the original and encapsulated subgraphs produce similar results.
//
// TODO(phawkins): currently, some information in control edges
// is not preserved. Suppose you have A and B in the main
// graph, C and D in a subgraph. B and C have control deps from A, D has control
// dep from B. Originally D must run after C, post-transformation this
// dependency is lost.
Status EncapsulateSubgraphsInFunctions(
    string group_attribute, const Graph& graph_in,
    const RewriteSubgraphFn& rewrite_subgraph_fn, bool parallel_checking,
    std::unique_ptr<Graph>* graph_out, FunctionLibraryDefinition* library);

// The attribute that marks function calls produced by the encapsulate
// subgraphs pass and that should in turn be compiled via _XlaLaunch operators.
extern const char* const kXlaCompiledKernelAttr;

// Does `node` have the kXlaCompiledKernelAttr attribute?
bool IsXlaCompiledKernel(const Node& node);

// Functions produce by the EncapsulateSubgraphs pass have their arguments
// ordered such that compile-time constant arguments are first in the argument
// order. The functions are annotated with the following attribute giving the
// number of constant arguments.
extern const char* const kXlaNumConstantArgsAttr;

class EncapsulateSubgraphsPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_ENCAPSULATE_SUBGRAPHS_PASS_H_
