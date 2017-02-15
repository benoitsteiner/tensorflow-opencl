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
#ifndef THIRD_PARTY_TENSORFLOW_CORE_COMMON_RUNTIME_SHAPE_REFINER_H_
#define THIRD_PARTY_TENSORFLOW_CORE_COMMON_RUNTIME_SHAPE_REFINER_H_

#include <vector>

#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

// ShapeRefiner performs shape inference for TensorFlow Graphs.  It is
// responsible for instantiating InferenceContext objects for each
// Node in the Graph, and providing/storing the 'input_tensor' Tensors
// used by Shape Inference functions, when available at graph
// construction time.
class ShapeRefiner {
 public:
  explicit ShapeRefiner(const OpRegistryInterface* ops);

  // Performs validation of 'node' and runs 'node's shape function,
  // storing its shape outputs.
  //
  // All inputs of 'node' must be added to ShapeRefiner prior to
  // adding 'node'.
  //
  // Returns an error if:
  //  - the shape function for 'node' was not registered.
  //  - 'node' was added before its inputs.
  //  - The shape inference function returns an error.
  Status AddNode(const Node* node);

  // Sets 'node's 'output_port' output to have shape 'shape'.
  //
  // Returns an error if 'node' was not previously added to this
  // object, if 'output_port' is invalid, or if 'shape' is
  // not compatible with the existing shape of the output.
  Status SetShape(const Node* node, int output_port,
                  shape_inference::ShapeHandle shape);

  // Returns the InferenceContext for 'node', if present.
  shape_inference::InferenceContext* GetContext(const Node* node) const {
    auto it = node_to_context_.find(node);
    if (it == node_to_context_.end()) {
      return nullptr;
    }
    return it->second.get();
  }

 private:
  // Extracts the subgraph ending at 'node' that is statically
  // computable and inserts into 'out_graph'. If statically computable,
  // 'is_constant_graph' will be true.
  Status ExtractConstantSubgraph(
      Node* node, Graph* out_graph, bool* is_constant_graph,
      std::vector<std::pair<string, Tensor>>* const_inputs) TF_MUST_USE_RESULT;

  Status EvaluateConstantTensorForEdge(const Node* node, int dst_idx,
                                       bool* evaluated, Tensor* result);

  // This function tries to materialize as much information about the 'node''s
  // dst_idx input as a statically computable shape, and the result may be
  // partially known, depending on what is statically inferable.
  //
  // This is called when node.input[dst_idx] is a tensor that is used to define
  // the shape of some other tensor (e.g., the second argument to Reshape is a
  // <shape> tensor, where each element of the shape tensor is a dimension of
  // the target tensor).  It returns in <result> a shape for that input.
  //
  // Unlike simply resolving node.input[dst_idx] to a constant and then
  // converting that to a shape, this function can return a partial shape. This
  // is useful for cases where the shape tensor is only partially defined, such
  // as with calls for: reshape(x, shape(y)) where shape(y) is partially
  // defined.
  //
  // The implementation has op implementations for ops commonly called on shape
  // tensors, and the implementations are specialized to shape tensors (namely,
  // the output is a vector).
  //
  // <target_context> is used when creating new DimensionHandle and ShapeHandle
  // objects.
  Status ConstantPartialShape(shape_inference::InferenceContext* target_context,
                              const Node* node, int dst_idx,
                              shape_inference::ShapeHandle* result);

  const OpRegistryInterface* ops_registry_ = nullptr;

  // Stores a map from a node to its InferenceContext.
  //
  // Owns values.
  std::unordered_map<const Node*,
                     std::unique_ptr<shape_inference::InferenceContext>>
      node_to_context_;

  // Holds a cache from 'tensor name' to the tensor that is
  // evaluatable as a constant expression.  This reduces repeated
  // execution of the entire constant subgraph as a graph is being
  // built up.  This could be changed to some kind of size-based LRU
  // cache to avoid consuming too much memory, if that eventually
  // becomes a concern.
  //
  // Only tensors less than 1KiB are currently stored in the cache.
  static constexpr int64 kMaxTensorSize = 1024;
  std::unordered_map<string, Tensor> const_tensor_map_;
  TF_DISALLOW_COPY_AND_ASSIGN(ShapeRefiner);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_COMMON_RUNTIME_SHAPE_REFINER_H_
