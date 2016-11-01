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

#include "tensorflow/core/graph/control_flow.h"

#include <deque>
#include <vector>

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

Status BuildControlFlowInfo(Graph* g, std::vector<ControlFlowInfo>* info) {
  info->clear();
  info->resize(g->num_node_ids());

  std::vector<const Node*> parent_nodes;
  parent_nodes.resize(g->num_node_ids());

  Node* src_node = g->source_node();
  ControlFlowInfo& src_info = (*info)[src_node->id()];
  src_info.frame = src_node;
  src_info.parent_frame = src_node;

  string frame_name;
  std::deque<Node*> ready;
  ready.push_back(src_node);
  while (!ready.empty()) {
    Node* curr_node = ready.front();
    ready.pop_front();
    const ControlFlowInfo& curr_info = (*info)[curr_node->id()];
    const Node* frame = curr_info.frame;
    const Node* parent = curr_info.parent_frame;
    frame_name = curr_info.frame_name;

    if (IsExit(curr_node)) {
      // Exit to the parent frame.
      const ControlFlowInfo& parent_info = (*info)[parent->id()];
      frame = parent_info.frame;
      parent = parent_info.parent_frame;
      frame_name = parent_info.frame_name;
    }

    for (const Edge* out_edge : curr_node->out_edges()) {
      Node* out = out_edge->dst();
      int out_id = out->id();
      ControlFlowInfo* out_info = &(*info)[out_id];
      const Node* out_parent = out_info->parent_frame;
      bool is_visited = (parent_nodes[out_id] != nullptr);

      // Skip Sink/Source nodes.
      if (!out->IsOp()) continue;

      // Add to ready queue if not seen.
      if (!is_visited) {
        parent_nodes[out->id()] = curr_node;
        ready.push_back(out);
      }

      // Process the node 'out'.
      if (IsEnter(out)) {
        if (is_visited) {
          const string& parent_frame = (*info)[out_parent->id()].frame_name;
          if (parent_frame != frame_name) {
            return errors::InvalidArgument(
                "The node '", out->name(),
                "' has inputs from different "
                "frames. The input '",
                curr_node->name(), "' is in frame '", frame_name,
                "'. The input '", parent_nodes[out->id()]->name(),
                "' is in frame '", parent_frame, "'.");
          }
        } else {
          out_info->frame = out;
          out_info->parent_frame = frame;
          TF_RETURN_IF_ERROR(
              GetNodeAttr(out->def(), "frame_name", &out_info->frame_name));
          if (out_info->frame_name.empty()) {
            return errors::InvalidArgument("The Enter node ", out->name(),
                                           " must have a frame name.");
          }
        }
      } else {
        if (is_visited) {
          if (out_info->frame_name != frame_name) {
            return errors::InvalidArgument(
                "The node '", out->name(),
                "' has inputs from different "
                "frames. The input '",
                curr_node->name(), "' is in frame '", frame_name,
                "'. The input '", parent_nodes[out->id()]->name(),
                "' is in frame '", out_info->frame_name, "'.");
          }
        } else {
          out_info->frame = frame;
          out_info->parent_frame = parent;
          out_info->frame_name = frame_name;
        }
      }
    }
  }
  return Status::OK();
}

}  // namespace tensorflow
