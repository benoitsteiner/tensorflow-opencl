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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_GRAPH_DUMPER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_GRAPH_DUMPER_H_

#include <string>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace hlo_graph_dumper {

// Dumps a graph of the computation to the GraphViz server and returns
// a description of the rendered graph (e.g., a URL).
string DumpGraph(const HloComputation& computation, const string& label,
                 bool show_addresses, bool show_layouts,
                 const HloExecutionProfile* hlo_execution_profile = nullptr);

// Dumps the HloModule::ToString() as a file into the provided directory path
// suffixed with the provided label.
//
// If do_prefix is true, a timestamp will be prepended onto the label to
// construct a filename in the directory path; otherwise, the label is used
// as the filename directly.
void DumpText(const HloModule& module, const string& label,
              const string& directory_path, bool do_prefix = true);

// Abstract interface for classes that render DOT graphs.
class GraphRendererInterface {
 public:
  virtual ~GraphRendererInterface() = default;

  // Renders a DOT graph, returning a description of the rendered output
  // (e.g., a URL)
  virtual string RenderGraph(const string& graph) = 0;
};

// Graph renderers may be added using a registration mechanism, e.g.:
// XLA_REGISTER_GRAPH_RENDERER(AGraphRendererClass, 100)
// The renderer with the highest numeric priority value is used.

#define XLA_REGISTER_GRAPH_RENDERER(factory, ...) \
  XLA_INTERNAL_REGISTER_GRAPH_RENDERER(factory, __COUNTER__, ##__VA_ARGS__)

// Internal implementation details below this point.

// Class that registers a graph renderer. Higher-priority renders are chosen
// first.
class Registrar {
 public:
  Registrar(GraphRendererInterface* dumper, int priority);
};

#define XLA_INTERNAL_REGISTER_GRAPH_RENDERER(factory, ctr, ...)   \
  static ::xla::hlo_graph_dumper::Registrar                       \
      XLA_INTERNAL_REGISTER_GRAPH_RENDERER_NAME(ctr)(new factory, \
                                                     ##__VA_ARGS__)

// __COUNTER__ must go through another macro to be properly expanded
#define XLA_INTERNAL_REGISTER_GRAPH_RENDERER_NAME(ctr) ___##ctr##__object_

}  // namespace hlo_graph_dumper
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_GRAPH_DUMPER_H_
