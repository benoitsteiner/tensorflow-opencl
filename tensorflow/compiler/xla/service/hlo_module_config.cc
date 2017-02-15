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

#include "tensorflow/compiler/xla/service/hlo_module_config.h"

#include <atomic>
#include <vector>

#include "tensorflow/compiler/xla/shape_layout.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace xla {

using tensorflow::strings::StrAppend;

HloModuleConfig::HloModuleConfig(const ProgramShape& program_shape)
    : entry_computation_layout_(program_shape) {}

string HloModuleConfig::compilation_cache_key() const {
  string key = tensorflow::strings::StrCat("profiling=", hlo_profiling_enabled_,
                                           "::hybrid=", has_hybrid_result_);
  StrAppend(&key, "::(");
  std::vector<string> params;
  for (const ShapeLayout& param_layout :
       entry_computation_layout_.parameter_layouts()) {
    params.push_back(param_layout.shape().DebugString());
  }
  StrAppend(&key, tensorflow::str_util::Join(params, ", "), ") => ",
            entry_computation_layout_.result_shape().SerializeAsString());
  if (seed() != 0) {
    // TODO(b/32083678): force recompilation to reset global state.
    static std::atomic<int> counter{0};
    StrAppend(&key, "forcing recompile ", counter++);
  }
  if (replica_count() != 1) {
    StrAppend(&key, "::replica_count=", replica_count());
  }
  StrAppend(&key, "::fast_math_disabled=", fast_math_disabled_);
  return key;
}

}  // namespace xla
