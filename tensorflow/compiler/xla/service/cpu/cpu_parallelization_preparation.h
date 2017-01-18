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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_PARALLELIZATION_PREPARATION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_PARALLELIZATION_PREPARATION_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass.h"

namespace xla {
namespace cpu {

// This pass prepares an HLO module for parallel execution by transforming
// subgraphs of the top-level computation into embedded computations which can
// be executed in parallel.
// TODO(b/29630486): Currently, it is limited to turning all instructions (which
// are not constants or parameters) in the entry computation into embedded
// computations.  However, it could make sense to coarsen the parallelization to
// improve cache locality.  Also, we will need to do something to intelligently
// handle While constructs.
class ParallelizationPreparation : public HloPass {
 public:
  explicit ParallelizationPreparation() : HloPass("cpu-parallel-prepare") {}
  ~ParallelizationPreparation() override {}

  // Run instruction fusion on the given computation. Returns whether the
  // computation was changed.
  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_PARALLELIZATION_PREPARATION_H_
