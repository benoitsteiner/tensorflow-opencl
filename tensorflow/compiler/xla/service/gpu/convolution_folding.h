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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CONVOLUTION_FOLDING_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CONVOLUTION_FOLDING_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass.h"

namespace xla {
namespace gpu {

class ConvolutionFolding : public HloPass {
 public:
  ConvolutionFolding() : HloPass("convolution-folding") {}
  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CONVOLUTION_FOLDING_H_
