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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_ELEMENTAL_IR_EMITTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_ELEMENTAL_IR_EMITTER_H_

#include <functional>
#include <string>
#include <utility>

#include "external/llvm/include/llvm/IR/IRBuilder.h"
#include "external/llvm/include/llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/elemental_ir_emitter.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/llvm_ir/loop_emitter.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace xla {
namespace gpu {

class GpuElementalIrEmitter : public ElementalIrEmitter {
 public:
  // A NestedComputer computes an element of the output of the given computation
  // given an ArraySlice of its input elements.
  using NestedComputer = std::function<StatusOr<llvm::Value*>(
      const HloComputation&, tensorflow::gtl::ArraySlice<llvm::Value*>)>;

  GpuElementalIrEmitter(const HloModuleConfig& hlo_module_config,
                        llvm::Module* module, llvm::IRBuilder<>* ir_builder,
                        NestedComputer compute_nested);

  llvm_ir::ElementGenerator MakeElementGenerator(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator) const override;

 protected:
  StatusOr<llvm::Value*> EmitFloatUnaryOp(
      const HloInstruction* op, llvm::Value* operand_value) const override;

  StatusOr<llvm::Value*> EmitFloatBinaryOp(
      const HloInstruction* op, llvm::Value* lhs_value,
      llvm::Value* rhs_value) const override;

  StatusOr<llvm::Value*> EmitErfcInv(PrimitiveType prim_type,
                                     llvm::Value* value) const override;

  llvm::Value* EmitThreadId() const override;

 private:
  // Emit IR to call a device function named "callee_name" on the given
  // operand. Returns the IR value that represents the return value.
  llvm::Value* EmitDeviceFunctionCall(
      const string& callee_name,
      tensorflow::gtl::ArraySlice<llvm::Value*> operands,
      tensorflow::gtl::ArraySlice<PrimitiveType> input_type,
      PrimitiveType output_type,
      tensorflow::gtl::ArraySlice<llvm::Attribute::AttrKind> attributes) const;

  // Emit IR to call a device function of type [T] -> T.  It adjusts the
  // callee_name to account for float/double types.
  // Returns the IR value that represents the return value.
  StatusOr<llvm::Value*> EmitMathCall(
      const string& callee_name,
      tensorflow::gtl::ArraySlice<llvm::Value*> operands,
      tensorflow::gtl::ArraySlice<PrimitiveType> input_types,
      PrimitiveType output_type) const;

  NestedComputer compute_nested_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_ELEMENTAL_IR_EMITTER_H_
