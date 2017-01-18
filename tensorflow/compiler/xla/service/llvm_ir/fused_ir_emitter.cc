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

#include "tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.h"

#include <functional>

#include "external/llvm/include/llvm/IR/BasicBlock.h"
#include "external/llvm/include/llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/elemental_ir_emitter.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ops.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

using llvm_ir::IrArray;

Status FusedIrEmitter::DefaultAction(HloInstruction* hlo) {
  generators_[hlo] =
      [=](const IrArray::Index& index) -> StatusOr<llvm::Value*> {
    if (generated_value_cache_[hlo].count(index.multidim()) > 0) {
      llvm::Value* generated_value =
          generated_value_cache_[hlo][index.multidim()];
      llvm::BasicBlock* generated_value_bb = nullptr;
      if (auto* generated_instruction =
              llvm::dyn_cast<llvm::Instruction>(generated_value)) {
        generated_value_bb = generated_instruction->getParent();
      }
      // Ideally, we should be able to reuse the cached generated value if it
      // dominates the current insertion block. However, the check for dominance
      // can be expensive and unreliable when the function is being constructed.
      //
      // It's also worth experimenting what if we don't do caching at all.
      // LLVM's CSE or GVN should be able to easily merge common subexpressions
      // that would be regenerated without caching. But this might increase the
      // JIT compilation time.
      if (generated_value_bb == nullptr ||
          generated_value_bb == ir_builder_->GetInsertBlock()) {
        VLOG(3) << "The cached generated value is reused.";
        return generated_value;
      }
      VLOG(3) << "The cached generated value can't be reuse, because it is at "
                 "a different BB ("
              << llvm_ir::AsString(generated_value_bb->getName())
              << ") from the current insertion block ("
              << llvm_ir::AsString(ir_builder_->GetInsertBlock()->getName())
              << ").";
    }

    TF_ASSIGN_OR_RETURN(
        generated_value_cache_[hlo][index.multidim()],
        elemental_emitter_->MakeElementGenerator(hlo, generators_)(index));
    return generated_value_cache_[hlo][index.multidim()];
  };
  return Status::OK();
}

Status FusedIrEmitter::HandleConstant(HloInstruction* constant,
                                      const Literal& literal) {
  llvm::Constant* initializer =
      llvm_ir::ConvertLiteralToIrConstant(literal, ir_builder_);
  llvm::GlobalVariable* global = new llvm::GlobalVariable(
      *ir_builder_->GetInsertBlock()->getModule(), initializer->getType(),
      /*isConstant=*/true, llvm::GlobalValue::ExternalLinkage, initializer,
      /*Name=*/"");
  generators_[constant] = [=](const IrArray::Index& index) {
    return IrArray(global, constant->shape())
        .EmitReadArrayElement(index, ir_builder_);
  };

  return Status::OK();
}

Status FusedIrEmitter::HandleGetTupleElement(HloInstruction* get_tuple_element,
                                             HloInstruction* operand) {
  // Lookup ir value for 'operand'.
  auto it = gte_values_.find(operand);
  if (it == gte_values_.end()) {
    return Unimplemented(
        "GetTupleElement fusion currently only supports"
        " parameter operands, but found operand: %s",
        operand->name().c_str());
  }
  // Emit code to lookup tuple element pointer, and store it in 'gte_values_'.
  llvm::Value* tuple_element_ptr = llvm_ir::EmitGetTupleElement(
      get_tuple_element->shape(), get_tuple_element->tuple_index(),
      /*alignment=*/1, it->second, ir_builder_);
  gte_values_.insert(std::make_pair(get_tuple_element, tuple_element_ptr));
  // Emit code to read base tuple element array (if non-tuple shaped).
  if (!ShapeUtil::IsTuple(get_tuple_element->shape())) {
    generators_[get_tuple_element] =
        [=](const IrArray::Index& index) -> StatusOr<llvm::Value*> {
      // TODO(b/34080002) Add aliasing information to tuple element IrArray.
      return IrArray(tuple_element_ptr, get_tuple_element->shape())
          .EmitReadArrayElement(index, ir_builder_);
    };
  }
  return Status::OK();
}

Status FusedIrEmitter::HandleParameter(HloInstruction* parameter) {
  generators_[parameter] = [=](const IrArray::Index& index) {
    return parameter_arrays_[parameter->parameter_number()]
        .EmitReadArrayElement(index, ir_builder_);
  };
  // Store ir value for fusion operand associated with fusion parameter to be
  // accessed by subsequent fused GetTupleElement instructions.
  gte_values_.insert(std::make_pair(
      parameter,
      parameter_arrays_[parameter->parameter_number()].GetBasePointer()));
  return Status::OK();
}

Status FusedIrEmitter::FinishVisit(HloInstruction* root) {
  fused_root_ = root;
  return tensorflow::Status::OK();
}

FusedIrEmitter::Generator FusedIrEmitter::GetRootGenerator() const {
  CHECK_NE(nullptr, fused_root_)
      << "GetRootGenerator should be called after Accept.";
  return generators_.at(fused_root_);
}

FusedIrEmitter::Generator FusedIrEmitter::GetGenerator(
    const HloInstruction* instruction) const {
  return generators_.at(instruction);
}

}  // namespace xla
