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

#include "tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.h"

#include <stddef.h>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
// IWYU pragma: no_include "llvm/IR/Attributes.gen.inc"
// IWYU pragma: no_include "llvm/IR/Intrinsics.gen.inc"
#include "external/llvm/include/llvm/ADT/APInt.h"
#include "external/llvm/include/llvm/IR/BasicBlock.h"
#include "external/llvm/include/llvm/IR/Instructions.h"
#include "external/llvm/include/llvm/IR/Intrinsics.h"
#include "external/llvm/include/llvm/IR/Module.h"
#include "external/llvm/include/llvm/IR/Type.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace xla {
namespace gpu {

using llvm_ir::IrArray;
using llvm_ir::SetToFirstInsertPoint;
using tensorflow::strings::StrAppend;

// Returns whether operand is a floating-point literal with the given value.
bool IsFPLiteralWithValue(const HloInstruction* operand, float value) {
  return operand->opcode() == HloOpcode::kConstant &&
         LiteralUtil::IsAllFloat(operand->literal(), value);
}

GpuElementalIrEmitter::GpuElementalIrEmitter(
    const HloModuleConfig& hlo_module_config, llvm::Module* module,
    llvm::IRBuilder<>* ir_builder, NestedComputer compute_nested)
    : ElementalIrEmitter(hlo_module_config, module, ir_builder),
      hlo_module_config_(hlo_module_config),
      compute_nested_(std::move(compute_nested)) {}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitLibdeviceMathCall(
    const string& callee_name,
    tensorflow::gtl::ArraySlice<llvm::Value*> operands,
    tensorflow::gtl::ArraySlice<PrimitiveType> input_types,
    PrimitiveType output_type) const {
  // The libdevice math functions differentiate between "double" and "float" by
  // appending an 'f' to the function's name.
  string munged_callee = callee_name;
  switch (output_type) {
    case F32:
      StrAppend(&munged_callee, "f");
      break;
    case F64:
      break;
    default:
      return Unimplemented("Bad type for libdevice math call: %s",
                           PrimitiveType_Name(output_type).c_str());
  }
  return EmitMathCall(munged_callee, operands, input_types, output_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitLlvmIntrinsicMathCall(
    const string& callee_name,
    tensorflow::gtl::ArraySlice<llvm::Value*> operands,
    tensorflow::gtl::ArraySlice<PrimitiveType> input_types,
    PrimitiveType output_type) const {
  // llvm intrinsics differentiate between float/double functions via the ".f32"
  // and ".f64" suffixes.
  string munged_callee = callee_name;
  switch (output_type) {
    case F32:
      StrAppend(&munged_callee, ".f32");
      break;
    case F64:
      StrAppend(&munged_callee, ".f64");
      break;
    default:
      return Unimplemented("Bad type for llvm intrinsic math call: %s",
                           PrimitiveType_Name(output_type).c_str());
  }
  return EmitMathCall(munged_callee, operands, input_types, output_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitMathCall(
    const string& callee_name,
    tensorflow::gtl::ArraySlice<llvm::Value*> operands,
    tensorflow::gtl::ArraySlice<PrimitiveType> input_types,
    PrimitiveType output_type) const {
  // Binary math functions tranform are of type [T] -> T.
  for (PrimitiveType input_type : input_types) {
    if (output_type != input_type) {
      return Unimplemented("Input type ≠ output type: %s ≠ %s",
                           PrimitiveType_Name(input_type).c_str(),
                           PrimitiveType_Name(output_type).c_str());
    }
  }

  return EmitDeviceFunctionCall(
      callee_name, operands, input_types, output_type,
      {llvm::Attribute::ReadNone, llvm::Attribute::NoUnwind});
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitFloatBinaryOp(
    const HloInstruction* op, llvm::Value* lhs_value,
    llvm::Value* rhs_value) const {
  PrimitiveType lhs_input_type = op->operand(0)->shape().element_type();
  PrimitiveType rhs_input_type = op->operand(1)->shape().element_type();
  PrimitiveType output_type = op->shape().element_type();
  switch (op->opcode()) {
    case HloOpcode::kRemainder: {
      return EmitLibdeviceMathCall("__nv_fmod", {lhs_value, rhs_value},
                                   {lhs_input_type, rhs_input_type},
                                   output_type);
    }
    case HloOpcode::kPower: {
      return EmitPowerOp(op, lhs_value, rhs_value);
    }
    default:
      return ElementalIrEmitter::EmitFloatBinaryOp(op, lhs_value, rhs_value);
  }
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitPowerOp(
    const HloInstruction* op, llvm::Value* lhs_value,
    llvm::Value* rhs_value) const {
  CHECK_EQ(op->opcode(), HloOpcode::kPower);
  PrimitiveType lhs_input_type = op->operand(0)->shape().element_type();
  PrimitiveType rhs_input_type = op->operand(1)->shape().element_type();
  PrimitiveType output_type = op->shape().element_type();
  llvm::Type* llvm_ty = lhs_value->getType();

  auto make_sqrt = [&, this]() -> StatusOr<llvm::Value*> {
    // NVPTX has four relevant square root instructions:
    //   sqrt.approx{.ftz}.f32
    //   sqrt.rn{.ftz}.f32
    //   sqrt.rn.f64
    //   rsqrt.approx.f64
    // We rely on LLVM's NVPTX backend to pick the right one based on our
    // fast-math options.  (If fast-math is enabled, llvm may compute the 64-bit
    // sqrt from the rsqrt approximation.)
    return EmitLlvmIntrinsicMathCall("llvm.sqrt", {lhs_value}, {lhs_input_type},
                                     output_type);
  };

  const HloInstruction* rhs = op->operand(1);
  if (IsFPLiteralWithValue(rhs, .5)) {
    VLOG(10) << "emitting pow(A, .5) as sqrt(A): " << op->ToString();
    return make_sqrt();
  }

  if (!hlo_module_config_.fast_math_disabled() &&
      IsFPLiteralWithValue(rhs, -.5)) {
    VLOG(10) << "emitting pow(A, -.5) as 1/sqrt(A): " << op->ToString();
    // LLVM's NVPTX backend knows how to transform 1/sqrt(A) into the NVPTX
    // rsqrt.approx instruction.
    TF_ASSIGN_OR_RETURN(auto* sqrt, make_sqrt());
    return ir_builder_->CreateFDiv(llvm::ConstantFP::get(llvm_ty, 1), sqrt);
  }

  VLOG(10) << "emitting pow as regular call to pow(): " << op->ToString();
  return EmitLibdeviceMathCall("__nv_pow", {lhs_value, rhs_value},
                               {lhs_input_type, rhs_input_type}, output_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitErfcInv(
    PrimitiveType prim_type, llvm::Value* value) const {
  return EmitLibdeviceMathCall("__nv_erfcinv", {value}, {prim_type}, prim_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitFloatUnaryOp(
    const HloInstruction* op, llvm::Value* operand_value) const {
  PrimitiveType input_type = op->operand(0)->shape().element_type();
  PrimitiveType output_type = op->shape().element_type();
  switch (op->opcode()) {
    case HloOpcode::kExp:
      return EmitLibdeviceMathCall("__nv_exp", {operand_value}, {input_type},
                                   output_type);
    case HloOpcode::kFloor:
      return EmitLibdeviceMathCall("__nv_floor", {operand_value}, {input_type},
                                   output_type);
    case HloOpcode::kCeil:
      return EmitLibdeviceMathCall("__nv_ceil", {operand_value}, {input_type},
                                   output_type);
    case HloOpcode::kLog:
      return EmitLibdeviceMathCall("__nv_log", {operand_value}, {input_type},
                                   output_type);
    case HloOpcode::kTanh:
      return EmitLibdeviceMathCall("__nv_tanh", {operand_value}, {input_type},
                                   output_type);
    default:
      return ElementalIrEmitter::EmitFloatUnaryOp(op, operand_value);
  }
}

llvm::Value* GpuElementalIrEmitter::EmitDeviceFunctionCall(
    const string& callee_name,
    tensorflow::gtl::ArraySlice<llvm::Value*> operands,
    tensorflow::gtl::ArraySlice<PrimitiveType> input_types,
    PrimitiveType output_type,
    tensorflow::gtl::ArraySlice<llvm::Attribute::AttrKind> attributes) const {
  std::vector<llvm::Type*> ir_input_types;
  for (PrimitiveType input_type : input_types) {
    ir_input_types.push_back(
        llvm_ir::PrimitiveTypeToIrType(input_type, ir_builder_));
  }
  llvm::FunctionType* callee_type = llvm::FunctionType::get(
      llvm_ir::PrimitiveTypeToIrType(output_type,
                                     ir_builder_),  // The return type.
      ir_input_types,                               // The parameter types.
      false);                                       // No variadic arguments.

  // Declares the callee if it is not declared already.
  llvm::Function* callee = llvm::cast<llvm::Function>(
      ir_builder_->GetInsertBlock()->getModule()->getOrInsertFunction(
          llvm_ir::AsStringRef(callee_name), callee_type));

  for (auto attribute : attributes) {
    callee->addFnAttr(attribute);
  }

  return ir_builder_->CreateCall(callee, llvm_ir::AsArrayRef(operands));
}

llvm::Value* GpuElementalIrEmitter::EmitThreadId() const {
  llvm::Value* block_id = ir_builder_->CreateIntCast(
      llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x,
                                   {}, {}, ir_builder_),
      ir_builder_->getIntNTy(128), /*isSigned=*/true, "block.id");
  llvm::Value* thread_id_in_block = ir_builder_->CreateIntCast(
      llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x,
                                   {}, {}, ir_builder_),
      ir_builder_->getIntNTy(128), /*isSigned=*/true, "thread.id");
  llvm::Value* threads_per_block = ir_builder_->CreateIntCast(
      llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x,
                                   {}, {}, ir_builder_),
      ir_builder_->getIntNTy(128), /*isSigned=*/true, "threads_per_block");
  return ir_builder_->CreateNSWAdd(
      ir_builder_->CreateNSWMul(block_id, threads_per_block),
      thread_id_in_block);
}

llvm_ir::ElementGenerator GpuElementalIrEmitter::MakeElementGenerator(
    const HloInstruction* hlo,
    const HloToElementGeneratorMap& operand_to_generator) const {
  switch (hlo->opcode()) {
    case HloOpcode::kPad:
      return [=, &operand_to_generator](
                 const IrArray::Index& padded_index) -> StatusOr<llvm::Value*> {
        auto index = padded_index;
        llvm::Value* in_bounds =
            llvm::ConstantInt::get(ir_builder_->getInt1Ty(), 1);
        for (int i = 0; i < index.size(); ++i) {
          auto index_typed_const = [=](int64 n) {
            return llvm::ConstantInt::get(index[i]->getType(), n);
          };
          const auto& pad_dim = hlo->padding_config().dimensions(i);
          index[i] = ir_builder_->CreateSub(
              index[i], index_typed_const(pad_dim.edge_padding_low()));
          in_bounds = ir_builder_->CreateAnd(
              in_bounds,
              ir_builder_->CreateICmpSGE(index[i], index_typed_const(0)),
              "in_bounds");
          in_bounds = ir_builder_->CreateAnd(
              in_bounds,
              ir_builder_->CreateICmpEQ(
                  index_typed_const(0),
                  ir_builder_->CreateURem(
                      index[i],
                      index_typed_const(pad_dim.interior_padding() + 1))),
              "in_bounds");
          index[i] = ir_builder_->CreateSDiv(
              index[i], index_typed_const(pad_dim.interior_padding() + 1));
          in_bounds = ir_builder_->CreateAnd(
              in_bounds,
              ir_builder_->CreateICmpSLT(
                  index[i],
                  index_typed_const(hlo->operand(0)->shape().dimensions(i))),
              "in_bounds");
        }

        // if (in_bounds) {
        //   ret_value = operand0[index];  // source
        // } else {
        //   ret_value = *operand1;        // padding
        // }
        llvm::Value* ret_value_addr = llvm_ir::EmitAllocaAtFunctionEntry(
            llvm_ir::PrimitiveTypeToIrType(hlo->shape().element_type(),
                                           ir_builder_),
            "pad_result_addr", ir_builder_);
        llvm_ir::LlvmIfData if_data =
            llvm_ir::EmitIfThenElse(in_bounds, "in_bounds", ir_builder_);
        SetToFirstInsertPoint(if_data.true_block, ir_builder_);
        TF_ASSIGN_OR_RETURN(llvm::Value * operand_value,
                            operand_to_generator.at(hlo->operand(0))(index));
        ir_builder_->CreateStore(operand_value, ret_value_addr);

        SetToFirstInsertPoint(if_data.false_block, ir_builder_);
        TF_ASSIGN_OR_RETURN(llvm::Value * padding_value,
                            operand_to_generator.at(hlo->operand(1))({}));
        ir_builder_->CreateStore(padding_value, ret_value_addr);

        SetToFirstInsertPoint(if_data.after_block, ir_builder_);
        // Don't create phi(operand_value, padding_value) here, because invoking
        // operand_to_generator may create new basic blocks, making the parent
        // of operand_value or padding_value no longer a predecessor of
        // if_data.after_block.
        return ir_builder_->CreateLoad(ret_value_addr);
      };
    case HloOpcode::kMap:
      return [=, &operand_to_generator](
                 const IrArray::Index& index) -> StatusOr<llvm::Value*> {
        TF_RET_CHECK(!hlo->operands().empty())
            << "Zero operand map not implemented in GPU backend.";
        TF_RET_CHECK(hlo->to_apply()->num_parameters() > 0);
        std::vector<llvm::Value*> operand_elements;
        for (HloInstruction* operand : hlo->operands()) {
          TF_ASSIGN_OR_RETURN(llvm::Value * value,
                              operand_to_generator.at(operand)(index));
          operand_elements.push_back(value);
        }
        return compute_nested_(*hlo->to_apply(), operand_elements);
      };
    case HloOpcode::kReduceWindow:
      // Pseudocode:
      // for each index I in output
      //   value = init_value
      //   for each index W in window
      //     for each dimension i from 0 to rank - 1
      //       (input index I)[i] = O[i] * stride[i] + W[i] - pad_low[i]
      //     if I in bounds of input
      //       value = function(value, input[I])
      //     output[O] = value
      return [=, &operand_to_generator](
                 const IrArray::Index& index) -> StatusOr<llvm::Value*> {
        const HloInstruction* operand = hlo->operand(0);
        const Window& window = hlo->window();

        // TODO(b/31410564): Implement dilation for reduce-window.
        if (window_util::HasDilation(window)) {
          return Unimplemented(
              "Dilation for reduce-window not implemented on GPU. "
              "See b/31410564.");
        }

        PrimitiveType operand_element_type = operand->shape().element_type();
        llvm::Value* accum_ptr = llvm_ir::EmitAllocaAtFunctionEntry(
            llvm_ir::PrimitiveTypeToIrType(operand_element_type, ir_builder_),
            "reduce_window_accum_ptr", ir_builder_);
        {
          TF_ASSIGN_OR_RETURN(llvm::Value * init_value,
                              operand_to_generator.at(hlo->operand(1))({}));
          ir_builder_->CreateStore(init_value, accum_ptr);
        }

        llvm_ir::ForLoopNest loops(ir_builder_);
        std::vector<int64> window_size;
        for (const auto& dim : window.dimensions()) {
          window_size.push_back(dim.size());
        }
        const IrArray::Index window_index = loops.AddLoopsForShape(
            ShapeUtil::MakeShape(operand_element_type, window_size), "window");
        CHECK_EQ(window_index.size(), index.size());

        SetToFirstInsertPoint(loops.GetInnerLoopBodyBasicBlock(), ir_builder_);

        IrArray::Index input_index(index.size());
        llvm::Value* in_bounds = ir_builder_->getInt1(1);
        for (size_t i = 0; i < index.size(); ++i) {
          llvm::Value* stridden_index = ir_builder_->CreateNSWMul(
              index[i], ir_builder_->getInt64(window.dimensions(i).stride()));
          input_index[i] = ir_builder_->CreateNSWSub(
              ir_builder_->CreateNSWAdd(stridden_index, window_index[i]),
              ir_builder_->getInt64(window.dimensions(i).padding_low()));

          // We must check whether 0 ≤ input_index[i] < bound, as otherwise
          // we are in the pad and so can skip the computation. This
          // comparison is equivalent to the unsigned comparison
          // input_index[i] < bound, as a negative value wraps to a large
          // positive value.
          in_bounds = ir_builder_->CreateAnd(
              in_bounds,
              ir_builder_->CreateICmpULT(
                  input_index[i],
                  ir_builder_->getInt64(operand->shape().dimensions(i))));
        }

        llvm_ir::LlvmIfData if_data =
            llvm_ir::EmitIfThenElse(in_bounds, "in_bounds", ir_builder_);
        SetToFirstInsertPoint(if_data.true_block, ir_builder_);

        // We are not in pad, so do the computation.
        TF_ASSIGN_OR_RETURN(llvm::Value * input_value,
                            operand_to_generator.at(operand)(input_index));
        TF_ASSIGN_OR_RETURN(
            llvm::Value * accum_value,
            compute_nested_(*hlo->to_apply(),
                            {ir_builder_->CreateLoad(accum_ptr), input_value}));
        ir_builder_->CreateStore(accum_value, accum_ptr);

        SetToFirstInsertPoint(loops.GetOuterLoopExitBasicBlock(), ir_builder_);
        return ir_builder_->CreateLoad(accum_ptr);
      };
    case HloOpcode::kReduce:
      return [=, &operand_to_generator](
                 const IrArray::Index& output_index) -> StatusOr<llvm::Value*> {
        const HloInstruction* operand = hlo->operand(0);
        llvm::Value* accum_ptr =
            ir_builder()->CreateAlloca(llvm_ir::PrimitiveTypeToIrType(
                hlo->shape().element_type(), ir_builder()));
        TF_ASSIGN_OR_RETURN(llvm::Value * init_value,
                            operand_to_generator.at(hlo->operand(1))({}));
        ir_builder()->CreateStore(init_value, accum_ptr);

        llvm_ir::ForLoopNest loops(ir_builder_);
        IrArray::Index input_index = loops.AddLoopsForShapeOnDimensions(
            operand->shape(), hlo->dimensions(), "reduction_dim");
        if (!ShapeUtil::IsScalar(hlo->shape())) {
          // Here only input_index[hlo->dimensions()] are non-null, so we must
          // set the rest.
          size_t j = 0;
          for (size_t i = 0; i < input_index.size(); ++i) {
            if (input_index[i] == nullptr) {
              input_index[i] = output_index[j++];
            }
          }
          CHECK_EQ(output_index.size(), j);
        }

        SetToFirstInsertPoint(loops.GetInnerLoopBodyBasicBlock(), ir_builder());
        TF_ASSIGN_OR_RETURN(
            llvm::Value * input_value,
            operand_to_generator.at(hlo->operand(0))(input_index));
        TF_ASSIGN_OR_RETURN(
            llvm::Value * accum_value,
            compute_nested_(
                *hlo->to_apply(),
                {ir_builder()->CreateLoad(accum_ptr), input_value}));
        ir_builder()->CreateStore(accum_value, accum_ptr);
        SetToFirstInsertPoint(loops.GetOuterLoopExitBasicBlock(), ir_builder());
        return ir_builder()->CreateLoad(accum_ptr);
      };
    default:
      return ElementalIrEmitter::MakeElementGenerator(hlo,
                                                      operand_to_generator);
  }
}

}  // namespace gpu
}  // namespace xla
