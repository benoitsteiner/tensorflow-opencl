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

#include "tensorflow/compiler/xla/service/hlo_instruction.h"

#include <algorithm>
#include <deque>
#include <set>
#include <unordered_set>
#include <utility>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateParameter(
    int64 parameter_number, const Shape& shape, const string& name) {
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kParameter, shape));
  instruction->parameter_number_ = parameter_number;
  instruction->parameter_name_ = name;
  instruction->name_ = "%" + name;
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateTrace(
    const string& tag, HloInstruction* operand) {
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kTrace, ShapeUtil::MakeNil()));
  instruction->operands_.push_back(operand);
  instruction->literal_.reset(new Literal);
  *instruction->literal_->mutable_u8s() += tag;
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateConstant(
    std::unique_ptr<Literal> literal) {
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kConstant, literal->shape()));
  instruction->literal_ = std::move(literal);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateGetTupleElement(const Shape& shape,
                                      HloInstruction* operand, int64 index) {
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kGetTupleElement, shape));
  instruction->tuple_index_ = index;
  instruction->AppendOperand(operand);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateRng(
    const Shape& shape, RandomDistribution distribution,
    tensorflow::gtl::ArraySlice<HloInstruction*> parameters) {
  auto instruction = WrapUnique(new HloInstruction(HloOpcode::kRng, shape));
  instruction->distribution_ = distribution;
  instruction->shape_ = shape;
  for (HloInstruction* param : parameters) {
    instruction->AppendOperand(param);
  }
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateNary(
    const Shape& shape, HloOpcode opcode,
    tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
  if (opcode == HloOpcode::kCopy) {
    // It is impossible to copy an opaque shape, we don't know how big it is.
    CHECK(!ShapeUtil::IsOpaque(shape));
  }
  auto instruction = WrapUnique(new HloInstruction(opcode, shape));
  for (auto operand : operands) {
    instruction->AppendOperand(operand);
  }
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateUnary(
    const Shape& shape, HloOpcode opcode, HloInstruction* operand) {
  // Only certain opcodes are supported with CreateUnary: opcodes of unary
  // instructions with no auxiliary fields.
  switch (opcode) {
    case HloOpcode::kAbs:
    case HloOpcode::kBitcast:
    case HloOpcode::kCeil:
    case HloOpcode::kCopy:
    case HloOpcode::kExp:
    case HloOpcode::kFloor:
    case HloOpcode::kLog:
    case HloOpcode::kLogicalNot:
    case HloOpcode::kNegate:
    case HloOpcode::kSign:
    case HloOpcode::kSort:
    case HloOpcode::kTanh:
      break;
    default:
      LOG(FATAL) << "Invalid unary instruction opcode "
                 << HloOpcodeString(opcode);
  }
  return CreateNary(shape, opcode, {operand});
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateBinary(
    const Shape& shape, HloOpcode opcode, HloInstruction* lhs,
    HloInstruction* rhs) {
  // Only certain opcodes are supported with CreateBinary: opcodes of binary
  // instructions with no auxiliary fields.
  switch (opcode) {
    case (HloOpcode::kAdd):
    case (HloOpcode::kDivide):
    case (HloOpcode::kDot):
    case (HloOpcode::kEq):
    case (HloOpcode::kGe):
    case (HloOpcode::kGt):
    case (HloOpcode::kLe):
    case (HloOpcode::kLt):
    case (HloOpcode::kMaximum):
    case (HloOpcode::kMinimum):
    case (HloOpcode::kMultiply):
    case (HloOpcode::kNe):
    case (HloOpcode::kPower):
    case (HloOpcode::kRemainder):
    case (HloOpcode::kSubtract):
    case (HloOpcode::kLogicalAnd):
    case (HloOpcode::kLogicalOr):
      break;
    default:
      LOG(FATAL) << "Invalid binary instruction opcode "
                 << HloOpcodeString(opcode);
  }
  return CreateNary(shape, opcode, {lhs, rhs});
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateTernary(
    const Shape& shape, HloOpcode opcode, HloInstruction* lhs,
    HloInstruction* rhs, HloInstruction* ehs) {
  // Only certain opcodes are supported with CreateTernary: opcodes of ternary
  // instructions with no auxiliary fields.
  switch (opcode) {
    case (HloOpcode::kClamp):
    case (HloOpcode::kSelect):
      break;
    default:
      LOG(FATAL) << "Invalid ternary instruction opcode "
                 << HloOpcodeString(opcode);
  }
  return CreateNary(shape, opcode, {lhs, rhs, ehs});
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateVariadic(
    const Shape& shape, HloOpcode opcode,
    tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
  CHECK_EQ(HloOpcode::kTuple, opcode);
  return CreateNary(shape, opcode, operands);
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateMap(
    const Shape& shape, tensorflow::gtl::ArraySlice<HloInstruction*> operands,
    HloComputation* map_computation,
    tensorflow::gtl::ArraySlice<HloInstruction*> static_operands) {
  CHECK(static_operands.empty()) << "static_operands not yet supported";
  auto instruction = WrapUnique(new HloInstruction(HloOpcode::kMap, shape));
  for (auto operand : operands) {
    instruction->AppendOperand(operand);
  }
  instruction->to_apply_ = map_computation;
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateConvolve(
    const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
    const Window& window,
    const ConvolutionDimensionNumbers& dimension_numbers) {
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kConvolution, shape));
  instruction->AppendOperand(lhs);
  instruction->AppendOperand(rhs);
  instruction->window_ = MakeUnique<Window>(window);
  instruction->convolution_dimension_numbers_ =
      MakeUnique<ConvolutionDimensionNumbers>(dimension_numbers);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateCrossReplicaSum(const Shape& shape,
                                      HloInstruction* operand) {
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kCrossReplicaSum, shape));
  instruction->AppendOperand(operand);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateInfeed(
    const Shape& shape) {
  return WrapUnique(new HloInstruction(HloOpcode::kInfeed, shape));
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateSend(
    HloInstruction* operand) {
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kSend, ShapeUtil::MakeNil()));
  instruction->AppendOperand(operand);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateRecv(
    const Shape& shape) {
  return WrapUnique(new HloInstruction(HloOpcode::kRecv, shape));
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateReverse(
    const Shape& shape, HloInstruction* operand,
    tensorflow::gtl::ArraySlice<int64> dimensions) {
  auto instruction = WrapUnique(new HloInstruction(HloOpcode::kReverse, shape));
  instruction->AppendOperand(operand);
  instruction->dimensions_.assign(dimensions.begin(), dimensions.end());
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateWhile(
    const Shape& shape, HloComputation* condition, HloComputation* body,
    HloInstruction* init) {
  auto instruction = WrapUnique(new HloInstruction(HloOpcode::kWhile, shape));
  instruction->AppendOperand(init);
  instruction->condition_ = condition;
  instruction->body_ = body;
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateSlice(
    const Shape& shape, HloInstruction* operand,
    tensorflow::gtl::ArraySlice<int64> start_indices,
    tensorflow::gtl::ArraySlice<int64> limit_indices) {
  auto instruction = WrapUnique(new HloInstruction(HloOpcode::kSlice, shape));
  instruction->AppendOperand(operand);
  instruction->slice_starts_.assign(start_indices.begin(), start_indices.end());
  instruction->slice_limits_.assign(limit_indices.begin(), limit_indices.end());
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateDynamicSlice(
    const Shape& shape, HloInstruction* operand, HloInstruction* start_indices,
    tensorflow::gtl::ArraySlice<int64> slice_sizes) {
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kDynamicSlice, shape));
  instruction->AppendOperand(operand);
  instruction->AppendOperand(start_indices);
  instruction->dynamic_slice_sizes_.assign(slice_sizes.begin(),
                                           slice_sizes.end());
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateDynamicUpdateSlice(const Shape& shape,
                                         HloInstruction* operand,
                                         HloInstruction* update,
                                         HloInstruction* start_indices) {
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kDynamicUpdateSlice, shape));
  instruction->AppendOperand(operand);
  instruction->AppendOperand(update);
  instruction->AppendOperand(start_indices);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateConcatenate(
    const Shape& shape, tensorflow::gtl::ArraySlice<HloInstruction*> operands,
    int64 dimension) {
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kConcatenate, shape));
  for (auto operand : operands) {
    instruction->AppendOperand(operand);
  }
  instruction->dimensions_.push_back(dimension);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateConvert(
    const Shape& shape, HloInstruction* operand) {
  auto instruction = WrapUnique(new HloInstruction(HloOpcode::kConvert, shape));
  instruction->AppendOperand(operand);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateReduce(
    const Shape& shape, HloInstruction* arg, HloInstruction* init_value,
    tensorflow::gtl::ArraySlice<int64> dimensions_to_reduce,
    HloComputation* reduce_computation) {
  auto instruction = WrapUnique(new HloInstruction(HloOpcode::kReduce, shape));
  instruction->AppendOperand(arg);
  instruction->AppendOperand(init_value);
  instruction->dimensions_.assign(dimensions_to_reduce.begin(),
                                  dimensions_to_reduce.end());
  instruction->to_apply_ = reduce_computation;
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateReduceWindow(
    const Shape& shape, HloInstruction* operand, HloInstruction* init_value,
    const Window& window, HloComputation* reduce_computation) {
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kReduceWindow, shape));
  instruction->AppendOperand(operand);
  instruction->AppendOperand(init_value);
  instruction->to_apply_ = reduce_computation;
  instruction->window_ = MakeUnique<Window>(window);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateSelectAndScatter(
    const Shape& shape, HloInstruction* operand, HloComputation* select,
    const Window& window, HloInstruction* source, HloInstruction* init_value,
    HloComputation* scatter) {
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kSelectAndScatter, shape));
  instruction->AppendOperand(operand);
  instruction->AppendOperand(source);
  instruction->AppendOperand(init_value);
  instruction->select_ = select;
  instruction->scatter_ = scatter;
  instruction->window_ = MakeUnique<Window>(window);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateBroadcast(
    const Shape& shape, HloInstruction* operand,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kBroadcast, shape));
  instruction->AppendOperand(operand);
  instruction->dimensions_.assign(broadcast_dimensions.begin(),
                                  broadcast_dimensions.end());
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreatePad(
    const Shape& shape, HloInstruction* operand, HloInstruction* padding_value,
    const PaddingConfig& padding_config) {
  auto instruction = WrapUnique(new HloInstruction(HloOpcode::kPad, shape));
  instruction->AppendOperand(operand);
  instruction->AppendOperand(padding_value);
  instruction->padding_config_ = MakeUnique<PaddingConfig>(padding_config);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateReshape(
    const Shape& shape, HloInstruction* operand) {
  CHECK_EQ(ShapeUtil::ElementsIn(shape),
           ShapeUtil::ElementsIn(operand->shape()));
  auto instruction = WrapUnique(new HloInstruction(HloOpcode::kReshape, shape));
  instruction->AppendOperand(operand);
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateTranspose(
    const Shape& shape, HloInstruction* operand,
    tensorflow::gtl::ArraySlice<int64> dimensions) {
  CHECK_EQ(shape.dimensions().size(), dimensions.size());
  CHECK_EQ(shape.dimensions().size(), operand->shape().dimensions().size());
  CHECK(std::equal(operand->shape().dimensions().begin(),
                   operand->shape().dimensions().end(),
                   Permute(dimensions, shape.dimensions()).begin()));
  auto instruction =
      WrapUnique(new HloInstruction(HloOpcode::kTranspose, shape));
  instruction->AppendOperand(operand);
  instruction->dimensions_.assign(dimensions.begin(), dimensions.end());
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateFusion(
    const Shape& shape, FusionKind fusion_kind, HloInstruction* fused_root) {
  auto instruction = WrapUnique(new HloInstruction(HloOpcode::kFusion, shape));
  instruction->fusion_kind_ = fusion_kind;
  instruction->CloneAndFuseInternal(fused_root);
  instruction->CheckFusionInstruction();
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction>
HloInstruction::CreateFusionForBackwardConvolution(
    const Shape& shape, FusionKind fusion_kind, const Window& window,
    const ConvolutionDimensionNumbers& conv_dnums, HloInstruction* fused_root) {
  std::unique_ptr<HloInstruction> fusion =
      CreateFusion(shape, fusion_kind, fused_root);
  fusion->window_ = MakeUnique<Window>(window);
  fusion->convolution_dimension_numbers_ =
      MakeUnique<ConvolutionDimensionNumbers>(conv_dnums);
  return fusion;
}

HloInstruction* HloInstruction::FuseInstruction(
    HloInstruction* instruction_to_fuse) {
  CHECK_EQ(opcode_, HloOpcode::kFusion);

  // This fusion instruction must be a user of instruction_to_fuse.
  CHECK_NE(0, instruction_to_fuse->users().count(this));
  HloInstruction* fused_instruction = CloneAndFuseInternal(instruction_to_fuse);
  CheckFusionInstruction();
  return fused_instruction;
}

HloInstruction* HloInstruction::CloneAndFuseInternal(
    HloInstruction* instruction_to_fuse) {
  CHECK_EQ(opcode_, HloOpcode::kFusion);
  CHECK(instruction_to_fuse->IsFusable());

  bool new_fusion_instruction = fused_instructions_.empty();
  fused_instructions_.emplace_back(instruction_to_fuse->Clone());
  HloInstruction* clone = fused_instructions_.back().get();
  clone->parent_fusion_instruction_ = this;

  if (new_fusion_instruction) {
    fused_root_ = clone;
  } else {
    // instruction_to_fuse is necessarily an operand of the fusion instruction.
    // After fusion this will no longer be the case. Remove the operand from the
    // operand list and remove its corresponding fused parameter
    // instruction. Renumber parameters as necessary to make parameter numbers
    // consistent with their index in the fused_parameter_ vector.
    CHECK(std::find(operands_.begin(), operands_.end(), instruction_to_fuse) !=
          operands_.end());
    for (int64 operand_num = 0; operand_num < operand_count(); ++operand_num) {
      if (instruction_to_fuse == operands_[operand_num]) {
        // replace the fused parameter instruction's uses with the clone.
        HloInstruction* fused_parameter = fused_parameters_[operand_num];
        fused_parameter->ReplaceAllUsesWith(clone);

        // Remove the corresponding fused parameter and operand from their
        // respective vectors.
        fused_parameters_.erase(fused_parameters_.begin() + operand_num);
        operands_.erase(operands_.begin() + operand_num);

        // Renumber fused parameter numbers to match the vector index.
        while (operand_num < fused_parameters_.size()) {
          fused_parameters_[operand_num]->parameter_number_ = operand_num;
          operand_num++;
        }
        // Throw removed fused parameter instruction away.
        auto inst_it =
            std::find_if(fused_instructions_.begin(), fused_instructions_.end(),
                         [=](const std::unique_ptr<HloInstruction>& inst) {
                           return inst.get() == fused_parameter;
                         });
        CHECK(inst_it != fused_instructions_.end());
        fused_instructions_.erase(inst_it);
        break;
      }
    }
    // We've cloned instruction_to_fuse into this fusion instruction, so this
    // fusion instruction is no longer a use of instruction_to_fuse.
    instruction_to_fuse->RemoveUser(this);
  }

  // Add each operand of the clone as an operand of the fusion instruction. A
  // complication is that some clone operands may already be operands of the
  // fusion instruction.
  for (int64 operand_num = 0; operand_num < clone->operand_count();
       ++operand_num) {
    HloInstruction* operand = clone->mutable_operand(operand_num);

    // See if this operand is already an operand of the fusion node.
    CHECK_EQ(operands_.size(), fused_parameters_.size());
    HloInstruction* fused_param = nullptr;
    for (int64 i = 0; i < operands_.size(); ++i) {
      if (operands_[i] == operand) {
        fused_param = fused_parameters_[i];
        break;
      }
    }

    if (fused_param == nullptr) {
      // Clone's operand was not already an operand of the fusion
      // instruction. Add it as an operand and add a corresponding fused
      // parameter instruction.
      int64 param_no = fused_parameters_.size();
      std::unique_ptr<HloInstruction> param_instruction =
          CreateParameter(param_no, operand->shape(), "fusion_param");

      param_instruction->parent_fusion_instruction_ = this;
      fused_parameters_.push_back(param_instruction.get());
      fused_instructions_.push_back(std::move(param_instruction));
      AppendOperand(operand);

      fused_param = fused_instructions_.back().get();
    }
    clone->ReplaceOperandWith(operand_num, fused_param);
  }

  return clone;
}

RandomDistribution HloInstruction::random_distribution() const {
  CHECK_EQ(opcode_, HloOpcode::kRng);
  return distribution_;
}

namespace {

// Adds any HloComputations this instruction calls directly to the given set.
void CalledComputationsInternal(
    const HloInstruction& instruction,
    std::set<HloComputation*>* called_computations) {
  switch (instruction.opcode()) {
    case HloOpcode::kCall:
    case HloOpcode::kMap:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
      called_computations->insert(instruction.to_apply());
      break;
    case HloOpcode::kSelectAndScatter:
      called_computations->insert(instruction.select());
      called_computations->insert(instruction.scatter());
      break;
    case HloOpcode::kWhile:
      called_computations->insert(instruction.while_condition());
      called_computations->insert(instruction.while_body());
      break;
    case HloOpcode::kFusion:
      for (const auto& fused_instruction : instruction.fused_instructions()) {
        CalledComputationsInternal(*fused_instruction, called_computations);
      }
      break;
    default:
      break;
  }
}

}  // namespace

std::set<HloComputation*> HloInstruction::MakeCalledComputationsSet() const {
  std::set<HloComputation*> called_computations;
  CalledComputationsInternal(*this, &called_computations);
  return called_computations;
}

void HloInstruction::CheckFusionInstruction() const {
  CHECK_EQ(opcode_, HloOpcode::kFusion);

  // All instructions owned by this fusion instruction must be fused, and the
  // parent fusion instruction of the fused instructions must be 'this'.
  for (auto& instruction : fused_instructions_) {
    CHECK(instruction->IsFused());
    CHECK_EQ(this, instruction->fusion_instruction());
  }

  // Fused root instruction and fused parameters must all be owned by the fusion
  // instruction.
  bool root_owned = false;
  std::vector<bool> parameter_owned(fused_parameters_.size(), false);
  for (auto& instruction : fused_instructions_) {
    if (fused_root_ == instruction.get()) {
      CHECK(!root_owned);
      root_owned = true;
    }
    for (int i = 0; i < fused_parameters_.size(); ++i) {
      if (fused_parameters_[i] == instruction.get()) {
        CHECK(!parameter_owned[i]);
        parameter_owned[i] = true;
      }
    }
  }
  CHECK(root_owned);
  // Make sure all the parameter_owned entries are set
  for (int i = 0; i < parameter_owned.size(); i++) {
    CHECK(parameter_owned[i]);
  }

  // Fused root must have no users.
  CHECK_EQ(0, fused_root_->user_count());

  // All uses of fused instructions must be in the fusion instruction, and every
  // non-root instruction must have at least one use.
  for (auto& instruction : fused_instructions_) {
    if (instruction.get() != fused_root_) {
      CHECK_GT(instruction->user_count(), 0);
      for (auto& user : instruction->users()) {
        CHECK(user->IsFused());
        CHECK_EQ(this, user->fusion_instruction());
      }
    }
  }

  // Fused parameter instructions must be numbered contiguously and match up
  // (shapes compatible) with their respective operand.
  CHECK_EQ(operands_.size(), fused_parameters_.size());
  std::vector<bool> parameter_numbers(fused_parameters_.size(), false);
  for (auto fused_param : fused_parameters_) {
    int64 param_no = fused_param->parameter_number();
    CHECK_GE(param_no, 0);
    CHECK_LT(param_no, fused_parameters_.size());
    CHECK(!parameter_numbers[param_no]);
    parameter_numbers[param_no] = true;
    CHECK(ShapeUtil::Compatible(fused_param->shape(),
                                operands_[param_no]->shape()));
  }
  // Make sure all the parameter_numbers entries were seen
  for (int i = 0; i < parameter_numbers.size(); i++) {
    CHECK(parameter_numbers[i]);
  }

  // Operands must be distinct.
  std::set<HloInstruction*> operand_set(operands_.begin(), operands_.end());
  CHECK_EQ(operand_set.size(), operands_.size());
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateCall(
    const Shape& shape, tensorflow::gtl::ArraySlice<HloInstruction*> operands,
    HloComputation* computation) {
  std::unique_ptr<HloInstruction> instruction =
      WrapUnique(new HloInstruction(HloOpcode::kCall, shape));
  for (auto operand : operands) {
    instruction->AppendOperand(operand);
  }
  instruction->to_apply_ = computation;
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateCustomCall(
    const Shape& shape, tensorflow::gtl::ArraySlice<HloInstruction*> operands,
    tensorflow::StringPiece custom_call_target) {
  std::unique_ptr<HloInstruction> instruction =
      WrapUnique(new HloInstruction(HloOpcode::kCustomCall, shape));
  for (auto operand : operands) {
    instruction->AppendOperand(operand);
  }
  instruction->custom_call_target_ = custom_call_target.ToString();
  return instruction;
}

/* static */ std::unique_ptr<HloInstruction> HloInstruction::CreateTuple(
    tensorflow::gtl::ArraySlice<HloInstruction*> elements) {
  std::vector<Shape> element_shapes;
  for (auto element : elements) {
    element_shapes.push_back(element->shape());
  }
  Shape tuple_shape = ShapeUtil::MakeTupleShape(element_shapes);
  return CreateVariadic(tuple_shape, HloOpcode::kTuple, elements);
}

std::unique_ptr<HloInstruction> HloInstruction::CloneWithNewOperands(
    const Shape& shape, tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
  // Explicitly call the factory for the instruction type. This is more robust
  // in the face of code changes than copying fields explicitly. This also
  // properly sets the user fields of the operands.
  switch (opcode_) {
    // Unary ops.
    case HloOpcode::kAbs:
    case HloOpcode::kBitcast:
    case HloOpcode::kCeil:
    case HloOpcode::kCopy:
    case HloOpcode::kExp:
    case HloOpcode::kFloor:
    case HloOpcode::kLog:
    case HloOpcode::kLogicalNot:
    case HloOpcode::kNegate:
    case HloOpcode::kSign:
    case HloOpcode::kSort:
    case HloOpcode::kTanh:
      CHECK_EQ(operands.size(), 1);
      return CreateUnary(shape, opcode_, operands[0]);
    // Binary ops.
    case HloOpcode::kAdd:
    case HloOpcode::kDivide:
    case HloOpcode::kMultiply:
    case HloOpcode::kSubtract:
    case HloOpcode::kEq:
    case HloOpcode::kGe:
    case HloOpcode::kGt:
    case HloOpcode::kLe:
    case HloOpcode::kLt:
    case HloOpcode::kNe:
    case HloOpcode::kDot:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kLogicalAnd:
    case HloOpcode::kLogicalOr:
      CHECK_EQ(operands.size(), 2);
      return CreateBinary(shape, opcode_, operands[0], operands[1]);
    // Ternary ops.
    case HloOpcode::kClamp:
    case HloOpcode::kSelect:
      CHECK_EQ(operands.size(), 3);
      return CreateTernary(shape, opcode_, operands[0], operands[1],
                           operands[2]);
    // Other supported ops.
    case HloOpcode::kBroadcast:
      CHECK_EQ(operands.size(), 1);
      return CreateBroadcast(shape, operands[0], dimensions_);
    case HloOpcode::kCall:
      return CreateCall(shape, operands, to_apply_);
    case HloOpcode::kCustomCall:
      return CreateCustomCall(shape, operands, custom_call_target_);
    case HloOpcode::kConcatenate:
      return CreateConcatenate(shape, operands, dimensions(0));
    case HloOpcode::kConvert:
      CHECK_EQ(operands.size(), 1);
      return CreateConvert(shape, operands[0]);
    case HloOpcode::kConvolution:
      CHECK_EQ(operands.size(), 2);
      return CreateConvolve(shape, operands[0], operands[1], *window_,
                            *convolution_dimension_numbers_);
    case HloOpcode::kCrossReplicaSum:
      CHECK_EQ(operands.size(), 1);
      return CreateCrossReplicaSum(shape, operands[0]);
    case HloOpcode::kGetTupleElement:
      CHECK_EQ(operands.size(), 1);
      return CreateGetTupleElement(shape, operands[0], tuple_index());
    case HloOpcode::kMap:
      return CreateMap(shape, operands, to_apply_);
    case HloOpcode::kPad:
      CHECK_EQ(operands.size(), 2);
      return CreatePad(shape, operands[0], operands[1], *padding_config_);
    case HloOpcode::kReduce:
      CHECK_EQ(operands.size(), 2);
      return CreateReduce(shape, operands[0], operands[1], dimensions_,
                          to_apply_);
    case HloOpcode::kReduceWindow:
      CHECK_EQ(operands.size(), 2);
      return CreateReduceWindow(shape, operands[0], operands[1], *window_,
                                to_apply_);
    case HloOpcode::kSelectAndScatter:
      CHECK_EQ(operands.size(), 3);
      return CreateSelectAndScatter(shape, operands[0], select_, *window_,
                                    operands[1], operands[2], scatter_);
    case HloOpcode::kRecv:
      CHECK_EQ(operands.size(), 0);
      return CreateRecv(shape);
    case HloOpcode::kReverse:
      CHECK_EQ(operands.size(), 1);
      return CreateReverse(shape, operands[0], dimensions_);
    case HloOpcode::kRng:
      return CreateRng(shape, distribution_, operands);
    case HloOpcode::kReshape:
      CHECK_EQ(operands.size(), 1);
      return CreateReshape(shape, operands[0]);
    case HloOpcode::kSend:
      CHECK_EQ(operands.size(), 1);
      return CreateSend(operands[0]);
    case HloOpcode::kSlice:
      CHECK_EQ(operands.size(), 1);
      return CreateSlice(shape, operands[0], slice_starts_, slice_limits_);
    case HloOpcode::kDynamicSlice:
      return CreateDynamicSlice(shape, operands[0], operands[1],
                                dynamic_slice_sizes_);
    case HloOpcode::kDynamicUpdateSlice:
      CHECK_EQ(operands.size(), 3);
      return CreateDynamicUpdateSlice(shape, operands[0], operands[1],
                                      operands[2]);
    case HloOpcode::kTranspose:
      CHECK_EQ(operands.size(), 1);
      return CreateTranspose(shape, operands[0], dimensions_);
    case HloOpcode::kTuple:
      return CreateTuple(operands_);
    case HloOpcode::kWhile:
      CHECK_EQ(operands.size(), 1);
      return CreateWhile(shape, condition_, body_, operands[0]);
    case HloOpcode::kConstant:
      return CreateConstant(LiteralUtil::CloneToUnique(*literal_));
    case HloOpcode::kFusion:
      return CloneFusionWithNewOperands(shape, operands);
    case HloOpcode::kParameter:
      return CreateParameter(parameter_number_, shape, parameter_name_);
    // Unsupported ops for cloning.
    case HloOpcode::kUpdate:
    case HloOpcode::kIndex:
    case HloOpcode::kInfeed:
    case HloOpcode::kTrace:
      LOG(FATAL) << "Not yet implemented, clone: " << HloOpcodeString(opcode_);
  }
}

std::unique_ptr<HloInstruction> HloInstruction::Clone() {
  std::unique_ptr<HloInstruction> clone =
      CloneWithNewOperands(shape_, operands_);
  clone->name_ = name() + ".clone";
  return clone;
}

std::unique_ptr<HloInstruction> HloInstruction::CloneFusionWithNewOperands(
    const Shape& shape, tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
  CHECK_EQ(opcode_, HloOpcode::kFusion);

  auto new_instruction =
      WrapUnique(new HloInstruction(HloOpcode::kFusion, shape));
  // Add the operands to our new fusion instruction.
  for (HloInstruction* new_operand : operands) {
    new_instruction->AppendOperand(new_operand);
  }
  // Clone all the fused instructions for the new fusion instruction.
  std::map<HloInstruction*, HloInstruction*> old_to_new;
  std::list<std::unique_ptr<HloInstruction>> new_fused_instructions;
  // Create the list of fused parameters by mapping through the cloned,
  // fused instructions.
  std::vector<HloInstruction*> new_fused_parameters;
  for (HloInstruction* old_fused_parameter : fused_parameters_) {
    new_fused_instructions.push_back(old_fused_parameter->Clone());
    HloInstruction* new_fusion_parameter = new_fused_instructions.back().get();
    new_fusion_parameter->parent_fusion_instruction_ = new_instruction.get();
    new_fused_parameters.push_back(new_fusion_parameter);
    InsertOrDie(&old_to_new, old_fused_parameter, new_fusion_parameter);
  }
  for (auto old_fused_instruction_iter = fused_instructions_.rbegin();
       old_fused_instruction_iter != fused_instructions_.rend();
       ++old_fused_instruction_iter) {
    HloInstruction* old_fused_instruction = old_fused_instruction_iter->get();
    if (old_fused_instruction->opcode() == HloOpcode::kParameter) {
      FindOrDie(old_to_new, old_fused_instruction);
      continue;
    }
    std::vector<HloInstruction*> new_operands;
    for (int64 operand_idx = 0;
         operand_idx < old_fused_instruction->operand_count(); ++operand_idx) {
      HloInstruction* old_operand =
          old_fused_instruction->mutable_operand(operand_idx);
      new_operands.push_back(FindOrDie(old_to_new, old_operand));
    }
    new_fused_instructions.push_back(
        old_fused_instruction->CloneWithNewOperands(
            old_fused_instruction->shape(), new_operands));
    HloInstruction* new_fused_instruction = new_fused_instructions.back().get();
    new_fused_instruction->parent_fusion_instruction_ = new_instruction.get();
    InsertOrDie(&old_to_new, old_fused_instruction, new_fused_instruction);
  }
  // We iterated the fusion instructions in reverse post order which means
  // that we must reverse our new list of fusion instructions.
  std::reverse(new_fused_instructions.begin(), new_fused_instructions.end());
  new_instruction->fusion_kind_ = fusion_kind_;
  new_instruction->fused_instructions_ = std::move(new_fused_instructions);
  new_instruction->fused_parameters_ = std::move(new_fused_parameters);
  new_instruction->fused_root_ = FindOrDie(old_to_new, fused_root_);
  new_instruction->CheckFusionInstruction();
  return new_instruction;
}

const Literal& HloInstruction::literal() const {
  CHECK_EQ(HloOpcode::kConstant, opcode_);
  return *literal_;
}

bool HloInstruction::CanHaveDimensionsField() const {
  return (opcode() == HloOpcode::kReverse ||
          opcode() == HloOpcode::kConcatenate ||
          opcode() == HloOpcode::kReduce || opcode() == HloOpcode::kBroadcast ||
          opcode() == HloOpcode::kTranspose);
}

const std::vector<int64>& HloInstruction::dimensions() const {
  CHECK(CanHaveDimensionsField());
  return dimensions_;
}

int64 HloInstruction::dimensions(int64 index) const {
  return dimensions()[index];
}

int64 HloInstruction::concatenate_dimension() const {
  CHECK(opcode() == HloOpcode::kConcatenate);
  CHECK_EQ(1, dimensions_.size());
  return dimensions(0);
}

int64 HloInstruction::tuple_index() const {
  CHECK_EQ(HloOpcode::kGetTupleElement, opcode_);
  return tuple_index_;
}

const HloInstruction* HloInstruction::operand(int64 i) const {
  return operands_[i];
}

HloInstruction* HloInstruction::mutable_operand(int64 i) {
  CHECK(operands_[i] != nullptr);
  return operands_[i];
}

int64 HloInstruction::operand_index(const HloInstruction* target) const {
  for (int64 i = 0; i < operand_count(); ++i) {
    if (target == operand(i)) {
      return i;
    }
  }
  LOG(FATAL) << "target was not an operand";
}

void HloInstruction::AppendOperand(HloInstruction* operand) {
  operands_.push_back(operand);
  operand->AddUser(this);
}

void HloInstruction::AddUser(HloInstruction* user) { users_.insert(user); }

bool HloInstruction::IsConstant() const {
  return opcode_ == HloOpcode::kConstant;
}

bool HloInstruction::HasConstantOperand() const {
  for (const HloInstruction* operand : operands_) {
    if (operand->IsConstant()) {
      return true;
    }
  }
  return false;
}

void HloInstruction::AddControlPredecessor(HloInstruction* instruction) {
  control_predecessors_.insert(instruction);
}

bool HloInstruction::Identical(
    const HloInstruction& other,
    std::function<bool(const HloInstruction*, const HloInstruction*)>
        eq_operands,
    std::function<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  // An instruction is always identical to itself.
  if (this == &other) {
    return true;
  }

  // Identical instruction must have the same opcode and identical operands.  In
  // general, there is no need to check shape because shape is inferred from the
  // shape of the operands.
  if (opcode() != other.opcode() ||
      !ContainersEqual(operands(), other.operands(), eq_operands)) {
    return false;
  }

  // Perform opcode specific checks.
  switch (opcode()) {
    // The result of these instructions only depend upon their opcode and
    // operands.
    case HloOpcode::kAbs:
    case HloOpcode::kAdd:
    case HloOpcode::kCeil:
    case HloOpcode::kClamp:
    case HloOpcode::kCopy:
    case HloOpcode::kCrossReplicaSum:
    case HloOpcode::kDivide:
    case HloOpcode::kDot:
    case HloOpcode::kEq:
    case HloOpcode::kExp:
    case HloOpcode::kFloor:
    case HloOpcode::kGe:
    case HloOpcode::kGt:
    case HloOpcode::kLe:
    case HloOpcode::kLog:
    case HloOpcode::kLogicalAnd:
    case HloOpcode::kLogicalNot:
    case HloOpcode::kLogicalOr:
    case HloOpcode::kLt:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNe:
    case HloOpcode::kNegate:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kSelect:
    case HloOpcode::kSign:
    case HloOpcode::kSubtract:
    case HloOpcode::kTanh:
    case HloOpcode::kTuple:
      return true;

    // These opcodes have complex or special behavior so just return false.
    case HloOpcode::kFusion:
    case HloOpcode::kRng:
    case HloOpcode::kTrace:
    case HloOpcode::kWhile:
      return false;

    case HloOpcode::kParameter:
      return parameter_number() == other.parameter_number() &&
             // Check the shape too because `this` and `other` may be in
             // different HloComputations.
             ShapeUtil::Compatible(shape(), other.shape());

    // A constant is defined by the value in the literal.
    case HloOpcode::kConstant:
      return LiteralUtil::Equal(literal(), other.literal());

    // A convert result is determined by the primitive type that the operand is
    // converted into.
    case HloOpcode::kConvert:
      return shape().element_type() == other.shape().element_type();

    // Convolution has a window and dimensions.
    case HloOpcode::kConvolution:
      return protobuf_util::ProtobufEquals(window(), other.window()) &&
             protobuf_util::ProtobufEquals(
                 convolution_dimension_numbers(),
                 other.convolution_dimension_numbers());

    // Reduction results are determined by the reduction dimension and the
    // reduction computation.
    case HloOpcode::kReduce:
      return dimensions() == other.dimensions() &&
             eq_computations(to_apply(), other.to_apply());
    case HloOpcode::kReduceWindow:
      return eq_computations(to_apply(), other.to_apply()) &&
             protobuf_util::ProtobufEquals(window(), other.window());

    // SelectAndScatter is determined by both select and scatter
    // computation as well as the window configuration.
    case HloOpcode::kSelectAndScatter:
      return eq_computations(select(), other.select()) &&
             eq_computations(scatter(), other.scatter()) &&
             protobuf_util::ProtobufEquals(window(), other.window());

    case HloOpcode::kReshape:
      return ShapeUtil::Compatible(shape(), other.shape());

    // Transpose result is determined by the final shape and the permutation.
    case HloOpcode::kTranspose:
      return ShapeUtil::Compatible(shape(), other.shape()) &&
             dimensions() == other.dimensions();

    // Remaining instructions with special values.
    case HloOpcode::kBitcast:
      return ShapeUtil::Equal(shape(), other.shape());
    case HloOpcode::kBroadcast:
      return ShapeUtil::Compatible(shape(), other.shape()) &&
             dimensions() == other.dimensions();
    case HloOpcode::kConcatenate:
      return dimensions() == other.dimensions();
    case HloOpcode::kGetTupleElement:
      return tuple_index() == other.tuple_index();
    case HloOpcode::kPad:
      return protobuf_util::ProtobufEquals(padding_config(),
                                           other.padding_config());
    case HloOpcode::kSlice:
      return slice_starts_ == other.slice_starts_ &&
             slice_limits_ == other.slice_limits_;
    case HloOpcode::kDynamicSlice:
      return ShapeUtil::Compatible(shape(), other.shape()) &&
             dynamic_slice_sizes_ == other.dynamic_slice_sizes_;
    case HloOpcode::kDynamicUpdateSlice:
      return ShapeUtil::Compatible(shape(), other.shape());
    case HloOpcode::kCall:
    case HloOpcode::kMap:
      return eq_computations(to_apply(), other.to_apply());
    case HloOpcode::kCustomCall:
      return custom_call_target_ == other.custom_call_target_;
    case HloOpcode::kReverse:
      return dimensions() == other.dimensions();

    // These opcodes are not yet supported.
    case HloOpcode::kIndex:
    case HloOpcode::kInfeed:
    case HloOpcode::kSort:
    case HloOpcode::kUpdate:
    case HloOpcode::kSend:
    case HloOpcode::kRecv:
      return false;
  }
}

bool HloInstruction::IsRank2Transpose() const {
  return (opcode_ == HloOpcode::kTranspose) &&
         dimensions_ == std::vector<int64>({1, 0}) &&
         shape_.dimensions_size() == 2 &&
         std::equal(shape_.dimensions().begin(), shape_.dimensions().end(),
                    operands_[0]->shape_.dimensions().rbegin());
}

void HloInstruction::RemoveUser(HloInstruction* user) {
  auto user_it = users_.find(user);
  CHECK(user_it != users_.end());
  users_.erase(user_it);
}

void HloInstruction::ReplaceUseWith(HloInstruction* user,
                                    HloInstruction* new_producer) {
  CHECK(ShapeUtil::Compatible(shape(), new_producer->shape()))
      << "this shape: " << ShapeUtil::HumanString(shape())
      << ", replacement shape: "
      << ShapeUtil::HumanString(new_producer->shape());
  auto user_it = std::find(users_.begin(), users_.end(), user);
  CHECK(user_it != users_.end()) << "Instruction " << user
                                 << " not a use of instruction " << this;
  users_.erase(user_it);

  CHECK_GT(std::count(user->operands_.begin(), user->operands_.end(), this), 0);
  std::replace(user->operands_.begin(), user->operands_.end(), this,
               new_producer);
  new_producer->AddUser(user);
}

void HloInstruction::ReplaceOperandWith(int64 operand_num,
                                        HloInstruction* new_operand) {
  CHECK_GE(operand_num, 0);
  CHECK_LT(operand_num, operand_count());
  HloInstruction* old_operand = mutable_operand(operand_num);
  CHECK(ShapeUtil::Compatible(old_operand->shape(), new_operand->shape()))
      << old_operand->shape().ShortDebugString() << " is not compatible with "
      << new_operand->shape().ShortDebugString();
  operands_[operand_num] = new_operand;

  if (std::find(operands_.begin(), operands_.end(), old_operand) ==
      operands_.end()) {
    old_operand->RemoveUser(this);
  }
  new_operand->AddUser(this);
}

void HloInstruction::ReplaceAllUsesWith(HloInstruction* new_producer) {
  // We can't use range-based loop because the iterator is invalidated by call
  // to ReplaceUseWith.
  for (auto user = users_.begin(); user != users_.end();) {
    auto this_user = user;
    user++;
    // It's possible that new_producer is a user of this instruction as might
    // be the case when replacing an instruction with a kCopy of itself. In
    // this case, don't do the replacement to avoid creating a cycle in the
    // graph.
    if (*this_user != new_producer) {
      ReplaceUseWith(*this_user, new_producer);
    }
  }
}

void HloInstruction::DetachFromOperands() {
  CHECK_EQ(0, user_count());
  // An intruction may be repeated as an operand. To avoid calling RemoveUser
  // twice on the same operand, keep a set of already detached operands.
  std::set<HloInstruction*> detached_operands;
  for (int64 operand_num = 0; operand_num < operand_count(); ++operand_num) {
    HloInstruction* operand = operands_[operand_num];
    if (detached_operands.count(operand) == 0) {
      operand->RemoveUser(this);
      detached_operands.insert(operand);
    }
    operands_[operand_num] = nullptr;
  }
}

HloComputation* HloInstruction::to_apply() const {
  switch (opcode_) {
    case HloOpcode::kCall:
    case HloOpcode::kMap:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kReduce:
      return to_apply_;
    default:
      LOG(FATAL) << "Invalid instruction for to_apply(): " << ToString();
  }
}

void HloInstruction::set_to_apply(HloComputation* computation) {
  switch (opcode_) {
    case HloOpcode::kCall:
    case HloOpcode::kMap:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kReduce:
      to_apply_ = computation;
      break;
    default:
      LOG(FATAL) << "Invalid instruction for to_apply(): " << ToString();
  }
}

const string& HloInstruction::custom_call_target() const {
  CHECK_EQ(opcode_, HloOpcode::kCustomCall);
  return custom_call_target_;
}

HloComputation* HloInstruction::while_condition() const {
  CHECK_EQ(HloOpcode::kWhile, opcode_);
  return condition_;
}

HloComputation* HloInstruction::while_body() const {
  CHECK_EQ(HloOpcode::kWhile, opcode_);
  return body_;
}

void HloInstruction::set_while_condition(HloComputation* computation) {
  CHECK_EQ(HloOpcode::kWhile, opcode_);
  condition_ = computation;
}

void HloInstruction::set_while_body(HloComputation* computation) {
  CHECK_EQ(HloOpcode::kWhile, opcode_);
  body_ = computation;
}

HloComputation* HloInstruction::select() const {
  CHECK_EQ(HloOpcode::kSelectAndScatter, opcode_);
  return select_;
}

HloComputation* HloInstruction::scatter() const {
  CHECK_EQ(HloOpcode::kSelectAndScatter, opcode_);
  return scatter_;
}

void HloInstruction::set_select(HloComputation* computation) {
  CHECK_EQ(HloOpcode::kSelectAndScatter, opcode_);
  select_ = computation;
}

void HloInstruction::set_scatter(HloComputation* computation) {
  CHECK_EQ(HloOpcode::kSelectAndScatter, opcode_);
  scatter_ = computation;
}

string HloInstruction::SignatureString() const {
  string operands = tensorflow::str_util::Join(
      operands_, ", ", [](string* out, HloInstruction* operand) {
        tensorflow::strings::StrAppend(
            out, ShapeUtil::HumanString(operand->shape()));
      });
  return tensorflow::strings::StrCat("(", operands, ") -> ",
                                     ShapeUtil::HumanString(shape()));
}

string HloInstruction::ToString() const {
  string operands;
  if (opcode() == HloOpcode::kConstant) {
    // For constants, emit the actual value in place of an empty operand list.
    if (ShapeUtil::ElementsIn(shape()) <= 10) {
      // LiteralUtil::ToString emits multidimensional arrays over multiple
      // lines. Compact this into one line by stripping out white space.
      string tmp = LiteralUtil::ToString(literal());
      std::replace(tmp.begin(), tmp.end(), '\n', ' ');
      std::vector<string> v = tensorflow::str_util::Split(tmp, ' ');
      bool first = true;
      // Concatenate elements in "v" with spaces separating them, but ignoring
      // empty entries.
      for (const auto& s : v) {
        if (s.empty()) continue;
        tensorflow::strings::StrAppend(&operands, (first ? "" : " "), s);
        first = false;
      }
    } else {
      // Don't try emitting large constants.
      operands = "{...}";
    }
  } else {
    operands = tensorflow::str_util::Join(
        operands_, ", ", [](string* out, HloInstruction* operand) {
          tensorflow::strings::StrAppend(
              out, ShapeUtil::HumanStringWithLayout(operand->shape()), " ",
              operand->name());
        });
  }
  string extra;
  if (LayoutUtil::HasLayout(shape())) {
    if (ShapeUtil::IsTuple(shape())) {
      // Tuple shapes are recursive, so the layout field of the top-level shape
      // does not include all layout information. In this case, print out the
      // entire shape with layout.
      tensorflow::strings::StrAppend(&extra, ", layout=",
                                     ShapeUtil::HumanStringWithLayout(shape()));
    } else {
      tensorflow::strings::StrAppend(
          &extra, tensorflow::strings::Printf(
                      ", layout=%s",
                      LayoutUtil::HumanString(shape().layout()).c_str()));
    }
  }
  if (CanHaveDimensionsField()) {
    tensorflow::strings::StrAppend(
        &extra, ", dimensions={",
        tensorflow::str_util::Join(dimensions(), ", "), "}");
  }
  if (window_ != nullptr) {
    tensorflow::strings::StrAppend(&extra, ", window=",
                                   window_util::ToString(*window_));
  }
  if (padding_config_ != nullptr) {
    tensorflow::strings::StrAppend(&extra, ", padding=",
                                   padding_config_->ShortDebugString());
  }
  if (convolution_dimension_numbers_ != nullptr) {
    tensorflow::strings::StrAppend(
        &extra,
        tensorflow::strings::Printf(
            ", "
            "conv_dim_nums={batch_dim=%lld,feature_dim=%lld,spatial_dims=(%s),"
            "kernel_input_feature_dims=%lld,kernel_output_feature_dim=%lld,"
            "kernel_spatial_dims=(%s)}",
            convolution_dimension_numbers_->batch_dimension(),
            convolution_dimension_numbers_->feature_dimension(),
            tensorflow::str_util::Join(
                convolution_dimension_numbers_->spatial_dimensions(), ",")
                .c_str(),
            convolution_dimension_numbers_->kernel_input_feature_dimension(),
            convolution_dimension_numbers_->kernel_output_feature_dimension(),
            tensorflow::str_util::Join(
                convolution_dimension_numbers_->kernel_spatial_dimensions(),
                ",")
                .c_str()));
  }
  if (to_apply_ != nullptr) {
    tensorflow::strings::StrAppend(&extra, ", computation=", to_apply_->name());
  }
  if (opcode() == HloOpcode::kWhile) {
    tensorflow::strings::StrAppend(&extra, ", condition=",
                                   while_condition()->name());
    tensorflow::strings::StrAppend(&extra, ", body=", while_body()->name());
  }
  if (opcode() == HloOpcode::kGetTupleElement) {
    tensorflow::strings::StrAppend(&extra, ", index=", tuple_index());
  }
  return tensorflow::strings::Printf(
      "%s %s = %s(%s)%s", ShapeUtil::HumanString(shape()).c_str(),
      name().c_str(), HloOpcodeString(opcode()).c_str(), operands.c_str(),
      extra.c_str());
}

string HloInstruction::ToShortString() const {
  return tensorflow::strings::Printf(
      "%s = %s(%s)", name().c_str(), HloOpcodeString(opcode()).c_str(),
      tensorflow::str_util::Join(operands_, ", ", [](string* out,
                                                     HloInstruction* operand) {
        tensorflow::strings::StrAppend(out, operand->name());
      }).c_str());
}

HloInstruction* HloInstruction::tracing() const { return trace_instruction_; }

void HloInstruction::set_tracing(HloInstruction* trace_instruction) {
  trace_instruction_ = trace_instruction;
}

const string& HloInstruction::tracing_tag() const {
  CHECK_EQ(HloOpcode::kTrace, opcode());
  CHECK(literal_ != nullptr);
  return literal_->u8s();
}

bool HloInstruction::IsFused() const {
  return parent_fusion_instruction_ != nullptr;
}

bool HloInstruction::IsFusable() const {
  // Instructions which are traced should not be fused.
  if (tracing()) {
    return false;
  }

  // Some kinds of instructions don't make sense to fuse.
  switch (opcode_) {
    case HloOpcode::kFusion:
    case HloOpcode::kInfeed:
    case HloOpcode::kParameter:
    case HloOpcode::kTrace:
    case HloOpcode::kSend:
    case HloOpcode::kRecv:
      return false;
    default:
      return true;
  }
}

HloInstruction* HloInstruction::fusion_instruction() const {
  CHECK(IsFused());
  return parent_fusion_instruction_;
}

HloInstruction* HloInstruction::fused_expression_root() const {
  CHECK_EQ(opcode_, HloOpcode::kFusion);
  return fused_root_;
}

HloInstruction* HloInstruction::fused_parameter(int64 parameter_number) const {
  CHECK_EQ(opcode_, HloOpcode::kFusion);
  CHECK_GE(parameter_number, 0);
  CHECK_LT(parameter_number, fused_parameters_.size());
  return fused_parameters_[parameter_number];
}

const std::list<std::unique_ptr<HloInstruction>>&
HloInstruction::fused_instructions() const {
  CHECK_EQ(opcode_, HloOpcode::kFusion);
  return fused_instructions_;
}

HloInstruction::HloInstruction(HloOpcode opcode, const Shape& shape)
    : shape_(shape), opcode_(opcode), name_("%" + HloOpcodeString(opcode)) {
  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(shape_));
}

Status HloInstruction::AcceptInternalVisit(DfsHloVisitor* visitor) {
  switch (opcode_) {
    case HloOpcode::kAbs:
      return visitor->HandleAbs(this, operands_[0]);
    case HloOpcode::kSign:
      return visitor->HandleSign(this, operands_[0]);
    case HloOpcode::kConstant:
      return visitor->HandleConstant(this, *literal_);
    case HloOpcode::kGetTupleElement:
      return visitor->HandleGetTupleElement(this, operands_[0]);
    case HloOpcode::kParameter:
      return visitor->HandleParameter(this);
    case HloOpcode::kEq:
    case HloOpcode::kGe:
    case HloOpcode::kGt:
    case HloOpcode::kLe:
    case HloOpcode::kLt:
    case HloOpcode::kNe:
      return visitor->HandleCompare(this, opcode_, operands_[0], operands_[1]);
    case HloOpcode::kAdd:
      return visitor->HandleAdd(this, operands_[0], operands_[1]);
    case HloOpcode::kDivide:
      return visitor->HandleDivide(this, operands_[0], operands_[1]);
    case HloOpcode::kSubtract:
      return visitor->HandleSubtract(this, operands_[0], operands_[1]);
    case HloOpcode::kMaximum:
      return visitor->HandleMaximum(this, operands_[0], operands_[1]);
    case HloOpcode::kMinimum:
      return visitor->HandleMinimum(this, operands_[0], operands_[1]);
    case HloOpcode::kLogicalAnd:
      return visitor->HandleLogicalAnd(this, operands_[0], operands_[1]);
    case HloOpcode::kLogicalOr:
      return visitor->HandleLogicalOr(this, operands_[0], operands_[1]);
    case HloOpcode::kConcatenate:
      return visitor->HandleConcatenate(this, operands_);
    case HloOpcode::kConvert:
      return visitor->HandleConvert(this, operands_[0]);
    case HloOpcode::kCopy:
      return visitor->HandleCopy(this, operands_[0]);
    case HloOpcode::kMultiply:
      return visitor->HandleMultiply(this, operands_[0], operands_[1]);
    case HloOpcode::kDot:
      return visitor->HandleDot(this, operands_[0], operands_[1]);
    case HloOpcode::kPower:
      return visitor->HandlePower(this, operands_[0], operands_[1]);
    case HloOpcode::kRemainder:
      return visitor->HandleRemainder(this, operands_[0], operands_[1]);
    case HloOpcode::kSelect:
      return visitor->HandleSelect(this, operands_[0], operands_[1],
                                   operands_[2]);
    case HloOpcode::kConvolution:
      return visitor->HandleConvolution(this, operands_[0], operands_[1],
                                        window());
    case HloOpcode::kCrossReplicaSum:
      return visitor->HandleCrossReplicaSum(this);
    case HloOpcode::kTuple:
      return visitor->HandleTuple(this, operands_);
    case HloOpcode::kMap:
      return visitor->HandleMap(this, operands_, to_apply_, {});
    case HloOpcode::kClamp:
      return visitor->HandleClamp(this, operands_[0], operands_[1],
                                  operands_[2]);
    case HloOpcode::kReduce:
      return visitor->HandleReduce(this, operands_[0], operands_[1],
                                   dimensions_, to_apply_);
    case HloOpcode::kReduceWindow:
      return visitor->HandleReduceWindow(this, operands_[0], window(),
                                         to_apply_);
    case HloOpcode::kSelectAndScatter:
      return visitor->HandleSelectAndScatter(this);
    case HloOpcode::kNegate:
      return visitor->HandleNegate(this, operands_[0]);
    case HloOpcode::kExp:
      return visitor->HandleExp(this, operands_[0]);
    case HloOpcode::kFloor:
      return visitor->HandleFloor(this, operands_[0]);
    case HloOpcode::kCeil:
      return visitor->HandleCeil(this, operands_[0]);
    case HloOpcode::kLog:
      return visitor->HandleLog(this, operands_[0]);
    case HloOpcode::kTanh:
      return visitor->HandleTanh(this, operands_[0]);
    case HloOpcode::kLogicalNot:
      return visitor->HandleLogicalNot(this, operands_[0]);
    case HloOpcode::kBitcast:
      return visitor->HandleBitcast(this);
    case HloOpcode::kBroadcast:
      return visitor->HandleBroadcast(this);
    case HloOpcode::kPad:
      return visitor->HandlePad(this);
    case HloOpcode::kReshape:
      return visitor->HandleReshape(this);
    case HloOpcode::kTranspose:
      return visitor->HandleTranspose(this);
    case HloOpcode::kReverse:
      return visitor->HandleReverse(this, operands_[0]);
    case HloOpcode::kSlice:
      return visitor->HandleSlice(this, operands_[0]);
    case HloOpcode::kDynamicSlice:
      return visitor->HandleDynamicSlice(this, operands_);
    case HloOpcode::kDynamicUpdateSlice:
      return visitor->HandleDynamicUpdateSlice(this, operands_[0], operands_[1],
                                               operands_[2]);
    case HloOpcode::kSort:
      return visitor->HandleSort(this, operands_[0]);
    case HloOpcode::kInfeed:
      return visitor->HandleInfeed(this);
    case HloOpcode::kRng:
      return visitor->HandleRng(this, distribution_);
    case HloOpcode::kWhile:
      return visitor->HandleWhile(this, operands_[0], condition_, body_);
    case HloOpcode::kFusion:
      return visitor->HandleFusion(this);
    case HloOpcode::kCall:
      return visitor->HandleCall(this, operands_, to_apply_);
    case HloOpcode::kCustomCall:
      return visitor->HandleCustomCall(this, operands_, custom_call_target_);
    case HloOpcode::kSend:
      return visitor->HandleSend(this);
    case HloOpcode::kRecv:
      return visitor->HandleRecv(this);

    // These opcodes are not handled here.
    case HloOpcode::kIndex:
    case HloOpcode::kTrace:
    case HloOpcode::kUpdate:
      break;
  }
  return Unimplemented("unhandled HloOpcode for DfsHloVisitor: %s",
                       HloOpcodeString(opcode_).c_str());
}

Status HloInstruction::AcceptInternal(DfsHloVisitor* visitor) {
  // Do not visit this HLO node again if it is already visited.
  if (visitor->DidVisit(*this)) {
    VLOG(3) << "Not visiting HLO " << this << " as it was already visited.";
    return Status::OK();
  }

  // If the instruction is in the visiting state, it means a cycle.
  if (visitor->IsVisiting(*this)) {
    return FailedPrecondition(
        "A cycle is detected while visiting instruction %s",
        ToString().c_str());
  }
  visitor->SetVisiting(*this);

  for (auto operand : operands_) {
    VLOG(3) << "Going to visit HLO " << operand << " as operand of HLO "
            << this;
    TF_RETURN_IF_ERROR(operand->AcceptInternal(visitor));
  }

  for (auto control_predecessor : control_predecessors_) {
    VLOG(3) << "Going to visit HLO " << control_predecessor
            << " as a control predecessor of HLO " << this;
    TF_RETURN_IF_ERROR(control_predecessor->AcceptInternal(visitor));
  }

  TF_RETURN_IF_ERROR(visitor->Preprocess(this));
  VLOG(3) << "Visiting HLO " << this;
  TF_RETURN_IF_ERROR(AcceptInternalVisit(visitor));
  visitor->SetVisited(*this);
  return visitor->Postprocess(this);
}

Status HloInstruction::Accept(DfsHloVisitor* visitor, bool call_finish_visit) {
  auto status = AcceptInternal(visitor);
  if (!status.ok()) {
    return status;
  }

  if (call_finish_visit) {
    return visitor->FinishVisit(this);
  } else {
    return Status::OK();
  }
}

namespace {

// Returns true if the given order is a topological sort of exactly those
// instructions rooted at 'root'.
bool OrderIsTopologicalSort(HloInstruction* root,
                            const std::vector<const HloInstruction*>& order) {
  // Create a map from instruction to its position in 'order'.
  std::unordered_map<const HloInstruction*, int> order_position;
  for (int i = 0; i < order.size(); i++) {
    if (!order_position.insert(std::make_pair(order[i], i)).second) {
      // Instruction order[i] is duplicated in the order.
      return false;
    }
  }
  // Verify that the operand of each instruction in the order is also in the
  // order *and* the operand's position is earlier (defs are before uses for all
  // ops).
  for (auto* instruction : order) {
    for (auto* operand : instruction->operands()) {
      if (order_position.count(operand) == 0 ||
          order_position.at(operand) >= order_position.at(instruction)) {
        return false;
      }
    }
  }

  // Create a vector of all instructions in a DFS search starting at
  // root. 'order' should contain exactly these instructions.
  std::vector<const HloInstruction*> visited;
  TF_CHECK_OK(root->Accept([&visited](HloInstruction* instruction) {
    visited.push_back(instruction);
    return Status::OK();
  }));

  if (order_position.size() != visited.size()) {
    return false;
  }
  for (auto* instruction : visited) {
    if (order_position.count(instruction) == 0) {
      return false;
    }
  }
  // Given the conditions above, the last element of order should always be the
  // root.
  CHECK_EQ(root, order[order.size() - 1]);

  return true;
}

}  // namespace

Status HloInstruction::Accept(FunctionVisitor::VisitorFunction visitor_func) {
  FunctionVisitor visitor(visitor_func);
  return this->Accept(&visitor);
}

Status HloInstruction::AcceptOrdered(
    DfsHloVisitor* visitor, const std::vector<const HloInstruction*>& order) {
  DCHECK(OrderIsTopologicalSort(this, order));
  for (auto* const_instruction : order) {
    // The visitor can mark instructions as visited to skip particular
    // instructions.
    if (visitor->DidVisit(*const_instruction)) {
      VLOG(3) << "Not visiting HLO " << const_instruction
              << " as it was already visited.";
      continue;
    }

    HloInstruction* instruction =
        const_cast<HloInstruction*>(const_instruction);

    TF_RETURN_IF_ERROR(visitor->Preprocess(instruction));
    VLOG(3) << "Visiting HLO " << instruction;
    TF_RETURN_IF_ERROR(instruction->AcceptInternalVisit(visitor));
    visitor->SetVisited(*instruction);
    TF_RETURN_IF_ERROR(visitor->Postprocess(instruction));
  }

  return visitor->FinishVisit(this);
}

const Shape& HloInstruction::shape() const {
  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(shape_));
  return shape_;
}

std::vector<int64> HloInstruction::OperandIndices(
    const HloInstruction* operand) const {
  std::vector<int64> result;
  for (int64 i = 0; i < operand_count(); ++i) {
    if (this->operand(i) == operand) {
      result.push_back(i);
    }
  }
  return result;
}

bool HloInstruction::IsElementwise() const {
  switch (opcode_) {
    // Nullary elementwise operations.
    case HloOpcode::kConstant:
      return true;

    // Unary elementwise operations.
    case HloOpcode::kAbs:
    case HloOpcode::kCeil:
    case HloOpcode::kConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kExp:
    case HloOpcode::kFloor:
    case HloOpcode::kLog:
    case HloOpcode::kLogicalNot:
    case HloOpcode::kNegate:
    case HloOpcode::kSign:
    case HloOpcode::kTanh:
      return true;

    // Binary elementwise operations.
    case HloOpcode::kAdd:
    case HloOpcode::kDivide:
    case HloOpcode::kEq:
    case HloOpcode::kGe:
    case HloOpcode::kGt:
    case HloOpcode::kLe:
    case HloOpcode::kLt:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNe:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kSubtract:
    case HloOpcode::kLogicalAnd:
    case HloOpcode::kLogicalOr:
      return true;

    // Ternary elementwise operations.
    case HloOpcode::kSelect:
      return !ShapeUtil::IsTuple(shape_);
    case HloOpcode::kClamp:
      return true;

    // Other operations.
    case HloOpcode::kMap:
      return true;
    case HloOpcode::kFusion:
      if (fusion_kind() != FusionKind::kLoop) {
        return false;
      }
      for (auto& fused : fused_instructions()) {
        if (fused->opcode() != HloOpcode::kParameter &&
            !fused->IsElementwise()) {
          return false;
        }
      }
      return true;

    default:
      return false;
  }
}

namespace {
bool IsInstructionElementwiseOnOperand(const HloInstruction* instruction,
                                       const HloInstruction* operand) {
  std::vector<int64> operand_indices = instruction->OperandIndices(operand);
  return std::all_of(
      operand_indices.begin(), operand_indices.end(),
      [instruction](int64 operand_index) {
        return instruction->IsElementwiseOnOperand(operand_index);
      });
}
}  // namespace

bool HloInstruction::IsElementwiseOnOperand(int64 operand_idx) const {
  // For all instructions other than kFusion, being elementwise on one of the
  // operands is equivalent to being elementwise on all the operands.
  if (opcode() != HloOpcode::kFusion) {
    return IsElementwise();
  }

  CHECK_EQ(HloOpcode::kFusion, opcode());
  if (fusion_kind() != FusionKind::kLoop) {
    return false;
  }

  // A loop-fusion is elementwise on an operand if all operations (computed
  // using BFS) between the operand and the fused root are elementwise.
  std::deque<HloInstruction*> worklist;
  std::unordered_set<const HloInstruction*> visited;
  worklist.push_back(fused_parameter(operand_idx));
  visited.insert(fused_parameter(operand_idx));
  while (!worklist.empty()) {
    HloInstruction* operand = worklist.front();
    worklist.pop_front();
    for (HloInstruction* user : operand->users()) {
      if (visited.count(user)) {
        continue;
      }
      if (user->IsElementwise() ||
          IsInstructionElementwiseOnOperand(user, operand)) {
        worklist.push_back(user);
        visited.insert(user);
      } else {
        return false;
      }
    }
  }
  return true;
}

HloInstruction::UseKind HloInstruction::OperandElementUse(int64 i) const {
  switch (opcode_) {
    case HloOpcode::kBitcast:
    case HloOpcode::kConcatenate:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
      return UseKind::kUsePermutingElements;
    case HloOpcode::kPad:
    case HloOpcode::kReduce:
      // Pad reuses the padding value but not the padded array elements.
      // Reduce reuses the init value but not the operand array elements.
      return i > 0 ? UseKind::kReuse : UseKind::kUsePermutingElements;
    case HloOpcode::kFusion: {
      tensorflow::gtl::FlatMap<const HloInstruction*, UseKind> cache;
      // We could rather iterate backwards thru fused_instructions_ here, as it
      // is in reverse postorder, and compute whether each fused instruction
      // reuses the value of this parameter, which would save stack space but
      // not allow us to finish early if we find a reuse.
      std::function<UseKind(const HloInstruction&)> reuses_parameter_elements =
          [i, &cache, &reuses_parameter_elements](const HloInstruction& hlo) {
            auto plus = [](const UseKind& a, const UseKind& b) {
              if (a == UseKind::kNoUse) return b;
              if (b == UseKind::kNoUse) return a;
              if (a == UseKind::kReuse || b == UseKind::kReuse) {
                return UseKind::kReuse;
              }
              if (a == UseKind::kUsePermutingElements ||
                  b == UseKind::kUsePermutingElements) {
                return UseKind::kReuse;
              }
              CHECK(UseKind::kUse == a && UseKind::kUse == b);
              return UseKind::kUse;
            };

            if (hlo.opcode_ == HloOpcode::kParameter &&
                hlo.parameter_number_ == i) {
              return UseKind::kUse;
            }
            if (cache.count(&hlo) == 0) {
              for (int64 j = 0; j < hlo.operands_.size(); ++j) {
                UseKind old = cache[&hlo];
                UseKind updated = plus(
                    old, std::min(hlo.OperandElementUse(j),
                                  reuses_parameter_elements(*hlo.operand(j))));
                cache[&hlo] = updated;
              }
            }
            return cache[&hlo];
          };
      return reuses_parameter_elements(*fused_root_);
    }
    default:
      return IsElementwise() ? UseKind::kUse : UseKind::kReuse;
  }
}

namespace {

// Prereq: `order` is a permutation of {0, 1, ..., `dims.size()-1`}
void Strip1SizedDimensions(tensorflow::protobuf::RepeatedField<int64>* dims,
                           std::vector<int64>* order) {
  // We can't merely call StripDegenerateDimensions here as we must also delete
  // the dimension indices.
  for (size_t i = 0; i < dims->size(); ++i) {
    if (1 == dims->Get(i)) {
      dims->erase(dims->begin() + i);
      // We must find this, as order must be a permutation of operand
      // dimensions.
      order->erase(std::find(order->begin(), order->end(), i));
    }
  }
}

}  // namespace

std::tuple<bool, std::vector<int64>, std::vector<int64>>
HloInstruction::ReshapeMerelyInsertsOrDeletes1SizedDimensions() const {
  if (HloOpcode::kReshape != opcode_) {
    return std::make_tuple(false, std::vector<int64>(), std::vector<int64>());
  }
  return ShapeUtil::InsertedOrDeleted1SizedDimensions(operand(0)->shape_,
                                                      shape_);
}

string FusionKindString(HloInstruction::FusionKind kind) {
  switch (kind) {
    case HloInstruction::FusionKind::kLoop:
      return "Loop";
    case HloInstruction::FusionKind::kInput:
      return "Input";
    case HloInstruction::FusionKind::kTransposeDot:
      return "TransposeDot";
    case HloInstruction::FusionKind::kConvBackwardFilter:
      return "ConvBackwardFilter";
    case HloInstruction::FusionKind::kConvBackwardInput:
      return "ConvBackwardInput";
  }
}

bool HloInstruction::CouldBeBitcast() const {
  switch (opcode_) {
    case HloOpcode::kTranspose:
      return true;
    case HloOpcode::kReshape:
      return std::get<0>(ReshapeMerelyInsertsOrDeletes1SizedDimensions());
    default:
      return false;
  }
}

}  // namespace xla
