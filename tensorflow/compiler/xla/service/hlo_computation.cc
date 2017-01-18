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

#include "tensorflow/compiler/xla/service/hlo_computation.h"

#include <stddef.h>
#include <algorithm>
#include <functional>
#include <list>
#include <queue>
#include <set>
#include <sstream>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

std::unique_ptr<HloComputation> HloComputation::Builder::Build(
    HloInstruction* root_instruction) {
  int parameter_count = 0;
  for (auto& instruction : instructions_) {
    if (instruction->opcode() == HloOpcode::kParameter) {
      parameter_count++;
    }
  }
  // If root_instruction is not specified use the last added instruction.
  HloInstruction* root =
      root_instruction ? root_instruction : last_added_instruction_;
  CHECK_NE(nullptr, root);

  return WrapUnique(
      new HloComputation(name_, parameter_count, &instructions_, root));
}

HloComputation::HloComputation(
    const string& name, int parameter_count,
    std::vector<std::unique_ptr<HloInstruction>>* instructions,
    HloInstruction* root_instruction)
    : name_(name),
      root_instruction_(root_instruction),
      instruction_name_uniquer_(/*separator=*/".") {
  param_instructions_.resize(parameter_count, nullptr);
  bool root_found = false;
  for (auto& instruction : *instructions) {
    if (instruction->opcode() == HloOpcode::kParameter) {
      int64 param_no = instruction->parameter_number();
      CHECK_GE(param_no, 0);
      CHECK_LT(param_no, param_instructions_.size());
      CHECK_EQ(nullptr, param_instructions_[param_no]);
      param_instructions_[param_no] = instruction.get();
    }
    root_found |= instruction.get() == root_instruction_;
    AddInstructionInternal(std::move(instruction));
  }
  CHECK(root_found);
}

HloInstruction* HloComputation::AddInstruction(
    std::unique_ptr<HloInstruction> instruction) {
  CHECK(instruction->opcode() != HloOpcode::kParameter)
      << "Parameter instructions cannot be added to a computation after "
      << "it has been built";
  return AddInstructionInternal(std::move(instruction));
}

HloInstruction* HloComputation::AddInstructionInternal(
    std::unique_ptr<HloInstruction> instruction) {
  // Generate a unique name for the instruction.
  instruction->set_name(
      instruction_name_uniquer_.GetUniqueName(instruction->name()));
  instruction->set_parent(this);
  HloInstruction* pinst = instruction.get();
  instruction_iterators_[pinst] =
      instructions_.insert(instructions_.end(), std::move(instruction));
  return pinst;
}

void HloComputation::RemoveInstructionAndUnusedOperands(
    HloInstruction* instruction) {
  CHECK_NE(root_instruction(), instruction);

  CHECK_EQ(0, instruction->user_count());
  CHECK_NE(instruction->opcode(), HloOpcode::kParameter)
      << "Cannot remove parameter instructions";
  std::queue<HloInstruction*> remove;
  remove.push(instruction);
  while (!remove.empty()) {
    HloInstruction* item = remove.front();
    remove.pop();
    if (item->user_count() != 0 || item == root_instruction_ ||
        item->opcode() == HloOpcode::kParameter) {
      continue;
    }
    for (int i = 0; i < item->operand_count(); ++i) {
      remove.push(item->mutable_operand(i));
    }

    // If an instruction has the same operand more than once, we must not remove
    // it again.
    RemoveInstruction(item);
  }
}

bool HloComputation::RemoveInstructionIfFound(HloInstruction* instruction) {
  CHECK_NE(instruction->opcode(), HloOpcode::kParameter)
      << "Cannot remove parameter instructions";
  CHECK_NE(root_instruction(), instruction) << "cannot remove root instruction";
  CHECK_EQ(0, instruction->user_count())
      << "instruction with users cannot be removed";

  if (instruction_iterators_.count(instruction) == 0) {
    return false;
  }
  auto inst_it = instruction_iterators_.at(instruction);
  (*inst_it)->set_parent(nullptr);
  instruction->DetachFromOperands();
  instructions_.erase(inst_it);
  return true;
}

void HloComputation::RemoveInstruction(HloInstruction* instruction) {
  CHECK(RemoveInstructionIfFound(instruction))
      << instruction->ToString() << " is not a member of computation "
      << name();
}

void HloComputation::ReplaceUsesOfInstruction(
    HloInstruction* instruction_to_replace, HloInstruction* instruction) {
  instruction_to_replace->ReplaceAllUsesWith(instruction);
  if (instruction_to_replace == root_instruction()) {
    set_root_instruction(instruction);
  }
}

void HloComputation::set_root_instruction(
    HloInstruction* new_root_instruction) {
  // The shape of the root (ignoring layout) is an invariant of the computation.
  CHECK(ShapeUtil::Compatible(new_root_instruction->shape(),
                              root_instruction_->shape()))
      << new_root_instruction->shape().ShortDebugString()
      << " is incompatible with "
      << root_instruction_->shape().ShortDebugString();
  bool root_found = false;
  for (auto& instruction : instructions_) {
    if (new_root_instruction == instruction.get()) {
      root_found = true;
      break;
    }
  }
  DCHECK(root_found);

  root_instruction_ = new_root_instruction;
}

namespace {

// Helper class which computes the post order of an expression rooted at a
// particular instruction.
class InstructionPostOrderer : public DfsHloVisitorWithDefault {
 public:
  // added_instructions is the set of instructions which have already been
  // accounted for in the post order in previous invocations of
  // GetOrder. Without this mechanism, instructions which are predecessors of
  // multiple root instructions of the computation can be added to the post
  // order more than once.
  static std::list<HloInstruction*> GetOrder(
      HloInstruction* root,
      tensorflow::gtl::FlatSet<HloInstruction*>* added_instructions) {
    InstructionPostOrderer orderer(added_instructions);
    TF_CHECK_OK(root->Accept(&orderer));
    return std::move(orderer.post_order_);
  }

 private:
  explicit InstructionPostOrderer(
      tensorflow::gtl::FlatSet<HloInstruction*>* added_instructions)
      : added_instructions_(added_instructions) {}
  ~InstructionPostOrderer() override {}

  Status DefaultAction(HloInstruction* hlo_instruction) override {
    if (added_instructions_->count(hlo_instruction) == 0) {
      post_order_.push_back(hlo_instruction);
      added_instructions_->insert(hlo_instruction);
    }
    return Status::OK();
  }

  std::list<HloInstruction*> post_order_;
  tensorflow::gtl::FlatSet<HloInstruction*>* added_instructions_;
};

// Helper which builds a post order of the HLO call graph.
void ComputeComputationPostOrder(
    HloComputation* computation,
    tensorflow::gtl::FlatSet<HloComputation*>* visited,
    std::list<HloComputation*>* post_order) {
  if (visited->count(computation) > 0) {
    return;
  }

  for (auto& instruction : computation->instructions()) {
    for (auto& called_computation : instruction->MakeCalledComputationsSet()) {
      ComputeComputationPostOrder(called_computation, visited, post_order);
    }
  }

  visited->insert(computation);
  post_order->push_back(computation);
  return;
}

}  // namespace

std::list<HloInstruction*> HloComputation::MakeInstructionPostOrder() const {
  std::list<HloInstruction*> post_order;
  std::list<HloInstruction*> trace_instructions;
  tensorflow::gtl::FlatSet<HloInstruction*> added_instructions;
  for (auto& instruction : instructions_) {
    if (instruction->opcode() == HloOpcode::kTrace) {
      // Trace instructions aren't handled by the DFS visitor. Add trace
      // instructions to the post order at the end (necessarily they have no
      // users).
      trace_instructions.push_back(instruction.get());
    } else if (instruction->users().empty()) {
      post_order.splice(post_order.end(),
                        InstructionPostOrderer::GetOrder(instruction.get(),
                                                         &added_instructions));
    }
  }
  post_order.splice(post_order.end(), trace_instructions);
  CHECK_EQ(instructions_.size(), post_order.size())
      << "number of instructions does not match post order size";
  return post_order;
}

std::list<HloComputation*> HloComputation::MakeEmbeddedComputationsList()
    const {
  tensorflow::gtl::FlatSet<HloComputation*> visited;
  std::list<HloComputation*> post_order;

  // To avoid special handling of this computation, cast away const of
  // 'this'. 'this' is immediately removed from the post order after
  // construction.
  ComputeComputationPostOrder(const_cast<HloComputation*>(this), &visited,
                              &post_order);

  // We don't want to include this computation in the post order.
  CHECK_EQ(this, post_order.back());
  post_order.pop_back();

  return post_order;
}

string HloComputation::ToString() const {
  std::ostringstream s;
  s << name() << " " << ShapeUtil::HumanString(ComputeProgramShape())
    << " { \n";
  for (const HloInstruction* instruction : MakeInstructionPostOrder()) {
    s << "  " << instruction->ToString() << "\n";
    if (instruction->opcode() == HloOpcode::kFusion) {
      for (const auto& fused_instruction : instruction->fused_instructions()) {
        s << "    " << fused_instruction->ToString() << "\n";
      }
    }
  }
  s << "}";
  return s.str();
}

void HloComputation::FuseInstructionsInto(
    tensorflow::gtl::ArraySlice<HloInstruction*> instructions_to_fuse,
    HloInstruction* fusion_instruction) {
  CHECK_EQ(HloOpcode::kFusion, fusion_instruction->opcode());
  HloInstruction* root = instructions_to_fuse.front();
  root->ReplaceAllUsesWith(fusion_instruction);
  if (root == root_instruction()) {
    set_root_instruction(fusion_instruction);
  }
  RemoveInstruction(root);
  for (size_t i = 1; i < instructions_to_fuse.size(); ++i) {
    HloInstruction* instruction = instructions_to_fuse[i];
    fusion_instruction->FuseInstruction(instruction);
    if (instruction->user_count() == 0) {
      RemoveInstruction(instruction);
    }
  }
}

HloInstruction* HloComputation::CreateFusionInstruction(
    tensorflow::gtl::ArraySlice<HloInstruction*> instructions_to_fuse,
    HloInstruction::FusionKind fusion_kind) {
  HloInstruction* root = instructions_to_fuse.front();
  HloInstruction* fusion_instruction = AddInstruction(
      HloInstruction::CreateFusion(root->shape(), fusion_kind, root));
  FuseInstructionsInto(instructions_to_fuse, fusion_instruction);
  return fusion_instruction;
}

HloInstruction* HloComputation::CreateFusionInstructionForBackwardConvolution(
    tensorflow::gtl::ArraySlice<HloInstruction*> instructions_to_fuse,
    HloInstruction::FusionKind fusion_kind, const Window& window,
    const ConvolutionDimensionNumbers& conv_dnums) {
  CHECK(HloInstruction::FusionKind::kConvBackwardFilter == fusion_kind ||
        HloInstruction::FusionKind::kConvBackwardInput == fusion_kind);
  HloInstruction* root = instructions_to_fuse.front();
  HloInstruction* fusion_instruction =
      AddInstruction(HloInstruction::CreateFusionForBackwardConvolution(
          root->shape(), fusion_kind, window, conv_dnums, root));
  FuseInstructionsInto(instructions_to_fuse, fusion_instruction);
  return fusion_instruction;
}

StatusOr<HloInstruction*> HloComputation::DeepCopyTuple(
    HloInstruction* instruction) {
  TF_RET_CHECK(ShapeUtil::IsTuple(instruction->shape()));
  std::vector<HloInstruction*> element_copies;
  for (int64 i = 0; i < ShapeUtil::TupleElementCount(instruction->shape());
       ++i) {
    HloInstruction* gte = AddInstruction(HloInstruction::CreateGetTupleElement(
        ShapeUtil::GetSubshape(instruction->shape(), {i}), instruction, i));
    // Recurse to copy tuple elements. For array elements, insert a kCopy
    // because GetTupleElement forwards a pointer to the tuple element buffer.
    HloInstruction* element_copy;
    if (ShapeUtil::IsTuple(gte->shape())) {
      TF_ASSIGN_OR_RETURN(element_copy, DeepCopyTuple(gte));
    } else {
      element_copy = AddInstruction(
          HloInstruction::CreateUnary(gte->shape(), HloOpcode::kCopy, gte));
    }
    element_copies.push_back(element_copy);
  }

  // Gather element copies into a tuple with a new Tuple instruction.
  return AddInstruction(HloInstruction::CreateTuple(element_copies));
}

StatusOr<HloInstruction*> HloComputation::DeepCopyInstruction(
    HloInstruction* instruction) {
  if (instruction->parent() != this) {
    return FailedPrecondition(
        "Can't deep copy instruction %s: instruction is not in computation %s",
        instruction->name().c_str(), name().c_str());
  }

  // For tuple instructions, perform a deep copy. For array instructions, copy
  // with a kCopy instruction.
  if (ShapeUtil::IsTuple(instruction->shape())) {
    return DeepCopyTuple(instruction);
  } else if (ShapeUtil::IsArray(instruction->shape())) {
    return AddInstruction(HloInstruction::CreateUnary(
        instruction->shape(), HloOpcode::kCopy, instruction));
  } else {
    return FailedPrecondition(
        "Can only copy array and tuple shaped instructions");
  }
}

Status HloComputation::AddControlDependency(HloInstruction* predecessor,
                                            HloInstruction* successor) {
  TF_RET_CHECK(instruction_iterators_.count(predecessor) > 0);
  TF_RET_CHECK(instruction_iterators_.count(successor) > 0);
  successor->AddControlPredecessor(predecessor);
  return Status::OK();
}

ProgramShape HloComputation::ComputeProgramShape() const {
  ProgramShape program_shape;

  for (auto* param_instruction : param_instructions_) {
    *program_shape.add_parameters() = param_instruction->shape();
    *program_shape.add_parameter_names() = param_instruction->parameter_name();
  }
  *program_shape.mutable_result() = root_instruction_->shape();

  LayoutUtil::ClearLayout(&program_shape);
  return program_shape;
}

bool HloComputation::operator==(const HloComputation& other) const {
  std::set<std::pair<const HloInstruction*, const HloInstruction*>> visited;
  std::function<bool(const HloInstruction*, const HloInstruction*)> eq =
      [&visited, &eq](const HloInstruction* a, const HloInstruction* b) {
        // If <a,b> are visited but not identical, the recursion should have
        // been aborted. So, if <a,b> are visited at this point, they must be
        // identical.
        if (visited.count(std::make_pair(a, b)) > 0) return true;
        visited.emplace(a, b);
        return a->Identical(
            *b, eq, [](const HloComputation* a, const HloComputation* b) {
              return *a == *b;
            });
      };
  return eq(root_instruction(), other.root_instruction());
}

void HloComputation::ReplaceWithNewInstruction(
    HloInstruction* old_instruction,
    std::unique_ptr<HloInstruction> new_instruction) {
  ReplaceInstruction(old_instruction,
                     AddInstruction(std::move(new_instruction)));
}

void HloComputation::ReplaceInstruction(HloInstruction* old_instruction,
                                        HloInstruction* new_instruction) {
  CHECK(ShapeUtil::Compatible(old_instruction->shape(),
                              new_instruction->shape()));
  VLOG(10) << "transformed " << old_instruction->ToString() << " to "
           << new_instruction->ToString();
  ReplaceUsesOfInstruction(old_instruction, new_instruction);
  RemoveInstructionAndUnusedOperands(old_instruction);
}

HloComputation::ReachabilityMap::ReachabilityMap(
    const std::list<HloInstruction*>& all_instructions) {
  const int n = all_instructions.size();
  int next_id = 0;
  for (const auto* hlo : all_instructions) {
    ids_[hlo] = next_id;
    next_id++;
  }
  DCHECK_EQ(n, ids_.size());  // instructions should be unique
  matrix_.Reset(n * n);
}

void HloComputation::ReachabilityMap::SetReachable(const HloInstruction* a,
                                                   const HloInstruction* b) {
  const int id_a = FindOrDie(ids_, a);
  const int id_b = FindOrDie(ids_, b);
  matrix_.set(id_a * ids_.size() + id_b);
}

bool HloComputation::ReachabilityMap::IsReachable(
    const HloInstruction* a, const HloInstruction* b) const {
  const int id_a = FindOrDie(ids_, a);
  const int id_b = FindOrDie(ids_, b);
  return matrix_.get(id_a * ids_.size() + id_b);
}

bool HloComputation::ReachabilityMap::IsConnected(
    const HloInstruction* a, const HloInstruction* b) const {
  const int id_a = FindOrDie(ids_, a);
  const int id_b = FindOrDie(ids_, b);
  return matrix_.get(id_a * ids_.size() + id_b) ||
         matrix_.get(id_b * ids_.size() + id_a);
}

void HloComputation::ReachabilityMap::SetReachableAndTransitiveClosure(
    const HloInstruction* a, const HloInstruction* b) {
  const int id_a = FindOrDie(ids_, a);
  const int id_b = FindOrDie(ids_, b);
  const int n = ids_.size();
  matrix_.set(id_a * n + id_b);

  // Copy transitive set for b into entries for a
  for (int i = 0; i < n; i++) {
    if (matrix_.get(id_b * n + i)) {
      matrix_.set(id_a * n + i);
    }
  }
}

std::unique_ptr<HloComputation::ReachabilityMap>
HloComputation::ComputeTransitiveOperands() const {
  const auto all = MakeInstructionPostOrder();
  auto result = MakeUnique<HloComputation::ReachabilityMap>(all);

  // Fill in the dependency bit matrix
  for (const auto* hlo : all) {
    for (const HloInstruction* operand : hlo->operands()) {
      result->SetReachableAndTransitiveClosure(hlo, operand);
    }
  }
  return result;
}

Status HloComputation::Accept(DfsHloVisitor* visitor) const {
  // Visit all dead roots.
  for (auto& instruction : instructions()) {
    if (instruction->user_count() == 0 &&
        instruction.get() != root_instruction()) {
      // Call FinishVisit only at the end.
      TF_RETURN_IF_ERROR(
          instruction->Accept(visitor, /*call_finish_visit=*/false));
    }
  }
  // Visit root instruction last.
  return root_instruction()->Accept(visitor, /*call_finish_visit=*/true);
}

Status HloComputation::Accept(
    const FunctionVisitor::VisitorFunction& visitor_func) const {
  FunctionVisitor visitor(visitor_func);
  return this->Accept(&visitor);
}

}  // namespace xla
