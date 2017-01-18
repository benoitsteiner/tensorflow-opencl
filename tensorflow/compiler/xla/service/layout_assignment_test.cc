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

#include "tensorflow/compiler/xla/service/layout_assignment.h"

#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_layout.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace xla {
namespace {

class LayoutAssignmentTest : public HloTestBase {
 protected:
  void AssignLayouts(HloModule* module,
                     ComputationLayout* entry_computation_layout) {
    LayoutAssignment layout_assignment(entry_computation_layout);
    EXPECT_IS_OK(layout_assignment.Run(module).status());
  }
};

TEST_F(LayoutAssignmentTest, ComputationLayout) {
  // Verify the layouts of the root and parameter instructions of a computation
  // match the ComputationLayout for two different layouts.
  std::vector<std::initializer_list<int64>> minor_to_majors = {{0, 1}, {1, 0}};
  for (auto& minor_to_major : minor_to_majors) {
    auto builder = HloComputation::Builder(TestName());
    Shape ashape = ShapeUtil::MakeShape(F32, {42, 12});
    auto param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, ashape, "param0"));
    auto param1 = builder.AddInstruction(
        HloInstruction::CreateParameter(1, ashape, "param1"));
    auto add = builder.AddInstruction(
        HloInstruction::CreateBinary(ashape, HloOpcode::kAdd, param0, param1));
    HloModule module(TestName());
    HloComputation* computation = module.AddEntryComputation(builder.Build());

    Layout layout = LayoutUtil::MakeLayout(minor_to_major);
    Shape shape(ashape);
    *shape.mutable_layout() = layout;
    const ShapeLayout shape_layout(shape);

    ComputationLayout computation_layout(computation->ComputeProgramShape());
    *computation_layout.mutable_parameter_layout(0) = shape_layout;
    *computation_layout.mutable_parameter_layout(1) = shape_layout;
    *computation_layout.mutable_result_layout() = shape_layout;
    AssignLayouts(&module, &computation_layout);
    EXPECT_TRUE(LayoutUtil::Equal(layout, param0->shape().layout()));
    EXPECT_TRUE(LayoutUtil::Equal(layout, param1->shape().layout()));
    EXPECT_TRUE(LayoutUtil::Equal(layout, add->shape().layout()));
  }
}

TEST_F(LayoutAssignmentTest, ComputationLayoutMixedLayout) {
  // Verify the layouts of the root and parameter instructions of a computation
  // match the ComputationLayout which has mixed layout.
  auto builder = HloComputation::Builder(TestName());
  Shape ashape = ShapeUtil::MakeShape(F32, {42, 12});
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ashape, "param0"));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, ashape, "param1"));
  builder.AddInstruction(
      HloInstruction::CreateBinary(ashape, HloOpcode::kAdd, param0, param1));
  HloModule module(TestName());
  HloComputation* computation = module.AddEntryComputation(builder.Build());

  Layout col_major_layout = LayoutUtil::MakeLayout({1, 0});
  Shape col_major_shape(ashape);
  *col_major_shape.mutable_layout() = col_major_layout;
  const ShapeLayout col_major(col_major_shape);

  Layout row_major_layout = LayoutUtil::MakeLayout({0, 1});
  Shape row_major_shape(ashape);
  *row_major_shape.mutable_layout() = row_major_layout;
  const ShapeLayout row_major(row_major_shape);

  ComputationLayout computation_layout(computation->ComputeProgramShape());
  *computation_layout.mutable_parameter_layout(0) = col_major;
  *computation_layout.mutable_parameter_layout(1) = row_major;
  *computation_layout.mutable_result_layout() = col_major;

  AssignLayouts(&module, &computation_layout);
  EXPECT_TRUE(LayoutUtil::Equal(col_major_layout, param0->shape().layout()));
  EXPECT_TRUE(LayoutUtil::Equal(row_major_layout, param1->shape().layout()));
  EXPECT_TRUE(LayoutUtil::Equal(
      col_major_layout, computation->root_instruction()->shape().layout()));
}

TEST_F(LayoutAssignmentTest, FusionInstruction) {
  // Verify that the layout of the fused parameters in a fusion instruction
  // match that of the fusion operands. Other fused instructions should have no
  // layout.
  std::vector<std::initializer_list<int64>> minor_to_majors = {{0, 1}, {1, 0}};
  for (auto& minor_to_major : minor_to_majors) {
    auto builder = HloComputation::Builder(TestName());
    auto constant_literal1 = test_utils::CreateR2LiteralWithLayout<float>(
        {{1.0, 2.0}, {3.0, 4.0}}, minor_to_major);
    auto constant_literal2 = test_utils::CreateR2LiteralWithLayout<float>(
        {{5.0, 6.0}, {7.0, 8.0}}, minor_to_major);
    Shape ashape = constant_literal1->shape();

    auto constant1 = builder.AddInstruction(
        HloInstruction::CreateConstant(std::move(constant_literal1)));
    auto constant2 = builder.AddInstruction(
        HloInstruction::CreateConstant(std::move(constant_literal2)));
    auto add = builder.AddInstruction(HloInstruction::CreateBinary(
        ashape, HloOpcode::kAdd, constant1, constant2));
    auto negate1 = builder.AddInstruction(
        HloInstruction::CreateUnary(ashape, HloOpcode::kNegate, add));
    auto negate2 = builder.AddInstruction(
        HloInstruction::CreateUnary(ashape, HloOpcode::kNegate, negate1));

    HloModule module(TestName());
    HloComputation* computation = module.AddEntryComputation(builder.Build());

    auto fusion = computation->CreateFusionInstruction(
        {negate2, negate1, add}, HloInstruction::FusionKind::kLoop);

    Layout layout = LayoutUtil::MakeLayout(minor_to_major);
    Shape shape(ashape);
    *shape.mutable_layout() = layout;
    const ShapeLayout shape_layout(shape);

    ComputationLayout computation_layout(computation->ComputeProgramShape());
    *computation_layout.mutable_result_layout() = shape_layout;

    AssignLayouts(&module, &computation_layout);

    EXPECT_TRUE(LayoutUtil::Equal(
        layout, fusion->fused_parameter(0)->shape().layout()));
    EXPECT_TRUE(LayoutUtil::Equal(
        layout, fusion->fused_parameter(1)->shape().layout()));
    EXPECT_TRUE(LayoutUtil::Equal(
        layout, fusion->fused_expression_root()->shape().layout()));

    // Inner fused node should not have layout.
    EXPECT_FALSE(LayoutUtil::HasLayout(
        fusion->fused_expression_root()->operand(0)->shape()));
  }
}

TEST_F(LayoutAssignmentTest, TupleLayout) {
  // Verify the layouts of a tuple are assigned properly (the element layouts
  // match their source).
  auto builder = HloComputation::Builder(TestName());
  auto constant0 = builder.AddInstruction(HloInstruction::CreateConstant(
      test_utils::CreateR2LiteralWithLayout<float>({{1.0, 2.0}, {3.0, 4.0}},
                                                   {0, 1})));
  auto constant1 = builder.AddInstruction(HloInstruction::CreateConstant(
      test_utils::CreateR2LiteralWithLayout<float>({{1.0, 2.0}, {3.0, 4.0}},
                                                   {1, 0})));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({constant0, constant1}));

  // To avoid having to construct a tuple layout in the ComputationLayout below,
  // make the result of the instruction be an array.
  auto get_element0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(constant0->shape(), tuple, 0));
  auto negate = builder.AddInstruction(HloInstruction::CreateUnary(
      constant0->shape(), HloOpcode::kNegate, get_element0));

  HloModule module(TestName());
  module.AddEntryComputation(builder.Build());

  ComputationLayout computation_layout(
      module.entry_computation()->ComputeProgramShape());

  AssignLayouts(&module, &computation_layout);

  EXPECT_FALSE(
      LayoutUtil::LayoutsInShapesEqual(constant0->shape(), constant1->shape()));

  EXPECT_TRUE(LayoutUtil::HasLayout(tuple->shape()));
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(
      negate->shape(), computation_layout.result_layout().shape()));
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(
      ShapeUtil::GetTupleElementShape(tuple->shape(), 1), constant1->shape()));
}

TEST_F(LayoutAssignmentTest, TupleSelect) {
  // Verify layouts of a select with tuple operands is assigned properly.
  auto builder = HloComputation::Builder(TestName());
  auto constant0 = builder.AddInstruction(HloInstruction::CreateConstant(
      test_utils::CreateR2LiteralWithLayout<float>({{1.0, 2.0}, {3.0, 4.0}},
                                                   {0, 1})));
  auto constant1 = builder.AddInstruction(HloInstruction::CreateConstant(
      test_utils::CreateR2LiteralWithLayout<float>({{1.0, 2.0}, {3.0, 4.0}},
                                                   {1, 0})));
  auto tuple0 = builder.AddInstruction(
      HloInstruction::CreateTuple({constant0, constant1}));
  auto tuple1 = builder.AddInstruction(
      HloInstruction::CreateTuple({constant0, constant1}));

  auto pred = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));

  auto select = builder.AddInstruction(HloInstruction::CreateTernary(
      tuple0->shape(), HloOpcode::kSelect, pred, tuple0, tuple1));

  HloModule module(TestName());
  module.AddEntryComputation(builder.Build());

  ComputationLayout computation_layout(
      module.entry_computation()->ComputeProgramShape());
  Shape result_shape =
      ShapeUtil::MakeTupleShape({constant0->shape(), constant1->shape()});
  TF_CHECK_OK(computation_layout.mutable_result_layout()->CopyLayoutFromShape(
      result_shape));

  AssignLayouts(&module, &computation_layout);

  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(result_shape, select->shape()));
}

TEST_F(LayoutAssignmentTest, ConflictingLayoutTuple) {
  // Construct following computation which has conflicting layouts for two
  // elements of a tuple which share the same source logicalb buffer:
  //
  // %constant = Constant(...)
  // %inner_tuple = Tuple(%constant)
  // %nested_tuple = Tuple(%inner_tuple, %inner_tuple)
  //
  // Result layout col-major for the first element and row-major for the
  // second. This results in the conflict where the element of the inner_tuple
  // needs to be both col and row major. This is resolved by deep-copying the
  // tuple and assigning the layouts of the copied arrays as needed.
  auto builder = HloComputation::Builder(TestName());
  auto constant = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}})));
  auto inner_tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({constant}));
  auto nested_tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({inner_tuple, inner_tuple}));

  HloModule module(TestName());
  module.AddEntryComputation(builder.Build());

  ComputationLayout computation_layout(
      module.entry_computation()->ComputeProgramShape());
  Shape result_shape = nested_tuple->shape();
  *ShapeUtil::GetMutableSubshape(&result_shape, /*index=*/{0, 0}) =
      ShapeUtil::MakeShapeWithLayout(F32, {2, 2}, {1, 0});
  *ShapeUtil::GetMutableSubshape(&result_shape, /*index=*/{1, 0}) =
      ShapeUtil::MakeShapeWithLayout(F32, {2, 2}, {0, 1});
  TF_CHECK_OK(computation_layout.mutable_result_layout()->CopyLayoutFromShape(
      result_shape));

  LayoutAssignment layout_assignment(&computation_layout);
  AssignLayouts(&module, &computation_layout);

  // Layout assignment should have deep copied the result of the computation to
  // address the layout conflict. This results in several Tuple() and
  // GetTupleElement() instructions. Running algebraic simplification should
  // clean up the code to something like:
  //
  //  %constant = Constant(...) layout={1,0}
  //  %tuple.0 = Tuple(%constant) layout=({1,0})
  //  %copy = Copy(%constant) layout={0,1}  # layout transposed
  //  %tuple.1 = Tuple(%copy) layout=({0,1})
  //  %tuple.2 = Tuple(%tuple.0, %tuple.1) layout=(({1,0}), ({0,1}))
  //
  EXPECT_TRUE(
      AlgebraicSimplifier(/*is_layout_sensitive=*/true,
                          [](const Shape&, const Shape&) { return false; })
          .Run(&module)
          .ValueOrDie());
  HloInstruction* root = module.entry_computation()->root_instruction();
  // Verify layout of the root and the root's operands.
  EXPECT_TRUE(ShapeUtil::Equal(result_shape, root->shape()));
  EXPECT_TRUE(ShapeUtil::Equal(ShapeUtil::GetSubshape(result_shape, {0}),
                               root->operand(0)->shape()));
  EXPECT_TRUE(ShapeUtil::Equal(ShapeUtil::GetSubshape(result_shape, {1}),
                               root->operand(1)->shape()));

  // Verify some of the structure of the HLO graph.
  EXPECT_EQ(constant, root->operand(0)->operand(0));
  EXPECT_EQ(HloOpcode::kCopy, root->operand(1)->operand(0)->opcode());
  EXPECT_EQ(HloOpcode::kConstant,
            root->operand(1)->operand(0)->operand(0)->opcode());
}

TEST_F(LayoutAssignmentTest, ElementwiseAndReshape) {
  // param -> log -> reshape -> tanh
  auto builder = HloComputation::Builder(TestName());
  Shape ashape = ShapeUtil::MakeShape(F32, {1, 2, 3, 1});
  Shape bshape = ShapeUtil::MakeShape(F32, {2, 1, 3});
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ashape, "param"));
  auto log = builder.AddInstruction(
      HloInstruction::CreateUnary(ashape, HloOpcode::kLog, param));
  auto reshape =
      builder.AddInstruction(HloInstruction::CreateReshape(bshape, log));
  auto tanh = builder.AddInstruction(
      HloInstruction::CreateUnary(bshape, HloOpcode::kTanh, reshape));

  HloModule module(TestName());
  HloComputation* computation = module.AddEntryComputation(builder.Build(tanh));

  Shape ashape_with_layout(ashape);
  Shape bshape_with_layout(bshape);
  *ashape_with_layout.mutable_layout() = LayoutUtil::MakeLayout({0, 1, 2, 3});
  *bshape_with_layout.mutable_layout() = LayoutUtil::MakeLayout({0, 1, 2});

  ComputationLayout computation_layout(computation->ComputeProgramShape());
  *computation_layout.mutable_parameter_layout(0) =
      ShapeLayout(ashape_with_layout);
  *computation_layout.mutable_result_layout() = ShapeLayout(bshape_with_layout);
  AssignLayouts(&module, &computation_layout);

  auto log_minor_to_major =
      AsInt64Slice(log->shape().layout().minor_to_major());
  EXPECT_LT(PositionInContainer(log_minor_to_major, 1),
            PositionInContainer(log_minor_to_major, 2));

  auto reshape_minor_to_major =
      AsInt64Slice(reshape->shape().layout().minor_to_major());
  EXPECT_LT(PositionInContainer(reshape_minor_to_major, 0),
            PositionInContainer(reshape_minor_to_major, 2));
}

// Test whether LayoutAssignment assigns layouts to elementwise operations to
// keep linear indices valid across them, and to transpositions to make them
// bitcasts.
TEST_F(LayoutAssignmentTest, ElementwiseAndTranspose) {
  // param -> log -> transpose -> tanh
  auto builder = HloComputation::Builder(TestName());
  Shape ashape = ShapeUtil::MakeShape(F32, {42, 12});
  Shape bshape = ShapeUtil::MakeShape(F32, {12, 42});
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ashape, "param"));
  auto log = builder.AddInstruction(
      HloInstruction::CreateUnary(ashape, HloOpcode::kLog, param));
  auto transpose = builder.AddInstruction(
      HloInstruction::CreateTranspose(bshape, log, {1, 0}));
  auto tanh = builder.AddInstruction(
      HloInstruction::CreateUnary(bshape, HloOpcode::kTanh, transpose));
  HloModule module(TestName());
  auto computation = module.AddEntryComputation(builder.Build(tanh));

  Shape ashape_with_layout(ashape);
  Shape bshape_with_layout(bshape);
  *ashape_with_layout.mutable_layout() = LayoutUtil::MakeLayout({1, 0});
  *bshape_with_layout.mutable_layout() = LayoutUtil::MakeLayout({0, 1});

  ComputationLayout computation_layout(computation->ComputeProgramShape());
  *computation_layout.mutable_parameter_layout(0) =
      ShapeLayout(ashape_with_layout);
  *computation_layout.mutable_result_layout() = ShapeLayout(bshape_with_layout);
  AssignLayouts(&module, &computation_layout);

  EXPECT_TRUE(
      LayoutUtil::Equal(ashape_with_layout.layout(), log->shape().layout()));
  EXPECT_TRUE(LayoutUtil::Equal(bshape_with_layout.layout(),
                                transpose->shape().layout()));
  EXPECT_TRUE(
      LayoutUtil::Equal(bshape_with_layout.layout(), tanh->shape().layout()));
}

// Test whether LayoutAssignment assigns layouts to transpositions to make them
// bitcasts.
TEST_F(LayoutAssignmentTest, BroadcastAndTranspose) {
  // param -> broadcast -> transpose
  auto builder = HloComputation::Builder(TestName());
  Shape ashape = ShapeUtil::MakeShape(F32, {3, 4});
  Shape bshape = ShapeUtil::MakeShape(F32, {2, 3, 4});
  Shape cshape = ShapeUtil::MakeShape(F32, {4, 3, 2});
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ashape, "param"));
  auto broadcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(bshape, param, {1, 2}));
  auto transpose = builder.AddInstruction(
      HloInstruction::CreateTranspose(cshape, broadcast, {2, 1, 0}));
  HloModule module(TestName());
  HloComputation* computation =
      module.AddEntryComputation(builder.Build(transpose));

  Shape input_shape_with_layout(ashape);
  Shape output_shape_with_layout(cshape);
  *input_shape_with_layout.mutable_layout() = LayoutUtil::MakeLayout({1, 0});
  *output_shape_with_layout.mutable_layout() =
      LayoutUtil::MakeLayout({2, 1, 0});

  ComputationLayout computation_layout(computation->ComputeProgramShape());
  *computation_layout.mutable_parameter_layout(0) =
      ShapeLayout(input_shape_with_layout);
  *computation_layout.mutable_result_layout() =
      ShapeLayout(output_shape_with_layout);
  AssignLayouts(&module, &computation_layout);

  EXPECT_TRUE(ContainersEqual(broadcast->shape().layout().minor_to_major(),
                              tensorflow::gtl::ArraySlice<int64>{0, 1, 2}));
}

TEST_F(LayoutAssignmentTest, ReshapeOperandHasMultipleUsers) {
  // param[4] -> broadcast[3x4] ------> transpose[4x3]-------- -------> tuple
  //                            \                                     /
  //                             \-> tanh[3x4] -> broadcast2[2x3x4] -/
  //
  // The layout of `transpose` is set to {1,0} because it provides a buffer to
  // the computation result which has a fixed layout.. Therefore, `broadcast`
  // (the operand of transpose) is expected to have layout {0,1} so that the
  // transpose is a bitcast. Furthermore, `tanh` is expected to have the same
  // layout as `broadcast` (i.e. {0,1}) because `tanh` is elementwise.
  Shape f32_4 = ShapeUtil::MakeShape(F32, {4});
  Shape f32_34 = ShapeUtil::MakeShape(F32, {3, 4});
  Shape f32_43 = ShapeUtil::MakeShape(F32, {4, 3});
  Shape f32_234 = ShapeUtil::MakeShape(F32, {2, 3, 4});

  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_4, "param"));
  auto broadcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(f32_34, param, {3}));
  auto transpose = builder.AddInstruction(
      HloInstruction::CreateTranspose(f32_43, broadcast, {1, 0}));
  auto tanh = builder.AddInstruction(
      HloInstruction::CreateUnary(f32_34, HloOpcode::kTanh, broadcast));
  auto broadcast2 = builder.AddInstruction(
      HloInstruction::CreateBroadcast(f32_234, tanh, {2}));
  auto tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({transpose, broadcast2}));
  HloModule module(TestName());
  HloComputation* computation =
      module.AddEntryComputation(builder.Build(tuple));

  ComputationLayout computation_layout(computation->ComputeProgramShape());
  Shape param_shape_with_layout(f32_4);
  Shape transpose_shape_with_layout(f32_43);
  Shape broadcast2_shape_with_layout(f32_234);
  *param_shape_with_layout.mutable_layout() = LayoutUtil::MakeLayout({0});
  *transpose_shape_with_layout.mutable_layout() =
      LayoutUtil::MakeLayout({1, 0});
  *broadcast2_shape_with_layout.mutable_layout() =
      LayoutUtil::MakeLayout({2, 1, 0});

  *computation_layout.mutable_parameter_layout(0) =
      ShapeLayout(param_shape_with_layout);
  *computation_layout.mutable_result_layout() =
      ShapeLayout(ShapeUtil::MakeTupleShape(
          {transpose_shape_with_layout, broadcast2_shape_with_layout}));
  AssignLayouts(&module, &computation_layout);

  EXPECT_TRUE(ContainersEqual(broadcast->shape().layout().minor_to_major(),
                              tensorflow::gtl::ArraySlice<int64>{0, 1}));
  EXPECT_TRUE(ContainersEqual(transpose->shape().layout().minor_to_major(),
                              tensorflow::gtl::ArraySlice<int64>{1, 0}));
  EXPECT_TRUE(ContainersEqual(tanh->shape().layout().minor_to_major(),
                              tensorflow::gtl::ArraySlice<int64>{0, 1}));
}

// Add test which fails due to copy tuple.

}  // namespace
}  // namespace xla
