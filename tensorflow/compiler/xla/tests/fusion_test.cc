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

#include <math.h>
#include <algorithm>
#include <memory>
#include <new>
#include <utility>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

using tensorflow::gtl::ArraySlice;

namespace xla {
namespace {

const int test_width = 2, test_height = 3;

const float test_float_vals[3][test_width][test_height] = {
    {{-1.0, -1.0, 1.0}, {-3.0, 0.0, -1.0}},
    {{-3.0, 2.0, 1.0}, {0.0, -3.0, 1.0}},
    {{-3.0, 0.0, -3.0}, {-1.0, -2.0, 1.0}}};

// Test whether fusion operations are emitted with no errors and compute
// accurate outputs.
class FusionTest : public HloTestBase {
 protected:
  template <typename T, int Arity>
  void TestElementwise2D(HloOpcode opcode) {
    Array2D<float> operand_data[Arity];
    for (int i = 0; i < Arity; ++i) {
      new (&operand_data[i]) Array2D<float>(test_width, test_height);
    }
    Array2D<T> answer_data(test_width, test_height);
    for (int i = 0; i < test_width; ++i) {
      for (int j = 0; j < test_height; ++j) {
        float xs[Arity];
        for (int k = 0; k < Arity; ++k) {
          xs[k] = test_float_vals[k][i][j];
          operand_data[k](i, j) = xs[k];
        }
        answer_data(i, j) = ComputeElementwiseAnswer<T>(opcode, xs);
      }
    }

    auto builder = HloComputation::Builder(TestName());
    auto hlo_module = MakeUnique<HloModule>(TestName());

    auto prim_type = primitive_util::NativeToPrimitiveType<T>();

    HloInstruction* hlos[4];
    for (int i = 0; i < Arity; ++i) {
      hlos[i + 1] = builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR2FromArray2D(operand_data[i])));
    }
    auto answer_shape =
        ShapeUtil::MakeShape(prim_type, {test_width, test_height});
    std::unique_ptr<HloInstruction> root_hlo;
    switch (Arity) {
      case 1:
        root_hlo = HloInstruction::CreateUnary(answer_shape, opcode, hlos[1]);
        break;
      case 2:
        root_hlo = HloInstruction::CreateBinary(answer_shape, opcode, hlos[1],
                                                hlos[2]);
        break;
      case 3:
        root_hlo = HloInstruction::CreateTernary(answer_shape, opcode, hlos[1],
                                                 hlos[2], hlos[3]);
        break;
      default:
        LOG(FATAL) << "Bad arity: " << Arity;
    }
    hlos[0] = builder.AddInstruction(std::move(root_hlo));
    hlo_module->AddEntryComputation(builder.Build())
        ->CreateFusionInstruction(
            ArraySlice<HloInstruction*>(hlos, 0, Arity + 1),
            HloInstruction::FusionKind::kLoop);

    auto expected = LiteralUtil::CreateR2FromArray2D(answer_data);
    auto actual = ExecuteAndTransfer(std::move(hlo_module), {});
    if (primitive_util::IsFloatingPointType(prim_type)) {
      LiteralTestUtil::ExpectNear(*expected, *actual, ErrorSpec(1e-4));
    } else {
      LiteralTestUtil::ExpectEqual(*expected, *actual);
    }
  }

 private:
  template <typename T>
  T ComputeElementwiseAnswer(HloOpcode opcode, ArraySlice<float> xs);
};

template <>
float FusionTest::ComputeElementwiseAnswer<float>(HloOpcode opcode,
                                                  ArraySlice<float> xs) {
  switch (opcode) {
    case HloOpcode::kAdd:
      return xs[0] + xs[1];
    case HloOpcode::kSubtract:
      return xs[0] - xs[1];
    case HloOpcode::kMultiply:
      return xs[0] * xs[1];
    case HloOpcode::kDivide:
      return xs[0] / xs[1];
    case HloOpcode::kPower:
      return powf(xs[0], xs[1]);
    case HloOpcode::kMinimum:
      return std::min(xs[0], xs[1]);
    case HloOpcode::kMaximum:
      return std::max(xs[0], xs[1]);
    case HloOpcode::kClamp:
      return std::min(xs[2], std::max(xs[1], xs[0]));
    default:
      LOG(FATAL) << "No elementwise opcode: " << opcode;
  }
}

template <>
uint8 FusionTest::ComputeElementwiseAnswer<uint8>(HloOpcode opcode,
                                                  ArraySlice<float> xs) {
  switch (opcode) {
    case HloOpcode::kEq:
      return xs[0] == xs[1];
    case HloOpcode::kNe:
      return xs[0] != xs[1];
    case HloOpcode::kGt:
      return xs[0] > xs[1];
    case HloOpcode::kLt:
      return xs[0] < xs[1];
    case HloOpcode::kGe:
      return xs[0] >= xs[1];
    case HloOpcode::kLe:
      return xs[0] <= xs[1];
    default:
      LOG(FATAL) << "No comparatory opcode: " << opcode;
  }
}

XLA_TEST_F(FusionTest, Test) {
  // test expression:
  // slice(select({{T, F, T}, {F, T, F}},
  //              concat(transpose({{1.0}, {2.0}, {3.0}} +
  //                               {{-1.0}, {-1.0}, {-1.0}}),
  //                     {{1.62, 2.72, 3.14}}) +
  //                     (-{{1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}}),
  //              {{0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}})) = {{0.5}, {2.72}}
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = MakeUnique<HloModule>(TestName());
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{1.0}, {2.0}, {3.0}})));
  auto const1 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{-1.0}, {-1.0}, {-1.0}})));
  auto add2 = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {3, 1}), HloOpcode::kAdd, const0, const1));
  auto reshape3 = builder.AddInstruction(HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(F32, {1, 3}), add2, {1, 0}));
  auto const4 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{1.62, 2.72, 3.14}})));
  auto concat5 = builder.AddInstruction(HloInstruction::CreateConcatenate(
      ShapeUtil::MakeShape(F32, {2, 3}), {reshape3, const4}, 0));
  auto const6 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}})));
  auto negate7 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(F32, {2, 3}), HloOpcode::kNegate, const6));
  auto add8 = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {2, 3}), HloOpcode::kAdd, concat5, negate7));
  auto const9 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}})));
  auto const10 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2<bool>(
          {{true, false, true}, {false, true, false}})));
  auto select11 = builder.AddInstruction(
      HloInstruction::CreateTernary(ShapeUtil::MakeShape(F32, {2, 3}),
                                    HloOpcode::kSelect, const10, add8, const9));
  auto slice12 = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {2, 1}), select11, {0, 1}, {2, 2}));
  // CreateFusionInstruction needs the `instructions_to_fuse` argument in
  // reverse topological order, so the first element in `instructions_to_fuse`
  // must be the root.
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(
          {slice12, select11, const10, const9, add8, negate7, const6, concat5,
           const4, reshape3, add2, const1, const0},
          HloInstruction::FusionKind::kLoop);

  LiteralTestUtil::ExpectNear(*LiteralUtil::CreateR2<float>({{0.5}, {2.72}}),
                              *ExecuteAndTransfer(std::move(hlo_module), {}),
                              ErrorSpec(1e-4));
}

// Test whether we emit appropriate code for parameters of fusion instructions.
XLA_TEST_F(FusionTest, Parameter) {
  // Build a computation and fuse part of it so the fusion instruction has an
  // operand parameter.
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = MakeUnique<HloModule>(TestName());
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{1.0, 2.0, 3.0}})));
  auto copy1 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(F32, {1, 3}), HloOpcode::kCopy, const0));
  auto const2 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{-2.0, -2.0, -2.0}})));
  // add3 = copy1 + const2 = const0 + const2 = {1,2,3} + {-2,-2,-2} = {-1,0,+1}
  auto add3 = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {1, 3}), HloOpcode::kAdd, copy1, const2));
  // CreateFusionInstruction needs `instructions_to_fuse` in reverse topological
  // order.
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{add3, const2},
                                HloInstruction::FusionKind::kLoop);

  LiteralTestUtil::ExpectNear(*LiteralUtil::CreateR2<float>({{-1.0, 0.0, 1.0}}),
                              *ExecuteAndTransfer(std::move(hlo_module), {}),
                              ErrorSpec(1e-4));
}

XLA_TEST_F(FusionTest, BroadcastIntoBinaryOp) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = MakeUnique<HloModule>(TestName());
  auto const_vector = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0})));
  auto const_array = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{-1.0, -2.0, -4.0}, {10.0, 20.0, 30.0}})));
  auto broadcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(const_array->shape(), const_vector, {1}));
  // add2 = broadcast(const_vector) + const_array
  //      = broadcast({1,2,3}) + {{-1.0, -2.0, -4.0}, {10.0, 20.0, 30.0}}
  //      = {{1, 2, 3}, {1, 2, 3}} + {{-1.0, -2.0, -4.0}, {10.0, 20.0, 30.0}}
  auto add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(ShapeUtil::MakeShape(F32, {2, 3}),
                                   HloOpcode::kAdd, broadcast, const_array));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{add2, broadcast},
                                HloInstruction::FusionKind::kLoop);

  LiteralTestUtil::ExpectNear(
      *LiteralUtil::CreateR2<float>({{0.0, 0.0, -1.0}, {11.0, 22.0, 33.0}}),
      *ExecuteAndTransfer(std::move(hlo_module), {}), ErrorSpec(1e-4));
}

XLA_TEST_F(FusionTest, ReshapeToScalar) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = MakeUnique<HloModule>(TestName());
  auto single_element_array = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2<int32>({{5}})));
  auto reshape = builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(S32, {}), single_element_array));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{reshape},
                                HloInstruction::FusionKind::kLoop);
  LiteralTestUtil::ExpectEqual(*LiteralUtil::CreateR0<int32>(5),
                               *ExecuteAndTransfer(std::move(hlo_module), {}));
}

XLA_TEST_F(FusionTest, Reshape_3by2_1by2by3) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = MakeUnique<HloModule>(TestName());
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<int32>({{1, 2}, {3, 4}, {5, 6}})));
  auto reshape1 = builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(S32, {1, 2, 3}), const0));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{reshape1},
                                HloInstruction::FusionKind::kLoop);
  LiteralTestUtil::ExpectEqual(
      *LiteralUtil::CreateR3<int32>({{{1, 2, 3}, {4, 5, 6}}}),
      *ExecuteAndTransfer(std::move(hlo_module), {}));
}

XLA_TEST_F(FusionTest, Reshape_1by2by3_3by2) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = MakeUnique<HloModule>(TestName());
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR3<int32>({{{1, 2, 3}, {4, 5, 6}}})));
  auto reshape1 = builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(S32, {3, 2}), const0));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{reshape1},
                                HloInstruction::FusionKind::kLoop);
  LiteralTestUtil::ExpectEqual(
      *LiteralUtil::CreateR2<int32>({{1, 2}, {3, 4}, {5, 6}}),
      *ExecuteAndTransfer(std::move(hlo_module), {}));
}

XLA_TEST_F(FusionTest, Reshape_1by1by1_) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = MakeUnique<HloModule>(TestName());
  auto const0 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR3<int32>({{{7}}})));
  auto reshape1 = builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(S32, {}), const0));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{reshape1},
                                HloInstruction::FusionKind::kLoop);
  LiteralTestUtil::ExpectEqual(*LiteralUtil::CreateR0<int32>(7),
                               *ExecuteAndTransfer(std::move(hlo_module), {}));
}

XLA_TEST_F(FusionTest, Reshape__1by1by1) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = MakeUnique<HloModule>(TestName());
  auto const0 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(7)));
  auto reshape1 = builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(S32, {1, 1, 1}), const0));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{reshape1},
                                HloInstruction::FusionKind::kLoop);
  LiteralTestUtil::ExpectEqual(*LiteralUtil::CreateR3<int32>({{{7}}}),
                               *ExecuteAndTransfer(std::move(hlo_module), {}));
}

XLA_TEST_F(FusionTest, Reshape__) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = MakeUnique<HloModule>(TestName());
  auto const0 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(7)));
  auto reshape1 = builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(S32, {}), const0));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{reshape1},
                                HloInstruction::FusionKind::kLoop);
  LiteralTestUtil::ExpectEqual(*LiteralUtil::CreateR0<int32>(7),
                               *ExecuteAndTransfer(std::move(hlo_module), {}));
}

XLA_TEST_F(FusionTest, Reshape_3by3_3by3) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = MakeUnique<HloModule>(TestName());
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}})));
  auto reshape1 = builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(S32, {3, 3}), const0));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{reshape1},
                                HloInstruction::FusionKind::kLoop);
  LiteralTestUtil::ExpectEqual(
      *LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}),
      *ExecuteAndTransfer(std::move(hlo_module), {}));
}

XLA_TEST_F(FusionTest, Transpose_2by3) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = MakeUnique<HloModule>(TestName());
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}})));
  auto reshape1 = builder.AddInstruction(HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(S32, {3, 2}), const0, {1, 0}));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{reshape1},
                                HloInstruction::FusionKind::kLoop);
  LiteralTestUtil::ExpectEqual(
      *LiteralUtil::CreateR2<int32>({{1, 4}, {2, 5}, {3, 6}}),
      *ExecuteAndTransfer(std::move(hlo_module), {}));
}

XLA_TEST_F(FusionTest, Transpose_3by3) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = MakeUnique<HloModule>(TestName());
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}})));
  auto reshape1 = builder.AddInstruction(HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(S32, {3, 3}), const0, {1, 0}));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{reshape1},
                                HloInstruction::FusionKind::kLoop);
  LiteralTestUtil::ExpectEqual(
      *LiteralUtil::CreateR2<int32>({{1, 4, 7}, {2, 5, 8}, {3, 6, 9}}),
      *ExecuteAndTransfer(std::move(hlo_module), {}));
}

XLA_TEST_F(FusionTest, Reverse) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = MakeUnique<HloModule>(TestName());
  auto const0 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<int32>({1, 2, 3})));
  auto reverse1 = builder.AddInstruction(HloInstruction::CreateReverse(
      ShapeUtil::MakeShape(S32, {3}), const0, {0}));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{reverse1},
                                HloInstruction::FusionKind::kLoop);

  LiteralTestUtil::ExpectEqual(*LiteralUtil::CreateR1<int32>({3, 2, 1}),
                               *ExecuteAndTransfer(std::move(hlo_module), {}));
}

std::unique_ptr<HloComputation> MakeReduceTestComputation() {
  auto builder = HloComputation::Builder("add");
  auto lhs = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, ShapeUtil::MakeShape(S32, {}), "lhs"));
  auto rhs = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, ShapeUtil::MakeShape(S32, {}), "rhs"));
  builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(S32, {}), HloOpcode::kAdd, lhs, rhs));
  return builder.Build();
}

XLA_TEST_F(FusionTest, DISABLED_ON_CPU(Reduce)) {
  auto hlo_module = MakeUnique<HloModule>(TestName());

  auto builder = HloComputation::Builder(TestName());
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<int32>({1, 2, 4, 8})));
  auto const1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));
  auto reduce2 = builder.AddInstruction(HloInstruction::CreateReduce(
      ShapeUtil::MakeShape(S32, {}), const0, const1, {0},
      hlo_module->AddEmbeddedComputation(MakeReduceTestComputation())));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{reduce2},
                                HloInstruction::FusionKind::kLoop);

  LiteralTestUtil::ExpectEqual(*LiteralUtil::CreateR0<int32>(15),
                               *ExecuteAndTransfer(std::move(hlo_module), {}));
}

XLA_TEST_F(FusionTest, DISABLED_ON_CPU(ReduceImplicitBroadcast)) {
  auto hlo_module = MakeUnique<HloModule>(TestName());

  auto builder = HloComputation::Builder(TestName());
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<int32>({1, 2, 4, 8})));
  auto const1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0)));
  auto reduce2 = builder.AddInstruction(HloInstruction::CreateReduce(
      ShapeUtil::MakeShape(S32, {}), const0, const1, {0},
      hlo_module->AddEmbeddedComputation(MakeReduceTestComputation())));
  auto negate3 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(S32, {1}), HloOpcode::kNegate, reduce2));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{negate3, reduce2},
                                HloInstruction::FusionKind::kLoop);

  LiteralTestUtil::ExpectEqual(*LiteralUtil::CreateR1<int32>({-15}),
                               *ExecuteAndTransfer(std::move(hlo_module), {}));
}

XLA_TEST_F(FusionTest, DISABLED_ON_CPU(ReduceWindow)) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = MakeUnique<HloModule>(TestName());
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<int32>({{2, 3, 5}, {7, 11, 13}, {17, 19, 23}})));
  auto const1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(1)));
  Window window;
  ASSERT_TRUE(
      tensorflow::protobuf::TextFormat::ParseFromString("dimensions:{\n"
                                                        "size:2\n"
                                                        "stride:1\n"
                                                        "padding_low:0\n"
                                                        "padding_high:0\n"
                                                        "window_dilation:1\n"
                                                        "base_dilation:1\n"
                                                        "}\n"
                                                        "dimensions:{\n"
                                                        "size:2\n"
                                                        "stride:1\n"
                                                        "padding_low:0\n"
                                                        "padding_high:0\n"
                                                        "window_dilation:1\n"
                                                        "base_dilation:1\n"
                                                        "}\n",
                                                        &window));
  auto nested_builder = HloComputation::Builder("mul");
  {
    auto x = nested_builder.AddInstruction(
        HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(S32, {}), "x"));
    auto y = nested_builder.AddInstruction(
        HloInstruction::CreateParameter(1, ShapeUtil::MakeShape(S32, {}), "y"));
    nested_builder.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(S32, {}), HloOpcode::kMultiply, x, y));
  }
  auto nested_computation =
      hlo_module->AddEmbeddedComputation(nested_builder.Build());
  auto reduce_window2 =
      builder.AddInstruction(HloInstruction::CreateReduceWindow(
          ShapeUtil::MakeShape(S32, {2, 2}), const0, const1, window,
          nested_computation));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{reduce_window2},
                                HloInstruction::FusionKind::kLoop);

  LiteralTestUtil::ExpectEqual(
      *LiteralUtil::CreateR2<int32>({{462, 2145}, {24871, 62491}}),
      *ExecuteAndTransfer(std::move(hlo_module), {}));
}

XLA_TEST_F(FusionTest, Add2D) { TestElementwise2D<float, 2>(HloOpcode::kAdd); }

XLA_TEST_F(FusionTest, Subtract2D) {
  TestElementwise2D<float, 2>(HloOpcode::kSubtract);
}

XLA_TEST_F(FusionTest, Multiply2D) {
  TestElementwise2D<float, 2>(HloOpcode::kMultiply);
}

XLA_TEST_F(FusionTest, Divide2D) {
  TestElementwise2D<float, 2>(HloOpcode::kDivide);
}

XLA_TEST_F(FusionTest, Power2D) {
  TestElementwise2D<float, 2>(HloOpcode::kPower);
}

XLA_TEST_F(FusionTest, Minimum2D) {
  TestElementwise2D<float, 2>(HloOpcode::kMinimum);
}

XLA_TEST_F(FusionTest, Maximum2D) {
  TestElementwise2D<float, 2>(HloOpcode::kMaximum);
}

XLA_TEST_F(FusionTest, Equal2D) { TestElementwise2D<uint8, 2>(HloOpcode::kEq); }

XLA_TEST_F(FusionTest, Inequal2D) {
  TestElementwise2D<uint8, 2>(HloOpcode::kNe);
}

XLA_TEST_F(FusionTest, Greater2D) {
  TestElementwise2D<uint8, 2>(HloOpcode::kGt);
}

XLA_TEST_F(FusionTest, Lesser2D) {
  TestElementwise2D<uint8, 2>(HloOpcode::kLt);
}

XLA_TEST_F(FusionTest, GreaterOrEqual2D) {
  TestElementwise2D<uint8, 2>(HloOpcode::kGe);
}

XLA_TEST_F(FusionTest, LesserOrEqual2D) {
  TestElementwise2D<uint8, 2>(HloOpcode::kLe);
}

XLA_TEST_F(FusionTest, Clamp2D) {
  TestElementwise2D<float, 3>(HloOpcode::kClamp);
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  std::vector<tensorflow::Flag> flag_list;
  xla::legacy_flags::AppendCpuCompilerFlags(&flag_list);
  xla::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << "\n" << usage;
    return 2;
  }
  testing::InitGoogleTest(&argc, argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return 2;
  }
  return RUN_ALL_TESTS();
}
