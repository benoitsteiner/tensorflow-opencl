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

#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class CopyOpTest : public HloTestBase {
 protected:
  void TestCopyOp(const Literal& literal) {
    auto builder = HloComputation::Builder(TestName());
    auto constant = builder.AddInstruction(
        HloInstruction::CreateConstant(MakeUnique<Literal>(literal)));
    builder.AddInstruction(HloInstruction::CreateUnary(
        constant->shape(), HloOpcode::kCopy, constant));
    auto computation = builder.Build();
    auto hlo_module = MakeUnique<HloModule>("test_module");
    hlo_module->AddEntryComputation(std::move(computation));

    std::unique_ptr<Literal> result =
        ExecuteAndTransfer(std::move(hlo_module), {});
    LiteralTestUtil::ExpectEqual(literal, *result);
  }

  void TestCopyConstantLayout021(size_t n1, size_t n2, size_t n3);
  void TestCopyConstantLayoutR4(size_t n1, size_t n2, size_t n3, size_t n4,
                                tensorflow::gtl::ArraySlice<int64> permutation);
};

TEST_F(CopyOpTest, CopyR0Bool) {
  TestCopyOp(*LiteralUtil::CreateR0<bool>(true));
}

TEST_F(CopyOpTest, CopyR1S0U32) {
  TestCopyOp(*LiteralUtil::CreateR1<uint32>({}));
}

TEST_F(CopyOpTest, CopyR1S3U32) {
  TestCopyOp(*LiteralUtil::CreateR1<uint32>({1, 2, 3}));
}

TEST_F(CopyOpTest, CopyR3F32_2x2x3) {
  TestCopyOp(
      *LiteralUtil::CreateR3({{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}},
                              {{1.1f, 2.1f, 3.1f}, {6.1f, 3.5f, 2.8f}}}));
}

TEST_F(CopyOpTest, CopyR4S32_2x2x3x2) {
  TestCopyOp(*LiteralUtil::CreateR4(
      {{{{1, -2}, {-4, 5}, {6, 7}}, {{8, 9}, {10, 11}, {12, 13}}},
       {{{10, 3}, {7, -2}, {3, 6}}, {{2, 5}, {-11, 5}, {-2, -5}}}}));
}

TEST_F(CopyOpTest, CopyR4S32_0x2x3x2) {
  TestCopyOp(*LiteralUtil::CreateR4FromArray4D(Array4D<int32>(0, 2, 3, 2)));
}

TEST_F(CopyOpTest, CopyParameterScalar) {
  auto builder = HloComputation::Builder(TestName());

  // Copy literal to device to use as parameter.
  auto literal = LiteralUtil::CreateR0<float>(42.0);
  Shape shape = literal->shape();
  auto constant_device_base = TransferToDevice(*literal);

  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param0"));
  builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCopy, param0));

  auto computation = builder.Build();

  auto hlo_module = MakeUnique<HloModule>("test_module");
  hlo_module->AddEntryComputation(std::move(computation));

  std::unique_ptr<Literal> result =
      ExecuteAndTransfer(std::move(hlo_module), {constant_device_base});
  LiteralTestUtil::ExpectR0Near<float>(42.0f, *result, error_spec_);
}

TEST_F(CopyOpTest, CopyConstantR2Twice) {
  auto builder = HloComputation::Builder(TestName());

  auto literal = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(literal)));

  auto copy = builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kCopy, constant));
  builder.AddInstruction(
      HloInstruction::CreateUnary(copy->shape(), HloOpcode::kCopy, copy));

  auto computation = builder.Build();

  auto hlo_module = MakeUnique<HloModule>("test_module");
  hlo_module->AddEntryComputation(std::move(computation));
  std::unique_ptr<Literal> result =
      ExecuteAndTransfer(std::move(hlo_module), {});
  LiteralTestUtil::ExpectR2Near<float>({{1.0, 2.0}, {3.0, 4.0}}, *result,
                                       error_spec_);
}

TEST_F(CopyOpTest, CopyConstantR2DifferentLayouts) {
  HloComputation::Builder builder(TestName());

  std::unique_ptr<Literal> literal =
      LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  // Reverse the minor-to-major order of the literal.
  Layout* literal_layout = literal->mutable_shape()->mutable_layout();
  ASSERT_EQ(2, literal_layout->minor_to_major_size());
  literal_layout->mutable_minor_to_major()->SwapElements(0, 1);

  HloInstruction* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(literal)));

  builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kCopy, constant));

  std::unique_ptr<HloComputation> computation = builder.Build();

  auto hlo_module = MakeUnique<HloModule>("test_module");
  hlo_module->AddEntryComputation(std::move(computation));
  std::unique_ptr<Literal> result =
      ExecuteAndTransfer(std::move(hlo_module), {});

  // The result of the computation has the default layout, which is the inverse
  // of the layout of the source literal.
  LiteralTestUtil::ExpectR2Near<float>({{1.0, 3.0}, {2.0, 4.0}}, *result,
                                       error_spec_);
}

void CopyOpTest::TestCopyConstantLayout021(size_t n1, size_t n2, size_t n3) {
  Array3D<int32> a(n1, n2, n3);
  for (size_t i = 0; i < n1; ++i) {
    for (size_t j = 0; j < n2; ++j) {
      for (size_t k = 0; k < n3; ++k) {
        a(i, j, k) = i * n3 * n2 + j * n3 + k;
      }
    }
  }

  HloComputation::Builder builder(TestName());

  std::unique_ptr<Literal> literal = LiteralUtil::CreateR3FromArray3D(a);

  HloInstruction* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(literal)));

  builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kCopy, constant));

  std::unique_ptr<HloComputation> computation = builder.Build();

  auto hlo_module = MakeUnique<HloModule>("test_module");
  auto config = MakeUnique<HloModuleConfig>(computation->ComputeProgramShape());
  *config->mutable_entry_computation_layout()->mutable_result_layout() =
      ShapeLayout(ShapeUtil::MakeShapeWithLayout(
          constant->shape().element_type(),
          AsInt64Slice(constant->shape().dimensions()), {1, 2, 0}));
  hlo_module->AddEntryComputation(std::move(computation));
  std::unique_ptr<Literal> result =
      ExecuteAndTransfer(std::move(hlo_module), std::move(config), {});

  LiteralTestUtil::ExpectR3EqualArray3D(a, *result);
}

void CopyOpTest::TestCopyConstantLayoutR4(
    size_t n1, size_t n2, size_t n3, size_t n4,
    tensorflow::gtl::ArraySlice<int64> permutation) {
  Array4D<int32> a(n1, n2, n3, n4);
  for (size_t i = 0; i < n1; ++i) {
    for (size_t j = 0; j < n2; ++j) {
      for (size_t k = 0; k < n3; ++k) {
        for (size_t l = 0; l < n4; ++l) {
          a(i, j, k, l) = i * n4 * n3 * n2 + j * n4 * n3 + k * n4 + l;
        }
      }
    }
  }

  HloComputation::Builder builder(TestName());

  std::unique_ptr<Literal> literal = LiteralUtil::CreateR4FromArray4D(a);

  HloInstruction* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(literal)));

  builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kCopy, constant));

  std::unique_ptr<HloComputation> computation = builder.Build();

  auto hlo_module = MakeUnique<HloModule>("test_module");
  auto config = MakeUnique<HloModuleConfig>(computation->ComputeProgramShape());
  *config->mutable_entry_computation_layout()->mutable_result_layout() =
      ShapeLayout(ShapeUtil::MakeShapeWithLayout(
          constant->shape().element_type(),
          AsInt64Slice(constant->shape().dimensions()), ({
            std::vector<int64> p(permutation.rbegin(), permutation.rend());
            p;
          })));
  hlo_module->AddEntryComputation(std::move(computation));
  std::unique_ptr<Literal> result =
      ExecuteAndTransfer(std::move(hlo_module), std::move(config), {});

  LiteralTestUtil::ExpectR4EqualArray4D(a, *result);
}

XLA_TEST_F(CopyOpTest, CopyConstantR3Layout021_SingleIncompleteTilePerLayer) {
  TestCopyConstantLayout021(2, 2, 3);
}

XLA_TEST_F(CopyOpTest, CopyConstantR3Layout021_SingleCompleteTilePerLayer) {
  TestCopyConstantLayout021(2, 32, 32);
}

XLA_TEST_F(CopyOpTest, CopyConstantR3Layout021_MultipleTilesPerLayer) {
  TestCopyConstantLayout021(2, 70, 35);
}

XLA_TEST_F(CopyOpTest, CopyConstantR4Layout0231_MultipleTilesPerLayer) {
  TestCopyConstantLayoutR4(2, 70, 7, 5, {0, 2, 3, 1});
}

XLA_TEST_F(CopyOpTest, CopyConstantR4Layout0312_MultipleTilesPerLayer) {
  TestCopyConstantLayoutR4(2, 14, 5, 35, {0, 3, 1, 2});
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
