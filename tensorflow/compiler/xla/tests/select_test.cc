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
#include <vector>

#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class SelectTest : public ClientLibraryTestBase {
 public:
  ErrorSpec error_spec_{0.0001};
};

TEST_F(SelectTest, SelectScalarF32True) {
  ComputationBuilder builder(client_, TestName());
  auto pred = builder.ConstantR0<bool>(true);
  auto on_true = builder.ConstantR0<float>(123.0f);
  auto on_false = builder.ConstantR0<float>(42.0f);
  auto result = builder.Select(pred, on_true, on_false);

  ComputeAndCompareR0<float>(&builder, 123.0f, {}, error_spec_);
}

TEST_F(SelectTest, SelectScalarS32True) {
  ComputationBuilder builder(client_, TestName());
  auto pred = builder.ConstantR0<bool>(true);
  auto on_true = builder.ConstantR0<int32>(-42);
  auto on_false = builder.ConstantR0<int32>(42);
  auto result = builder.Select(pred, on_true, on_false);

  ComputeAndCompareR0<int32>(&builder, -42, {});
}

TEST_F(SelectTest, SelectScalarF32False) {
  ComputationBuilder builder(client_, TestName());
  auto pred = builder.ConstantR0<bool>(false);
  auto on_true = builder.ConstantR0<float>(123.0f);
  auto on_false = builder.ConstantR0<float>(42.0f);
  auto result = builder.Select(pred, on_true, on_false);

  ComputeAndCompareR0<float>(&builder, 42.0f, {}, error_spec_);
}

XLA_TEST_F(SelectTest, SelectR1S0F32WithConstantR1S0PRED) {
  ComputationBuilder builder(client_, TestName());
  auto pred = builder.ConstantR1<bool>({});
  auto on_true = builder.ConstantR1<float>({});
  auto on_false = builder.ConstantR1<float>({});
  auto select = builder.Select(pred, on_true, on_false);

  ComputeAndCompareR1<float>(&builder, {}, {}, error_spec_);
}

TEST_F(SelectTest, SelectR1F32WithConstantR1PRED) {
  ComputationBuilder builder(client_, TestName());
  auto pred = builder.ConstantR1<bool>({false, true, false, true, false});
  auto on_true = builder.ConstantR1<float>({-2.5f, 25.5f, 2.25f, -10.0f, 6.0f});
  auto on_false = builder.ConstantR1<float>({10.0f, 5.0f, 1.0f, 10.0f, -6.0f});
  auto select = builder.Select(pred, on_true, on_false);

  ComputeAndCompareR1<float>(&builder, {10.0f, 25.5f, 1.0f, -10.0f, -6.0f}, {},
                             error_spec_);
}

XLA_TEST_F(SelectTest, SelectR1S0F32WithCmpR1S0S32s) {
  // Similar to SelectR1S0F32WithConstantR1S0PRED, except that the pred vector
  // is not a constant, but rather the result of comparing two other vectors.
  ComputationBuilder builder(client_, TestName());
  auto v1 = builder.ConstantR1<int32>({});
  auto v2 = builder.ConstantR1<int32>({});
  auto cmp = builder.Eq(v1, v2);
  auto on_true = builder.ConstantR1<float>({});
  auto on_false = builder.ConstantR1<float>({});
  auto select = builder.Select(cmp, on_true, on_false);

  ComputeAndCompareR1<float>(&builder, {}, {}, error_spec_);
}

TEST_F(SelectTest, SelectR1F32WithCmpR1S32s) {
  // Similar to SelectR1F32WithConstantR1PRED, except that the pred vector is
  // not a constant, but rather the result of comparing two other vectors.
  ComputationBuilder builder(client_, TestName());
  auto v1 = builder.ConstantR1<int32>({1, 2, 3, 4, 5});
  auto v2 = builder.ConstantR1<int32>({9, 2, 9, 4, 9});
  auto cmp = builder.Eq(v1, v2);
  auto on_true = builder.ConstantR1<float>({-2.5f, 25.5f, 2.25f, -10.0f, 6.0f});
  auto on_false = builder.ConstantR1<float>({10.0f, 5.0f, 1.0f, 10.0f, -6.0f});
  auto select = builder.Select(cmp, on_true, on_false);

  ComputeAndCompareR1<float>(&builder, {10.0f, 25.5f, 1.0f, -10.0f, -6.0f}, {},
                             error_spec_);
}

TEST_F(SelectTest, SelectR1F32WithCmpR1F32s) {
  // Similar to SelectR1F32WithCmpR1S32s, except "gt"-comparing two R1F32s.
  ComputationBuilder builder(client_, TestName());
  auto v1 = builder.ConstantR1<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  auto v2 = builder.ConstantR1<float>({-1.0f, -2.0f, 13.0f, 14.0f, 4.4f});
  auto cmp = builder.Gt(v1, v2);
  auto on_true = builder.ConstantR1<float>({-2.5f, 25.5f, 2.25f, -10.0f, 6.0f});
  auto on_false = builder.ConstantR1<float>({10.0f, 5.0f, 1.0f, 10.0f, -6.0f});
  auto select = builder.Select(cmp, on_true, on_false);

  ComputeAndCompareR1<float>(&builder, {-2.5f, 25.5f, 1.0f, 10.0f, 6.0f}, {},
                             error_spec_);
}

TEST_F(SelectTest, SelectR1F32WithCmpR1F32sFromParamsSmall) {
  // Selects among two R1F32s, which come from parameters. v1 and v2 are
  // compared, and selection between them happens based on a gt-comparison mask.
  ComputationBuilder builder(client_, TestName());

  ComputationDataHandle v1, v2;
  std::unique_ptr<GlobalData> param0_data = CreateR1Parameter<float>(
      {41.0f, 2.0f, 3.0f, 84.0f}, /*parameter_number=*/0, /*name=*/"v1",
      /*builder=*/&builder, /*data_handle=*/&v1);
  std::unique_ptr<GlobalData> param1_data = CreateR1Parameter<float>(
      {21.0f, 22.0f, 23.0f, 24.0f}, /*parameter_number=*/1, /*name=*/"v2",
      /*builder=*/&builder, /*data_handle=*/&v2);

  auto cmp = builder.Gt(v1, v2);
  auto select = builder.Select(cmp, v1, v2);
  ComputeAndCompareR1<float>(&builder, {41.0f, 22.0f, 23.0f, 84.0f},
                             {param0_data.get(), param1_data.get()},
                             error_spec_);
}

TEST_F(SelectTest, SelectR1F32WithCmpR1F32sFromParamsLarge) {
  // Similar to SelectR1F32WithCmpR1F32sFromParamsSmall, except that the
  // data size passed in and out is large.
  ComputationBuilder builder(client_, TestName());

  // Number of floats in the data passed into and out of the computation.
  constexpr int datalen = 15 * 1000;

  // The inputs are initialized with a special pattern where in the first third
  // of the data v1[i] > v2[i] and elsewhere it's vice versa.
  std::vector<float> v1vec;
  std::vector<float> v2vec;
  std::vector<float> expected_vec;
  for (int i = 0; i < datalen; ++i) {
    float smaller = i;
    float larger = i * 2;
    if (i < datalen / 3) {
      v1vec.push_back(larger);
      v2vec.push_back(smaller);
    } else {
      v1vec.push_back(smaller);
      v2vec.push_back(larger);
    }
    expected_vec.push_back(larger);
  }

  ComputationDataHandle v1, v2;
  std::unique_ptr<GlobalData> param0_data =
      CreateR1Parameter<float>(v1vec, /*parameter_number=*/0, /*name=*/"v1",
                               /*builder=*/&builder, /*data_handle=*/&v1);
  std::unique_ptr<GlobalData> param1_data =
      CreateR1Parameter<float>(v2vec, /*parameter_number=*/1, /*name=*/"v2",
                               /*builder=*/&builder, /*data_handle=*/&v2);

  auto cmp = builder.Gt(v1, v2);
  auto select = builder.Select(cmp, v1, v2);
  ComputeAndCompareR1<float>(&builder, expected_vec,
                             {param0_data.get(), param1_data.get()},
                             error_spec_);
}

TEST_F(SelectTest, SelectR1F32WithCmpR1S32ToScalar) {
  // "gt"-compares a R1S32 with a S32 scalar, and uses the resulting R1PRED to
  // select between two R1F32s.
  ComputationBuilder builder(client_, TestName());
  auto v = builder.ConstantR1<int32>({1, -1, 2, -2});
  auto s = builder.ConstantR0<int32>(0);
  auto cmp = builder.Gt(v, s);

  auto on_true = builder.ConstantR1<float>({11.0f, 22.0f, 33.0f, 44.0f});
  auto on_false =
      builder.ConstantR1<float>({-111.0f, -222.0f, -333.0f, -444.0f});
  auto select = builder.Select(cmp, on_true, on_false);

  ComputeAndCompareR1<float>(&builder, {11.0f, -222.0f, 33.0f, -444.0f}, {},
                             error_spec_);
}

TEST_F(SelectTest, SelectR1F32WithCmpR1F32ToScalar) {
  // "gt"-compares a R1F32 with a F32 scalar, and uses the resulting R1PRED to
  // select between two R1F32s.
  ComputationBuilder builder(client_, TestName());
  auto v = builder.ConstantR1<float>({1.0f, 2.0f, 3.0f, 4.0f});
  auto s = builder.ConstantR0<float>(2.5f);
  auto cmp = builder.Gt(v, s);

  auto on_true = builder.ConstantR1<float>({11.0f, 22.0f, 33.0f, 44.0f});
  auto on_false =
      builder.ConstantR1<float>({-111.0f, -222.0f, -333.0f, -444.0f});
  auto select = builder.Select(cmp, on_true, on_false);

  ComputeAndCompareR1<float>(&builder, {-111.0f, -222.0f, 33.0f, 44.0f}, {},
                             error_spec_);
}

XLA_TEST_F(SelectTest, SelectR1S0F32WithScalarPredicate) {
  for (bool which : {false, true}) {
    ComputationBuilder builder(client_, TestName());
    auto pred = builder.ConstantR0<bool>(which);
    auto on_true = builder.ConstantR1<float>({});
    auto on_false = builder.ConstantR1<float>({});
    auto select = builder.Select(pred, on_true, on_false);

    ComputeAndCompareR1<float>(&builder, {}, {}, error_spec_);
  }
}

TEST_F(SelectTest, SelectR1F32WithScalarPredicateTrue) {
  ComputationBuilder builder(client_, TestName());
  auto pred = builder.ConstantR0<bool>(true);
  auto on_true = builder.ConstantR1<float>({-2.5f, 25.5f});
  auto on_false = builder.ConstantR1<float>({10.0f, 5.0f});
  auto select = builder.Select(pred, on_true, on_false);

  ComputeAndCompareR1<float>(&builder, {-2.5f, 25.5f}, {}, error_spec_);
}

TEST_F(SelectTest, SelectR1F32WithScalarPredicateFalse) {
  ComputationBuilder builder(client_, TestName());
  auto pred = builder.ConstantR0<bool>(false);
  auto on_true = builder.ConstantR1<float>({-2.5f, 25.5f});
  auto on_false = builder.ConstantR1<float>({10.0f, 5.0f});
  auto select = builder.Select(pred, on_true, on_false);

  ComputeAndCompareR1<float>(&builder, {10.0f, 5.0f}, {}, error_spec_);
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
