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

#include <vector>

#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class FmaxSimpleTest : public ClientLibraryTestBase {};

TEST_F(FmaxSimpleTest, FmaxTenValues) {
  ComputationBuilder builder(client_, TestName());
  auto x = builder.ConstantR1<float>(
      {-0.0, 1.0, 2.0, -3.0, -4.0, 5.0, 6.0, -7.0, -8.0, 9.0});
  auto y = builder.ConstantR1<float>(
      {-0.0, -1.0, -2.0, 3.0, 4.0, -5.0, -6.0, 7.0, 8.0, -9.0});
  builder.Max(x, y);

  std::vector<float> expected = {-0.0, 1.0, 2.0, 3.0, 4.0,
                                 5.0,  6.0, 7.0, 8.0, 9.0};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
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
