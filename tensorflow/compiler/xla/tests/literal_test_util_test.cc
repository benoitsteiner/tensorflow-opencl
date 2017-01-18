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

// Tests that our utility functions for dealing with literals are correctly
// implemented.

#include "tensorflow/compiler/xla/tests/literal_test_util.h"

#include <vector>

#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

TEST(LiteralTestUtilTest, ComparesEqualTuplesEqual) {
  std::unique_ptr<Literal> literal = LiteralUtil::MakeTuple({
      LiteralUtil::CreateR0<int32>(42).get(),
      LiteralUtil::CreateR0<int32>(64).get(),
  });
  LiteralTestUtil::ExpectEqual(*literal, *literal);
}

TEST(LiteralTestUtilTest, ComparesUnequalTuplesUnequal) {
  // Implementation note: we have to use a death test here, because you can't
  // un-fail an assertion failure. The CHECK-failure is death, so we can make a
  // death assertion.
  auto unequal_things_are_equal = [] {
    std::unique_ptr<Literal> lhs = LiteralUtil::MakeTuple({
        LiteralUtil::CreateR0<int32>(42).get(),
        LiteralUtil::CreateR0<int32>(64).get(),
    });
    std::unique_ptr<Literal> rhs = LiteralUtil::MakeTuple({
        LiteralUtil::CreateR0<int32>(64).get(),
        LiteralUtil::CreateR0<int32>(42).get(),
    });
    CHECK(LiteralTestUtil::Equal(*lhs, *rhs)) << "LHS and RHS are unequal";
  };
  ASSERT_DEATH(unequal_things_are_equal(), "LHS and RHS are unequal");
}

TEST(LiteralTestUtilTest, ExpectNearFailurePlacesResultsInTemporaryDirectory) {
  auto dummy_lambda = [] {
    auto two = LiteralUtil::CreateR0<float>(2);
    auto four = LiteralUtil::CreateR0<float>(4);
    ErrorSpec error(0.001);
    CHECK(LiteralTestUtil::Near(*two, *four, error)) << "two is not near four";
  };

  tensorflow::Env* env = tensorflow::Env::Default();
  string pattern =
      tensorflow::io::JoinPath(tensorflow::testing::TmpDir(), "/tempfile-*");
  std::vector<string> files;
  TF_CHECK_OK(env->GetMatchingPaths(pattern, &files));
  for (const auto& f : files) {
    TF_CHECK_OK(env->DeleteFile(f)) << f;
  }

  ASSERT_DEATH(dummy_lambda(), "two is not near four");

  // Now check we wrote temporary files to the temporary directory that we can
  // read.
  std::vector<string> results;
  TF_CHECK_OK(env->GetMatchingPaths(pattern, &results));

  LOG(INFO) << "results: [" << tensorflow::str_util::Join(results, ", ") << "]";
  EXPECT_EQ(3, results.size());
  for (const string& result : results) {
    Literal literal;
    TF_CHECK_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(), result,
                                            &literal));
    if (result.find("expected") != string::npos) {
      EXPECT_EQ("2", LiteralUtil::ToString(literal));
    } else if (result.find("actual") != string::npos) {
      EXPECT_EQ("4", LiteralUtil::ToString(literal));
    } else if (result.find("miscompares") != string::npos) {
      EXPECT_EQ("true", LiteralUtil::ToString(literal));
    } else {
      FAIL() << "unknown file in temporary directory: " << result;
    }
  }
}

}  // namespace
}  // namespace xla
