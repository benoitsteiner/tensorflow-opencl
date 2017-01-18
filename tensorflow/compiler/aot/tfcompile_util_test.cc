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

#include "tensorflow/compiler/aot/tfcompile_util.h"

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace tfcompile {
namespace {

void ExpectErrorContains(Status status, StringPiece str) {
  EXPECT_NE(Status::OK(), status);
  EXPECT_TRUE(StringPiece(status.error_message()).contains(str))
      << "expected error: " << status.error_message() << " to contain: " << str;
}

TEST(ValidateCppIdent, Simple) {
  TF_EXPECT_OK(ValidateCppIdent("a", ""));
  TF_EXPECT_OK(ValidateCppIdent("abc", ""));
  TF_EXPECT_OK(ValidateCppIdent("_abc", ""));
  TF_EXPECT_OK(ValidateCppIdent("_abc123", ""));
  // Make sure we didn't skip a valid letter or digit
  string ident;
  for (char c = 'a'; c <= 'z'; c++) {
    ident.append(1, c);
  }
  for (char c = 'A'; c <= 'Z'; c++) {
    ident.append(1, c);
  }
  for (char c = '0'; c <= '9'; c++) {
    ident.append(1, c);
  }
  ident += "_";
  TF_EXPECT_OK(ValidateCppIdent(ident, ""));

  ExpectErrorContains(ValidateCppIdent("", ""), "empty identifier");
  ExpectErrorContains(ValidateCppIdent(" ", ""), "illegal leading char");
  ExpectErrorContains(ValidateCppIdent("0", ""), "illegal leading char");
  ExpectErrorContains(ValidateCppIdent(".", ""), "illegal leading char");
  ExpectErrorContains(ValidateCppIdent(":", ""), "illegal leading char");
  ExpectErrorContains(ValidateCppIdent("a.", ""), "illegal char");
  ExpectErrorContains(ValidateCppIdent("a:", ""), "illegal char");
  ExpectErrorContains(ValidateCppIdent("a:", ""), "illegal char");
}

TEST(ValidateConfig, Good) {
  Config config;
  Feed* feed = config.add_feed();
  feed->mutable_id()->set_node_name("foo");
  feed->mutable_id()->set_output_index(123);
  feed->set_name("foo_debug");
  feed = config.add_feed();
  feed->mutable_id()->set_node_name("bar");
  feed->mutable_id()->set_output_index(0);
  Fetch* fetch = config.add_fetch();
  fetch->mutable_id()->set_node_name("baz");
  fetch->mutable_id()->set_output_index(456);
  fetch->set_name("baz_debug");
  fetch = config.add_fetch();
  fetch->mutable_id()->set_node_name("banana");
  fetch->mutable_id()->set_output_index(0);
  TF_EXPECT_OK(ValidateConfig(config));
}

TEST(ValidateConfig, BadEmpty) {
  Config config;
  ExpectErrorContains(ValidateConfig(config),
                      "feeds and fetches must be specified");
}

TEST(ValidateConfig, BadNoFeed) {
  Config config;
  Fetch* fetch = config.add_fetch();
  fetch->mutable_id()->set_node_name("foo");
  ExpectErrorContains(ValidateConfig(config),
                      "feeds and fetches must be specified");
}

TEST(ValidateConfig, BadNoFetch) {
  Config config;
  Feed* feed = config.add_feed();
  feed->mutable_id()->set_node_name("foo");
  ExpectErrorContains(ValidateConfig(config),
                      "feeds and fetches must be specified");
}

TEST(ValidateConfig, BadFeedNodeName) {
  Config config;
  config.add_feed();
  ExpectErrorContains(ValidateConfig(config), "node_name must be non-empty");
}

TEST(ValidateConfig, BadFeedOutputIndex) {
  Config config;
  Feed* feed = config.add_feed();
  feed->mutable_id()->set_node_name("foo");
  feed->mutable_id()->set_output_index(-1);
  ExpectErrorContains(ValidateConfig(config), "output_index must be positive");
}

TEST(ValidateConfig, BadFetchNodeName) {
  Config config;
  Feed* feed = config.add_feed();
  feed->mutable_id()->set_node_name("foo");
  config.add_fetch();
  ExpectErrorContains(ValidateConfig(config), "node_name must be non-empty");
}

TEST(ValidateConfig, BadFetchOutputIndex) {
  Config config;
  Feed* feed = config.add_feed();
  feed->mutable_id()->set_node_name("foo");
  Fetch* fetch = config.add_fetch();
  fetch->mutable_id()->set_node_name("bar");
  fetch->mutable_id()->set_output_index(-1);
  ExpectErrorContains(ValidateConfig(config), "output_index must be positive");
}

TEST(ValidateConfig, DuplicateFeedName) {
  Config config;
  Feed* feed = config.add_feed();
  feed->mutable_id()->set_node_name("foo");
  feed->set_name("dup");
  feed = config.add_feed();
  feed->mutable_id()->set_node_name("bar");
  feed->set_name("dup");
  ExpectErrorContains(ValidateConfig(config), "duplicate feed name");
}

TEST(ValidateConfig, DuplicateFetchName) {
  Config config;
  Feed* feed = config.add_feed();
  feed->mutable_id()->set_node_name("foo");
  Fetch* fetch = config.add_fetch();
  fetch->mutable_id()->set_node_name("bar");
  fetch->set_name("dup");
  fetch = config.add_fetch();
  fetch->mutable_id()->set_node_name("baz");
  fetch->set_name("dup");
  ExpectErrorContains(ValidateConfig(config), "duplicate fetch name");
}

TEST(ValidateConfig, ConflictingFeedName) {
  Config config;
  Feed* feed = config.add_feed();
  feed->mutable_id()->set_node_name("foo");
  feed->set_name("conflict");
  feed = config.add_feed();
  feed->mutable_id()->set_node_name("bar");
  feed->set_name("conflict_data");
  ExpectErrorContains(ValidateConfig(config), "conflicting feed name");
}

TEST(ValidateConfig, ConflictingFetchName) {
  Config config;
  Feed* feed = config.add_feed();
  feed->mutable_id()->set_node_name("foo");
  Fetch* fetch = config.add_fetch();
  fetch->mutable_id()->set_node_name("bar");
  fetch->set_name("conflict");
  fetch = config.add_fetch();
  fetch->mutable_id()->set_node_name("baz");
  fetch->set_name("conflict_data");
  ExpectErrorContains(ValidateConfig(config), "conflicting fetch name");
}

}  // namespace
}  // namespace tfcompile
}  // namespace tensorflow
