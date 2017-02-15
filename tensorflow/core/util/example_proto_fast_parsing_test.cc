/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/util/example_proto_fast_parsing.h"

#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/util/example_proto_fast_parsing_test.pb.h"

namespace tensorflow {
namespace example {
namespace {

constexpr char kDenseInt64Key[] = "dense_int64";
constexpr char kDenseFloatKey[] = "dense_float";
constexpr char kDenseStringKey[] = "dense_string";

constexpr char kSparseInt64Key[] = "sparse_int64";
constexpr char kSparseFloatKey[] = "sparse_float";
constexpr char kSparseStringKey[] = "sparse_string";

string SerializedToReadable(string serialized) {
  string result;
  result += '"';
  for (char c : serialized)
    result += strings::StrCat("\\x", strings::Hex(c, strings::ZERO_PAD_2));
  result += '"';
  return result;
}

template <class T>
string Serialize(const T& example) {
  string serialized;
  example.SerializeToString(&serialized);
  return serialized;
}

// Tests that serialized gets parsed identically by TestFastParse(..)
// and the regular Example.ParseFromString(..).
void TestCorrectness(const string& serialized) {
  Example example;
  Example fast_example;
  EXPECT_TRUE(example.ParseFromString(serialized));
  EXPECT_TRUE(TestFastParse(serialized, &fast_example));
  EXPECT_EQ(example.DebugString(), fast_example.DebugString());
  if (example.DebugString() != fast_example.DebugString()) {
    LOG(ERROR) << "Bad serialized: " << SerializedToReadable(serialized);
  }
}

// Fast parsing does not differentiate between EmptyExample and EmptyFeatures
// TEST(FastParse, EmptyExample) {
//   Example example;
//   TestCorrectness(example);
// }

TEST(FastParse, IgnoresPrecedingUnknownTopLevelFields) {
  ExampleWithExtras example;
  (*example.mutable_features()->mutable_feature())["age"]
      .mutable_int64_list()
      ->add_value(13);
  example.set_extra1("some_str");
  example.set_extra2(123);
  example.set_extra3(234);
  example.set_extra4(345);
  example.set_extra5(4.56);
  example.add_extra6(5.67);
  example.add_extra6(6.78);
  (*example.mutable_extra7()->mutable_feature())["extra7"]
      .mutable_int64_list()
      ->add_value(1337);

  Example context;
  (*context.mutable_features()->mutable_feature())["zipcode"]
      .mutable_int64_list()
      ->add_value(94043);

  TestCorrectness(strings::StrCat(Serialize(example), Serialize(context)));
}

TEST(FastParse, IgnoresTrailingUnknownTopLevelFields) {
  Example example;
  (*example.mutable_features()->mutable_feature())["age"]
      .mutable_int64_list()
      ->add_value(13);

  ExampleWithExtras context;
  (*context.mutable_features()->mutable_feature())["zipcode"]
      .mutable_int64_list()
      ->add_value(94043);
  context.set_extra1("some_str");
  context.set_extra2(123);
  context.set_extra3(234);
  context.set_extra4(345);
  context.set_extra5(4.56);
  context.add_extra6(5.67);
  context.add_extra6(6.78);
  (*context.mutable_extra7()->mutable_feature())["extra7"]
      .mutable_int64_list()
      ->add_value(1337);

  TestCorrectness(strings::StrCat(Serialize(example), Serialize(context)));
}

TEST(FastParse, SingleInt64WithContext) {
  Example example;
  (*example.mutable_features()->mutable_feature())["age"]
      .mutable_int64_list()
      ->add_value(13);

  Example context;
  (*context.mutable_features()->mutable_feature())["zipcode"]
      .mutable_int64_list()
      ->add_value(94043);

  TestCorrectness(strings::StrCat(Serialize(example), Serialize(context)));
}

TEST(FastParse, DenseInt64WithContext) {
  Example example;
  (*example.mutable_features()->mutable_feature())["age"]
      .mutable_int64_list()
      ->add_value(0);

  Example context;
  (*context.mutable_features()->mutable_feature())["age"]
      .mutable_int64_list()
      ->add_value(15);

  string serialized = Serialize(example) + Serialize(context);

  {
    Example deserialized;
    EXPECT_TRUE(deserialized.ParseFromString(serialized));
    EXPECT_EQ(deserialized.DebugString(), context.DebugString());
    // Whoa! Last EQ is very surprising, but standard deserialization is what it
    // is and Servo team requested to replicate this 'feature'.
    // In future we should return error.
  }
  TestCorrectness(serialized);
}

TEST(FastParse, NonPacked) {
  TestCorrectness(
      "\x0a\x0e\x0a\x0c\x0a\x03\x61\x67\x65\x12\x05\x1a\x03\x0a\x01\x0d");
}

TEST(FastParse, Packed) {
  TestCorrectness(
      "\x0a\x0d\x0a\x0b\x0a\x03\x61\x67\x65\x12\x04\x1a\x02\x08\x0d");
}

TEST(FastParse, EmptyFeatures) {
  Example example;
  example.mutable_features();
  TestCorrectness(Serialize(example));
}

void TestCorrectnessJson(const string& json) {
  auto resolver = protobuf::util::NewTypeResolverForDescriptorPool(
      "type.googleapis.com", protobuf::DescriptorPool::generated_pool());
  string serialized;
  auto s = protobuf::util::JsonToBinaryString(
      resolver, "type.googleapis.com/tensorflow.Example", json, &serialized);
  EXPECT_TRUE(s.ok()) << s;
  delete resolver;
  TestCorrectness(serialized);
}

TEST(FastParse, JsonUnivalent) {
  TestCorrectnessJson(
      "{'features': {"
      "  'feature': {'age': {'int64_list': {'value': [0]} }}, "
      "  'feature': {'flo': {'float_list': {'value': [1.1]} }}, "
      "  'feature': {'byt': {'bytes_list': {'value': ['WW8='] }}}"
      "}}");
}

TEST(FastParse, JsonMultivalent) {
  TestCorrectnessJson(
      "{'features': {"
      "  'feature': {'age': {'int64_list': {'value': [0, 13, 23]} }}, "
      "  'feature': {'flo': {'float_list': {'value': [1.1, 1.2, 1.3]} }}, "
      "  'feature': {'byt': {'bytes_list': {'value': ['WW8=', 'WW8K'] }}}"
      "}}");
}

TEST(FastParse, SingleInt64) {
  Example example;
  (*example.mutable_features()->mutable_feature())["age"]
      .mutable_int64_list()
      ->add_value(13);
  TestCorrectness(Serialize(example));
}

TEST(FastParse, SomeFeatures) {
  Example example;

  (*example.mutable_features()->mutable_feature())[""];

  (*example.mutable_features()->mutable_feature())["empty_bytes_list"]
      .mutable_bytes_list();
  (*example.mutable_features()->mutable_feature())["empty_float_list"]
      .mutable_float_list();
  (*example.mutable_features()->mutable_feature())["empty_int64_list"]
      .mutable_int64_list();

  BytesList* bytes_list =
      (*example.mutable_features()->mutable_feature())["bytes_list"]
          .mutable_bytes_list();
  bytes_list->add_value("bytes1");
  bytes_list->add_value("bytes2");

  FloatList* float_list =
      (*example.mutable_features()->mutable_feature())["float_list"]
          .mutable_float_list();
  float_list->add_value(1.0);
  float_list->add_value(2.0);

  Int64List* int64_list =
      (*example.mutable_features()->mutable_feature())["int64_list"]
          .mutable_int64_list();
  int64_list->add_value(3);
  int64_list->add_value(270);
  int64_list->add_value(86942);

  TestCorrectness(Serialize(example));
}

string RandStr(random::SimplePhilox* rng) {
  static const char key_char_lookup[] =
      "0123456789{}~`!@#$%^&*()"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz";
  auto len = 1 + rng->Rand32() % 200;
  string str;
  str.reserve(len);
  while (len-- > 0) {
    str.push_back(
        key_char_lookup[rng->Rand32() % (sizeof(key_char_lookup) /
                                         sizeof(key_char_lookup[0]))]);
  }
  return str;
}

void Fuzz(random::SimplePhilox* rng) {
  // Generate keys.
  auto num_keys = 1 + rng->Rand32() % 100;
  std::unordered_set<string> unique_keys;
  for (auto i = 0; i < num_keys; ++i) {
    unique_keys.emplace(RandStr(rng));
  }

  // Generate serialized example.
  Example example;
  string serialized_example;
  auto num_concats = 1 + rng->Rand32() % 4;
  std::vector<Feature::KindCase> feat_types(
      {Feature::kBytesList, Feature::kFloatList, Feature::kInt64List});
  std::vector<string> all_keys(unique_keys.begin(), unique_keys.end());
  while (num_concats--) {
    example.Clear();
    auto num_active_keys = 1 + rng->Rand32() % all_keys.size();

    // Generate features.
    for (auto i = 0; i < num_active_keys; ++i) {
      auto fkey = all_keys[rng->Rand32() % all_keys.size()];
      auto ftype_idx = rng->Rand32() % feat_types.size();
      auto num_features = 1 + rng->Rand32() % 5;
      switch (static_cast<Feature::KindCase>(feat_types[ftype_idx])) {
        case Feature::kBytesList: {
          BytesList* bytes_list =
              (*example.mutable_features()->mutable_feature())[fkey]
                  .mutable_bytes_list();
          while (num_features--) {
            bytes_list->add_value(RandStr(rng));
          }
          break;
        }
        case Feature::kFloatList: {
          FloatList* float_list =
              (*example.mutable_features()->mutable_feature())[fkey]
                  .mutable_float_list();
          while (num_features--) {
            float_list->add_value(rng->RandFloat());
          }
          break;
        }
        case Feature::kInt64List: {
          Int64List* int64_list =
              (*example.mutable_features()->mutable_feature())[fkey]
                  .mutable_int64_list();
          while (num_features--) {
            int64_list->add_value(rng->Rand64());
          }
          break;
        }
        default: {
          QCHECK(false);
          break;
        }
      }
    }
    serialized_example += example.SerializeAsString();
  }

  // Test correctness.
  TestCorrectness(serialized_example);
}

TEST(FastParse, FuzzTest) {
  const uint64 seed = 1337;
  random::PhiloxRandom philox(seed);
  random::SimplePhilox rng(&philox);
  auto num_runs = 200;
  while (num_runs--) {
    LOG(INFO) << "runs left: " << num_runs;
    Fuzz(&rng);
  }
}

string MakeSerializedExample() {
  Example example;
  const int kFeatureNameLength = 10;
  const int kFeatureValueLength = 20;
  const int kBytesFeatureCount = 200;
  const int kFloatFeatureCount = 200;
  const int kInt64FeatureCount = 200;
  auto& fmap = *example.mutable_features()->mutable_feature();
  for (int i = 0; i < kBytesFeatureCount; ++i) {
    fmap[strings::StrCat(string('b', kFeatureNameLength), i)]
        .mutable_bytes_list()
        ->add_value(string('v', kFeatureValueLength));
  }
  for (int i = 0; i < kFloatFeatureCount; ++i) {
    fmap[strings::StrCat(string('f', kFeatureNameLength), i)]
        .mutable_float_list()
        ->add_value(123123123.123);
  }
  for (int i = 0; i < kInt64FeatureCount; ++i) {
    fmap[strings::StrCat(string('i', kFeatureNameLength), i)]
        .mutable_int64_list()
        ->add_value(10 * i);
  }
  string serialized;
  example.SerializeToString(&serialized);
  return serialized;
}

TEST(TestFastParseExample, Empty) {
  Result result;
  FastParseExampleConfig config;
  config.sparse.push_back({"test", DT_STRING});
  Status status = FastParseExample(config, gtl::ArraySlice<string>(),
                                   gtl::ArraySlice<string>(), nullptr, &result);
  EXPECT_TRUE(status.ok()) << status;
}

}  // namespace

}  // namespace example
}  // namespace tensorflow
