/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TOOLS_BENCHMARK_BENCHMARK_MODEL_H_
#define TENSORFLOW_TOOLS_BENCHMARK_BENCHMARK_MODEL_H_

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/stat_summarizer.h"

namespace tensorflow {
namespace benchmark_model {

// Used to help construct dummy inputs for the benchmarking.
struct InputLayerInfo {
  string name;
  DataType data_type;
  TensorShape shape;
};

// Loads a model from disk into a new session, and sets up the stats collection.
Status InitializeSession(int num_threads, const string& graph,
                         std::unique_ptr<Session>* session,
                         std::unique_ptr<StatSummarizer>* stats);

// Does a single run of the model that's been loaded into the given session.
Status RunBenchmark(const std::vector<InputLayerInfo>& inputs,
                    const std::vector<string>& outputs, Session* session,
                    StatSummarizer* stats);

// Runs the model multiple time, keeping track of timing information.
Status TimeMultipleRuns(double sleep_seconds, int num_runs,
                        const std::vector<InputLayerInfo>& inputs,
                        const std::vector<string>& outputs, Session* session,
                        StatSummarizer* stats);

// Handles all setup and argument parsing.
int Main(int argc, char** argv);

}  // namespace benchmark_model
}  // namespace tensorflow

#endif  // TENSORFLOW_TOOLS_BENCHMARK_BENCHMARK_MODEL_H_
