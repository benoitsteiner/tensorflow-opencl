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
#include <string>

#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"

int main(int argc, char** argv) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  xla::LocalClient* client(xla::ClientLibrary::LocalClientOrDie());

  // Transfer parameters.
  std::unique_ptr<xla::Literal> param0_literal =
      xla::LiteralUtil::CreateR1<float>({1.1f, 2.2f, 3.3f, 5.5f});
  std::unique_ptr<xla::GlobalData> param0_data =
      client->TransferToServer(*param0_literal).ConsumeValueOrDie();

  std::unique_ptr<xla::Literal> param1_literal =
      xla::LiteralUtil::CreateR2<float>(
          {{3.1f, 4.2f, 7.3f, 9.5f}, {1.1f, 2.2f, 3.3f, 4.4f}});
  std::unique_ptr<xla::GlobalData> param1_data =
      client->TransferToServer(*param1_literal).ConsumeValueOrDie();

  // Build computation.
  xla::ComputationBuilder builder(client, "");
  auto p0 = builder.Parameter(0, param0_literal->shape(), "param0");
  auto p1 = builder.Parameter(1, param1_literal->shape(), "param1");
  auto add = builder.Add(p1, p0, {0});

  xla::StatusOr<xla::Computation> computation_status = builder.Build();
  xla::Computation computation = computation_status.ConsumeValueOrDie();

  // Execute and transfer result of computation.
  xla::ExecutionProfile profile;
  xla::StatusOr<std::unique_ptr<xla::Literal>> result =
      client->ExecuteAndTransfer(
          computation,
          /*arguments=*/{param0_data.get(), param1_data.get()},
          /*shape_with_output_layout=*/nullptr,
          /*execution_profile=*/&profile);
  std::unique_ptr<xla::Literal> actual = result.ConsumeValueOrDie();

  LOG(INFO) << tensorflow::strings::Printf("computation took %lldns",
                                           profile.compute_time_ns());
  LOG(INFO) << xla::LiteralUtil::ToString(*actual);

  return 0;
}
