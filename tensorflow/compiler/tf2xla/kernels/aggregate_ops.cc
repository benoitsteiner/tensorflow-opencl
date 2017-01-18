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

#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"

namespace tensorflow {
namespace {

class AddNOp : public XlaOpKernel {
 public:
  explicit AddNOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    if (!ctx->ValidateInputsAreSameShape(this)) return;

    OP_REQUIRES(ctx, ctx->num_inputs() >= 1,
                errors::InvalidArgument("AddN requires at least one argument"));

    xla::ComputationDataHandle sum = ctx->Input(0);
    for (int i = 1; i < ctx->num_inputs(); ++i) {
      sum = ctx->builder()->Add(sum, ctx->Input(i));
    }

    ctx->SetOutput(0, sum);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(AddNOp);
};

REGISTER_XLA_OP("AddN", AddNOp);

}  // namespace
}  // namespace tensorflow
