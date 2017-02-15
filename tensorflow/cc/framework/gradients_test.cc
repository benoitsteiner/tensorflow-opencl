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

#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/testutil.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/equal_graph_def.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
using namespace ops;  // NOLINT(build/namespaces)

namespace {

// TODO(andydavis) Add more unit tests once more gradient functions are ported.
class GradientsTest : public ::testing::Test {
 protected:
  GradientsTest()
      : scope_expected_(Scope::NewRootScope()),
        scope_test_(Scope::NewRootScope()) {}

  void CompareTestAndExpectedGraphs() {
    GraphDef gdef_test;
    TF_ASSERT_OK(scope_test_.ToGraphDef(&gdef_test));
    GraphDef gdef_exp;
    TF_ASSERT_OK(scope_expected_.ToGraphDef(&gdef_exp));
    TF_EXPECT_GRAPH_EQ(gdef_test, gdef_exp);
  }

  Scope scope_expected_;
  Scope scope_test_;
};

// EX.
//      ^             ^
//    dy|           dx|        // MatMul Gradient Graph
//      |             |
//   MatMul_1      MatMul_2
//   ^   ^          ^    ^
//   |   |----------|    |
//   |        ^          |
//   |      dz|          |
//   |        |          |
//   |     Const_3       |
//   |                   |
//   |        ^          |
//   |       z|          |     // MatMul Forward Graph
//   |        |          |
//   |      MatMul_0     |
//   |     /        \    |
//   |    ^          ^   |
//   |    |          |   |
//   |---x|         y|---|
//        |          |
//        |          |
//      Const_0   Const_1
//

TEST_F(GradientsTest, OneMatMul) {
  for (const bool expected : {false, true}) {
    const Scope& scope = expected ? scope_expected_ : scope_test_;
    // Construct forward graph.
    auto x = Const(scope, {{1.0, 2.0}, {3.0, 4.0}});
    auto y = Const(scope, {{1.0, 0.0}, {0.0, 1.0}});
    auto z = MatMul(scope, x, y);
    TF_ASSERT_OK(scope.status());
    CHECK_NOTNULL(z.node());

    if (expected) {
      // Construct backward graph.
      auto dz = Const(scope, {{1.0, 1.0}, {1.0, 1.0}});
      auto dx = MatMul(scope, dz, y, MatMul::TransposeB(true));
      auto dy = MatMul(scope, x, dz, MatMul::TransposeA(true));
    } else {
      // Call AddSymbolicGradients.
      auto dz = Const(scope, {{1.0, 1.0}, {1.0, 1.0}});
      std::vector<Output> grad_outputs;
      TF_ASSERT_OK(
          AddSymbolicGradients(scope, {z}, {x, y}, {dz}, &grad_outputs));
    }
  }
  CompareTestAndExpectedGraphs();
}

TEST_F(GradientsTest, TwoMatMuls_Chained) {
  for (const bool expected : {false, true}) {
    const Scope& scope = expected ? scope_expected_ : scope_test_;
    // Construct forward graph.
    auto u = Const(scope, {{1.0, 2.0}, {3.0, 4.0}});
    auto v = Const(scope, {{1.0, 0.0}, {0.0, 1.0}});
    auto x = MatMul(scope, u, v);

    auto y = Const(scope, {{1.0, 0.0}, {0.0, 1.0}});
    auto z = MatMul(scope, x, y);

    TF_ASSERT_OK(scope.status());
    CHECK_NOTNULL(z.node());

    if (expected) {
      // Construct backward graph.
      auto dz = Const(scope, {{1.0, 1.0}, {1.0, 1.0}});
      auto dx = MatMul(scope, dz, y, MatMul::TransposeB(true));
      auto dy = MatMul(scope, x, dz, MatMul::TransposeA(true));

      auto du = MatMul(scope, dx, v, MatMul::TransposeB(true));
      auto dv = MatMul(scope, u, dx, MatMul::TransposeA(true));
    } else {
      // Call AddSymbolicGradients.
      auto dz = Const(scope, {{1.0, 1.0}, {1.0, 1.0}});
      std::vector<Output> grad_outputs;
      TF_ASSERT_OK(
          AddSymbolicGradients(scope, {z}, {u, v}, {dz}, &grad_outputs));
    }
  }
  CompareTestAndExpectedGraphs();
}

TEST_F(GradientsTest, TwoMatMuls_Independent) {
  for (const bool expected : {false, true}) {
    const Scope& scope = expected ? scope_expected_ : scope_test_;
    // Construct forward graph.
    auto t = Const(scope, {{1.0, 2.0}, {3.0, 4.0}});
    auto u = Const(scope, {{1.0, 0.0}, {0.0, 1.0}});
    auto v = MatMul(scope, t, u);
    TF_ASSERT_OK(scope.status());
    CHECK_NOTNULL(v.node());

    auto x = Const(scope, {{5.0, 6.0}, {7.0, 8.0}});
    auto y = Const(scope, {{1.0, 0.0}, {0.0, 1.0}});
    auto z = MatMul(scope, x, y);
    TF_ASSERT_OK(scope.status());
    CHECK_NOTNULL(z.node());

    if (expected) {
      // Construct backward graph.
      auto dv = Const(scope, {{1.0, 1.0}, {1.0, 1.0}});
      auto dt = MatMul(scope, dv, u, MatMul::TransposeB(true));
      auto du = MatMul(scope, t, dv, MatMul::TransposeA(true));

      auto dz = Const(scope, {{1.0, 1.0}, {1.0, 1.0}});
      auto dx = MatMul(scope, dz, y, MatMul::TransposeB(true));
      auto dy = MatMul(scope, x, dz, MatMul::TransposeA(true));
    } else {
      // Call AddSymbolicGradients.
      auto dv = Const(scope, {{1.0, 1.0}, {1.0, 1.0}});
      auto dz = Const(scope, {{1.0, 1.0}, {1.0, 1.0}});
      std::vector<Output> grad_outputs;
      TF_ASSERT_OK(AddSymbolicGradients(scope, {v, z}, {t, u, x, y}, {dv, dz},
                                        &grad_outputs));
    }
  }
  CompareTestAndExpectedGraphs();
}

TEST_F(GradientsTest, StackUnstack_Chained) {
  for (const bool expected : {false, true}) {
    const Scope& scope = expected ? scope_expected_ : scope_test_;
    // Construct forward graph.
    auto a = Const(scope, 1, {4, 2});
    auto b = Const(scope, 2, {4, 2});
    auto c = Const(scope, 3, {4, 2});

    auto pack = Stack(scope, {a, b, c});
    auto unpack = Unstack(scope, pack.output, 3);
    TF_ASSERT_OK(scope.status());

    // Construct grad inputs.
    auto dx = Const(scope, 4, {4, 2});
    auto dy = Const(scope, 5, {4, 2});
    auto dz = Const(scope, 6, {4, 2});

    if (expected) {
      // Construct backward graph.
      auto unpack_grad = Stack(scope, {dx, dy, dz});
      auto pack_grad = Unstack(scope, unpack_grad.output, 3);
    } else {
      // Call AddSymbolicGradients.
      std::vector<Output> grad_outputs;
      TF_ASSERT_OK(AddSymbolicGradients(scope, unpack.output, {a, b, c},
                                        {dx, dy, dz}, &grad_outputs));
    }
  }
  CompareTestAndExpectedGraphs();
}

TEST_F(GradientsTest, StackUnstack_StopBackprop) {
  // Tests that backprop stops before calculating gradients for Stack (because
  // only gradients w.r.t the output of Stack are requested).
  for (const bool expected : {false, true}) {
    const Scope& scope = expected ? scope_expected_ : scope_test_;
    // Construct forward graph.
    auto a = Const(scope, 1, {4, 2});
    auto b = Const(scope, 2, {4, 2});
    auto c = Const(scope, 3, {4, 2});

    auto pack = Stack(scope, {a, b, c});
    auto unpack = Unstack(scope, pack.output, 3);
    TF_ASSERT_OK(scope.status());

    // Construct grad inputs.
    auto dx = Const(scope, 4, {4, 2});
    auto dy = Const(scope, 5, {4, 2});
    auto dz = Const(scope, 6, {4, 2});

    if (expected) {
      // Construct backward graph.
      // NOTE: We should only expect the grad function for unpack in the
      // gradients graph, based on the requested grad outputs.
      auto unpack_grad = Stack(scope, {dx, dy, dz});
    } else {
      // Call AddSymbolicGradients.
      std::vector<Output> grad_outputs;
      TF_ASSERT_OK(AddSymbolicGradients(scope, unpack.output, {pack},
                                        {dx, dy, dz}, &grad_outputs));
    }
  }
  CompareTestAndExpectedGraphs();
}

TEST_F(GradientsTest, DependentGradOutputs) {
  // Tests that dependant gradients (in this case the gradients w.r.t to the
  // output and one input of MatMul) are computed properly.

  // Create two chained MatMul ops.
  auto u = Const(scope_test_, {{2}});
  auto v = Const(scope_test_, {{3}});
  auto x = MatMul(scope_test_, u, v);

  auto y = Const(scope_test_, {{4}});
  auto z = MatMul(scope_test_, x, y);

  TF_ASSERT_OK(scope_test_.status());
  CHECK_NOTNULL(z.node());

  // Call AddSymbolicGradients with '5' as initial gradients for 'dz'.
  // The gradient w.r.t to 'v' (returned in grad_outputs[0]) is dependent on
  // the gradient w.r.t. to 'x' (returned in grad_outputs[1]).
  auto dz = Const(scope_test_, {{5}});
  std::vector<Output> grad_outputs;
  TF_ASSERT_OK(
      AddSymbolicGradients(scope_test_, {z}, {v, x}, {dz}, &grad_outputs));

  std::vector<Tensor> outputs;
  test::GetTensors(scope_test_, {grad_outputs[0], grad_outputs[1]}, &outputs);

  // The gradients w.r.t to 'dz' are passed into AddSymbolicGradients as '5'.
  // Since z = MatMul(x, y), the gradients w.r.t 'x' are computed as:
  //   'dx' = 5 * 'y' = 5 * 4 = 20.
  // Since x = MatMul(u, v), the gradients w.r.t. 'v' are computed as:
  //   'dv' = 'dx' * 'u' = 20 * 2 = 40.
  test::ExpectTensorEqual<int>(outputs[0], test::AsTensor<int>({40}, {1, 1}));
  test::ExpectTensorEqual<int>(outputs[1], test::AsTensor<int>({20}, {1, 1}));
}

TEST_F(GradientsTest, MultipleNodeOutputGrads) {
  // Tests that gradients for multiple outputs of the same node are returned.
  auto x = Const(scope_test_, 1, {3, 4, 2});
  auto unpack = Unstack(scope_test_, x, 3);
  auto pack = Stack(scope_test_, unpack.output);

  // clang-format off
  auto dx = Const(scope_test_, {40, 41, 42, 43, 44, 45, 46, 47,
                                50, 51, 52, 53, 55, 55, 56, 57,
                                60, 61, 62, 63, 66, 66, 66, 67},
                               {3, 4, 2});
  // clang-format on

  std::vector<Output> grad_outputs;
  TF_ASSERT_OK(AddSymbolicGradients(scope_test_, {pack}, unpack.output, {dx},
                                    &grad_outputs));

  std::vector<Tensor> outputs;
  test::GetTensors(scope_test_,
                   {grad_outputs[0], grad_outputs[1], grad_outputs[2]},
                   &outputs);

  test::ExpectTensorEqual<int>(
      outputs[0],
      test::AsTensor<int>({40, 41, 42, 43, 44, 45, 46, 47}, {4, 2}));
  test::ExpectTensorEqual<int>(
      outputs[1],
      test::AsTensor<int>({50, 51, 52, 53, 55, 55, 56, 57}, {4, 2}));
  test::ExpectTensorEqual<int>(
      outputs[2],
      test::AsTensor<int>({60, 61, 62, 63, 66, 66, 66, 67}, {4, 2}));
}

// StopGradientSingleOutputMultiEdgeTest tests combinations of valid and
// 'NoGradient' (induced by StopGradient op) returned along multiple edges from
// a single nodes output.
class StopGradientSingleOutputMultiEdgeTest : public ::testing::Test {
 protected:
  StopGradientSingleOutputMultiEdgeTest() : scope_(Scope::NewRootScope()) {}

  void CheckGrad(const std::vector<bool>& stop_outputs,
                 const Tensor& expected_grad) {
    CHECK_EQ(3, stop_outputs.size());

    auto x = Const(scope_, {{1, 0}, {0, 1}});
    auto y = Const(scope_, {{1, 0}, {0, 1}});
    auto z = MatMul(scope_, x, y);

    // Create three output going edges from 'z'.
    // Add StopGradients according to 'stop_outputs'.
    auto out0 = stop_outputs[0]
                    ? StopGradient(scope_, (Identity(scope_, z))).output
                    : Identity(scope_, z).output;
    auto out1 = stop_outputs[1]
                    ? StopGradient(scope_, (Identity(scope_, z))).output
                    : Identity(scope_, z).output;
    auto out2 = stop_outputs[2]
                    ? StopGradient(scope_, (Identity(scope_, z))).output
                    : Identity(scope_, z).output;

    auto g0 = Const(scope_, {{1, 2}, {3, 4}});
    auto g1 = Const(scope_, {{5, 6}, {7, 8}});
    auto g2 = Const(scope_, {{9, 10}, {11, 12}});

    // Call AddSymbolicGradients and compare against 'expected_grad'.
    std::vector<Output> grad_outputs;
    TF_EXPECT_OK(AddSymbolicGradients(scope_, {out0, out1, out2}, {z},
                                      {g0, g1, g2}, &grad_outputs));

    if (expected_grad.NumElements() > 0) {
      Tensor output;
      test::GetTensor(scope_, grad_outputs[0], &output);
      test::ExpectTensorEqual<int>(output, expected_grad);
    } else {
      EXPECT_EQ(NoGradient(), grad_outputs[0]);
    }
  }

  Scope scope_;
};

TEST_F(StopGradientSingleOutputMultiEdgeTest, ValidGradAllEdges) {
  CheckGrad({false, false, false},
            test::AsTensor<int>({15, 18, 21, 24}, {2, 2}));
}

TEST_F(StopGradientSingleOutputMultiEdgeTest, StopGradFirstEdge) {
  CheckGrad({true, false, false},
            test::AsTensor<int>({14, 16, 18, 20}, {2, 2}));
}

TEST_F(StopGradientSingleOutputMultiEdgeTest, StopGradSecondEdge) {
  CheckGrad({false, true, false},
            test::AsTensor<int>({10, 12, 14, 16}, {2, 2}));
}

TEST_F(StopGradientSingleOutputMultiEdgeTest, StopGradThirdEdge) {
  CheckGrad({false, false, true}, test::AsTensor<int>({6, 8, 10, 12}, {2, 2}));
}

TEST_F(StopGradientSingleOutputMultiEdgeTest, StopGradFirstAndSecondEdges) {
  CheckGrad({true, true, false}, test::AsTensor<int>({9, 10, 11, 12}, {2, 2}));
}

TEST_F(StopGradientSingleOutputMultiEdgeTest, StopGradSecondAndThirdEdges) {
  CheckGrad({false, true, true}, test::AsTensor<int>({1, 2, 3, 4}, {2, 2}));
}

TEST_F(StopGradientSingleOutputMultiEdgeTest, StopGradFirstAndThirdEdges) {
  CheckGrad({true, false, true}, test::AsTensor<int>({5, 6, 7, 8}, {2, 2}));
}

TEST_F(StopGradientSingleOutputMultiEdgeTest, StopGradAllEdges) {
  CheckGrad({true, true, true}, Tensor());
}

// StopGradientMultiOutputTest tests combinations of valid and 'NoGradient'
// (induced by StopGradient op) returned along a single nodes multiple outputs.
class StopGradientMultiOutputTest : public ::testing::Test {
 protected:
  StopGradientMultiOutputTest() : scope_(Scope::NewRootScope()) {}

  void CheckGrad(const std::vector<bool>& stop_outputs,
                 const Tensor& expected_grad) {
    CHECK_EQ(3, stop_outputs.size());
    auto x = ops::Const(scope_, 1, {3, 2, 4});
    auto y = Unstack(scope_, x, 3);
    TF_ASSERT_OK(scope_.status());

    // Add StopGradients according to 'stop_outputs'.
    auto out0 =
        stop_outputs[0] ? StopGradient(scope_, y.output[0]) : y.output[0];
    auto out1 =
        stop_outputs[1] ? StopGradient(scope_, y.output[1]) : y.output[1];
    auto out2 =
        stop_outputs[2] ? StopGradient(scope_, y.output[2]) : y.output[2];

    auto g0 = Const(scope_, {1, 2, 3, 4, 5, 6, 7, 8}, {2, 4});
    auto g1 = Const(scope_, {9, 10, 11, 12, 13, 14, 15, 16}, {2, 4});
    auto g2 = Const(scope_, {17, 18, 19, 20, 21, 22, 23, 24}, {2, 4});

    // Call AddSymbolicGradients and compare against 'expected_grad'.
    std::vector<Output> grad_outputs;
    TF_EXPECT_OK(AddSymbolicGradients(scope_, {out0, out1, out2}, {x},
                                      {g0, g1, g2}, &grad_outputs));

    if (expected_grad.NumElements() > 0) {
      Tensor output;
      test::GetTensor(scope_, grad_outputs[0], &output);
      test::ExpectTensorEqual<int>(output, expected_grad);
    } else {
      EXPECT_EQ(NoGradient(), grad_outputs[0]);
    }
  }

  Scope scope_;
};

TEST_F(StopGradientMultiOutputTest, ValidGradAllOutputs) {
  // clang-format off
  CheckGrad({false, false, false}, test::AsTensor<int>(
    {1, 2, 3, 4, 5, 6, 7, 8,
     9, 10, 11, 12, 13, 14, 15, 16,
     17, 18, 19, 20, 21, 22, 23, 24},
    {3, 2, 4}));
  // clang-format on
}

TEST_F(StopGradientMultiOutputTest, StopGradFirstOutput) {
  // clang-format off
  CheckGrad({true, false, false}, test::AsTensor<int>(
    {0, 0, 0, 0, 0, 0, 0, 0,
     9, 10, 11, 12, 13, 14, 15, 16,
     17, 18, 19, 20, 21, 22, 23, 24},
    {3, 2, 4}));
  // clang-format on
}

TEST_F(StopGradientMultiOutputTest, StopGradSecondOutput) {
  // clang-format off
  CheckGrad({false, true, false}, test::AsTensor<int>(
    {1, 2, 3, 4, 5, 6, 7, 8,
     0, 0, 0, 0, 0, 0, 0, 0,
     17, 18, 19, 20, 21, 22, 23, 24},
    {3, 2, 4}));
  // clang-format on
}

TEST_F(StopGradientMultiOutputTest, StopGradThirdOutput) {
  // clang-format off
  CheckGrad({false, false, true}, test::AsTensor<int>(
    {1, 2, 3, 4, 5, 6, 7, 8,
     9, 10, 11, 12, 13, 14, 15, 16,
     0, 0, 0, 0, 0, 0, 0, 0},
    {3, 2, 4}));
  // clang-format on
}

TEST_F(StopGradientMultiOutputTest, StopGradFirstAndThirdOutputs) {
  // clang-format off
  CheckGrad({true, false, true}, test::AsTensor<int>(
    {0, 0, 0, 0, 0, 0, 0, 0,
     9, 10, 11, 12, 13, 14, 15, 16,
     0, 0, 0, 0, 0, 0, 0, 0},
    {3, 2, 4}));
  // clang-format on
}

TEST_F(StopGradientMultiOutputTest, StopAllOutputs) {
  CheckGrad({true, true, true}, Tensor());
}

}  // namespace
}  // namespace tensorflow
