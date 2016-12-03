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

#include <memory>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/kernels/hexagon/graph_transferer.h"
#include "tensorflow/core/kernels/hexagon/hexagon_ops_definitions.h"
#include "tensorflow/core/kernels/hexagon/i_graph_transfer_ops_definitions.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

const string NAME_A = "a";
const string NAME_B = "b";
const string NAME_A_PLUS_B = "a_plus_b";
constexpr float NODE_A_VAL = 2.0f;
constexpr float NODE_B_VAL = 3.0f;
constexpr float VALUE_TOLERANCE_FLOAT = 1e-8f;

class GraphTransfererTest : public ::testing::Test {
 protected:
  void SetUp() final {
  }

  GraphTransferer gt_;
};

static const std::vector<string> OP_TYPES{"INPUT",   "OUTPUT", "Conv2D",
                                          "MaxPool", "NoOp",   "Add"};
const GraphTransferer::OutputTensorMap EMPTY_OUTPUT_TENSOR_MAP;

class TestGraphTransferOpsDefinitions : public IGraphTransferOpsDefinitions {
 public:
  int GetTotalOpsCount() const final { return OP_TYPES.size(); }
  int GetInputNodeOpId() const final { return GetOpIdFor("INPUT"); }
  int GetOutputNodeOpId() const final { return GetOpIdFor("OUTPUT"); }
  int GetOpIdFor(const string& op_type) const final {
    for (int i = 0; i < OP_TYPES.size(); ++i) {
      if (OP_TYPES[i] == op_type) {
        return i;
      }
    }
    return -1;
  }

 private:
} TEST_GRAPH_TRANSFER_OPS_DEFINITIONS;

static GraphDef CreateAddGraphDef() {
  Scope root = Scope::NewRootScope();
  ops::Output node_a = ops::Const(root.WithOpName(NAME_A), NODE_A_VAL);
  ops::Output node_b = ops::Const(root.WithOpName(NAME_B), NODE_B_VAL);
  ops::Output node_add =
      ops::Add(root.WithOpName(NAME_A_PLUS_B), node_a, node_b);
  GraphDef def;
  TF_CHECK_OK(root.ToGraphDef(&def));
  return def;
}

static GraphDef CreateConvGraphDef() {
  Scope root = Scope::NewRootScope();
  Tensor input_data(DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillIota<float>(&input_data, 1.0f);
  ops::Output input =
      ops::Const(root.WithOpName("input"), ops::Input::Initializer(input_data));
  Tensor filter_data(DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillIota<float>(&filter_data, 1.0f);
  ops::Output filter = ops::Const(root.WithOpName("filter"),
                                  ops::Input::Initializer(filter_data));
  const std::vector<int> strides{1, 1, 1, 1};
  ops::Output conv =
      ops::Conv2D(root.WithOpName("conv"), input, filter, strides, "SAME");
  ops::Output softmax = ops::Softmax(root.WithOpName("softmax"), conv);
  GraphDef def;
  TF_CHECK_OK(root.ToGraphDef(&def));
  return def;
}

static GraphDef CreatePoolGraphDef() {
  Scope root = Scope::NewRootScope();
  Tensor input_data(DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillIota<float>(&input_data, 1.0f);
  ops::Output input =
      ops::Const(root.WithOpName("input"), ops::Input::Initializer(input_data));
  Tensor filter_data(DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillIota<float>(&filter_data, 1.0f);
  ops::Output filter = ops::Const(root.WithOpName("filter"),
                                  ops::Input::Initializer(filter_data));
  const std::vector<int> ksize{1, 1, 1, 1};
  const std::vector<int> padding{0, 0, 0, 0};
  const std::vector<int> strides{1, 1, 1, 1};
  ops::Output max_pool =
      ops::MaxPool(root.WithOpName("maxpool"), input, ksize, strides, "SAME");
  ops::Output softmax = ops::Softmax(root.WithOpName("softmax"), max_pool);
  GraphDef def;
  TF_CHECK_OK(root.ToGraphDef(&def));
  return def;
}

static const GraphTransferer::ConstNodeTransferParams* FindConstNodeParams(
    const GraphTransferer& gt, const string& name) {
  for (const GraphTransferer::ConstNodeTransferParams& params :
       gt.GetConstNodeParams()) {
    if (params.name == name) {
      return &params;
    }
  }
  return nullptr;
}

static const GraphTransferer::NodeTransferParams* FindOpNodeParams(
    const GraphTransferer& gt, const string& name) {
  for (const GraphTransferer::NodeTransferParams& params :
       gt.GetOpNodeParams()) {
    if (params.name == name) {
      return &params;
    }
  }
  return nullptr;
}

static const GraphTransferer::NodeInputParams* FindNodeInputParams(
    const GraphTransferer& gt, const int node_id) {
  for (const GraphTransferer::NodeInputParams& params :
       gt.GetNodeInputParams()) {
    if (params.node_id == node_id) {
      return &params;
    }
  }
  return nullptr;
}

static const GraphTransferer::NodeOutputParams* FindNodeOutputParams(
    const GraphTransferer& gt, const int node_id) {
  for (const GraphTransferer::NodeOutputParams& params :
       gt.GetNodeOutputParams()) {
    if (params.node_id == node_id) {
      return &params;
    }
  }
  return nullptr;
}

static void SanityCheckNodes(const GraphTransferer& gt) {
  for (const GraphTransferer::NodeTransferParams& params :
       gt.GetOpNodeParams()) {
    if (params.inputs_size > 0) {
      const GraphTransferer::NodeInputParams* input_params =
          FindNodeInputParams(gt, params.node_id);
      ASSERT_NE(nullptr, input_params);
      EXPECT_EQ(params.inputs_size,
                input_params->input_node_id_and_output_port_list.size());
      EXPECT_EQ(params.node_id, input_params->node_id);
      for (const std::tuple<int, int>& pair :
           input_params->input_node_id_and_output_port_list) {
        EXPECT_GE(std::get<1>(pair), 0);
      }
    }
    if (params.outputs_size > 0) {
      const GraphTransferer::NodeOutputParams* output_params =
          FindNodeOutputParams(gt, params.node_id);
      ASSERT_NE(nullptr, output_params);
      EXPECT_EQ(params.outputs_size, output_params->max_sizes.size());
      EXPECT_EQ(params.node_id, output_params->node_id);
      for (const int max_size : output_params->max_sizes) {
        EXPECT_GE(max_size, 0);
      }
    }
  }
}

TEST_F(GraphTransfererTest, LoadAddGraph) {
  GraphDef def = CreateAddGraphDef();
  ASSERT_TRUE(gt_.LoadGraphFromProto(TEST_GRAPH_TRANSFER_OPS_DEFINITIONS, def,
                                     {}, std::vector<string>{NAME_A_PLUS_B},
                                     EMPTY_OUTPUT_TENSOR_MAP)
                  .ok());
  SanityCheckNodes(gt_);

  const int const_node_count = gt_.GetConstNodeParams().size();
  ASSERT_EQ(2, const_node_count);
  const GraphTransferer::ConstNodeTransferParams* params_a =
      FindConstNodeParams(gt_, NAME_A);
  ASSERT_TRUE(params_a != nullptr);
  EXPECT_EQ(NAME_A, params_a->name);
  EXPECT_EQ(1, params_a->shape[0]);
  EXPECT_EQ(1, params_a->shape[1]);
  EXPECT_EQ(1, params_a->shape[2]);
  EXPECT_EQ(1, params_a->shape[3]);
  EXPECT_EQ(4, params_a->data_size);

  const GraphTransferer::ConstNodeTransferParams* params_b =
      FindConstNodeParams(gt_, NAME_B);
  ASSERT_TRUE(params_b != nullptr);
  EXPECT_EQ(1, params_b->shape[0]);
  EXPECT_EQ(1, params_b->shape[1]);
  EXPECT_EQ(1, params_b->shape[2]);
  EXPECT_EQ(1, params_b->shape[3]);
  EXPECT_EQ(4, params_b->data_size);
}

TEST_F(GraphTransfererTest, DryRunAddGraphA) {
  GraphDef def = CreateAddGraphDef();
  GraphTransferer::InputNodeInfo input_node_info;
  input_node_info.name = NAME_A;
  input_node_info.tensor = Tensor(DT_FLOAT, {});
  input_node_info.tensor.scalar<float>()() = 1.0f;
  const std::vector<GraphTransferer::InputNodeInfo> inputs{input_node_info};
  std::vector<string> outputs = {NAME_B, NAME_A_PLUS_B};
  std::vector<tensorflow::Tensor> output_tensors;
  Status status = gt_.DryRunInference(
      def, inputs, outputs, false /* initialize_by_zero */, &output_tensors);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(outputs.size(), output_tensors.size());
  EXPECT_NEAR(NODE_B_VAL, output_tensors.at(0).scalar<float>()(),
              VALUE_TOLERANCE_FLOAT);
  EXPECT_NEAR(1.0f + NODE_B_VAL, output_tensors.at(1).scalar<float>()(),
              VALUE_TOLERANCE_FLOAT);
}

TEST_F(GraphTransfererTest, DryRunAddGraphAUninitialized) {
  GraphDef def = CreateAddGraphDef();
  GraphTransferer::InputNodeInfo input_node_info;
  input_node_info.name = NAME_A;
  input_node_info.tensor = Tensor(DT_FLOAT, {});
  const std::vector<GraphTransferer::InputNodeInfo> inputs{input_node_info};
  std::vector<string> outputs = {NAME_B, NAME_A_PLUS_B};
  std::vector<tensorflow::Tensor> output_tensors;
  Status status = gt_.DryRunInference(
      def, inputs, outputs, true /* initialize_by_zero */, &output_tensors);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(outputs.size(), output_tensors.size());
  EXPECT_NEAR(NODE_B_VAL, output_tensors.at(0).scalar<float>()(),
              VALUE_TOLERANCE_FLOAT);
  EXPECT_NEAR(NODE_B_VAL, output_tensors.at(1).scalar<float>()(),
              VALUE_TOLERANCE_FLOAT);
}

TEST_F(GraphTransfererTest, DryRunAddGraphAB) {
  GraphDef def = CreateAddGraphDef();
  GraphTransferer::InputNodeInfo input_node_info_a;
  input_node_info_a.name = NAME_A;
  input_node_info_a.tensor = Tensor(DT_FLOAT, {});
  input_node_info_a.tensor.scalar<float>()() = 1.0f;
  GraphTransferer::InputNodeInfo input_node_info_b;
  input_node_info_b.name = NAME_B;
  input_node_info_b.tensor = Tensor(DT_FLOAT, {});
  input_node_info_b.tensor.scalar<float>()() = 10.0f;
  const std::vector<GraphTransferer::InputNodeInfo> inputs{input_node_info_a,
                                                           input_node_info_b};
  std::vector<string> outputs = {NAME_A_PLUS_B};
  std::vector<tensorflow::Tensor> output_tensors;
  Status status = gt_.DryRunInference(
      def, inputs, outputs, false /* initialize_by_zero */, &output_tensors);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(outputs.size(), output_tensors.size());
  EXPECT_NEAR(11.0f, output_tensors.at(0).scalar<float>()(),
              VALUE_TOLERANCE_FLOAT);
}

TEST_F(GraphTransfererTest, DryRunAddGraphForAllNodes) {
  // Set Node "a" as an input with value (= 1.0f)
  GraphTransferer::InputNodeInfo input_node_info_a;
  input_node_info_a.name = NAME_A;
  input_node_info_a.tensor = Tensor(DT_FLOAT, {});
  input_node_info_a.tensor.scalar<float>()() = 1.0f;

  // Setup dryrun arguments
  const std::vector<GraphTransferer::InputNodeInfo> inputs{input_node_info_a};
  GraphTransferer::OutputTensorInfo output_tensor_info;
  GraphDef def = CreateAddGraphDef();

  // dryrun
  const Status status = GraphTransferer::DryRunInferenceForAllNode(
      def, inputs, false /* initialize_by_zero */, &output_tensor_info);
  const std::vector<Tensor>& output_tensors = output_tensor_info.output_tensors;
  const std::unordered_map<string, Tensor*>& output_tensor_map =
      output_tensor_info.output_tensor_map;
  ASSERT_TRUE(status.ok()) << status;

  // Assert output node count
  ASSERT_EQ(3, output_tensors.size());
  ASSERT_EQ(1, output_tensor_map.count(NAME_A));
  ASSERT_EQ(1, output_tensor_map.count(NAME_B));
  ASSERT_EQ(1, output_tensor_map.count(NAME_A_PLUS_B));

  // Assert output nodes' values
  const float name_b_output = output_tensor_map.at(NAME_B)->scalar<float>()();
  const float name_a_b_output =
      output_tensor_map.at(NAME_A_PLUS_B)->scalar<float>()();
  EXPECT_NEAR(NODE_B_VAL, name_b_output, VALUE_TOLERANCE_FLOAT);
  EXPECT_NEAR(1.0f + NODE_B_VAL, name_a_b_output, VALUE_TOLERANCE_FLOAT);
}

TEST_F(GraphTransfererTest, LoadAddGraphWithOutputTensorMap) {
  GraphDef def = CreateAddGraphDef();
  GraphTransferer::InputNodeInfo input_node_info_a;
  input_node_info_a.name = NAME_A;
  input_node_info_a.tensor = Tensor(DT_FLOAT, {});
  input_node_info_a.tensor.scalar<float>()() = 1.0f;
  const std::vector<GraphTransferer::InputNodeInfo> inputs{input_node_info_a};
  GraphTransferer::OutputTensorInfo output_tensor_info;
  Status status = GraphTransferer::DryRunInferenceForAllNode(
      def, inputs, {}, &output_tensor_info);
  ASSERT_TRUE(status.ok()) << status;
  const GraphTransferer::OutputTensorMap& output_tensor_map =
      output_tensor_info.output_tensor_map;
  const std::vector<string> output_node_names = {NAME_A_PLUS_B};
  status = gt_.LoadGraphFromProto(TEST_GRAPH_TRANSFER_OPS_DEFINITIONS, def,
                                  inputs, output_node_names, output_tensor_map);
  ASSERT_TRUE(status.ok());
}

TEST_F(GraphTransfererTest, LoadConvGraph) {
  GraphDef def = CreateConvGraphDef();
  std::vector<GraphTransferer::InputNodeInfo> input_node_info_list;
  input_node_info_list.emplace_back(
      GraphTransferer::InputNodeInfo{"input", Tensor{DT_FLOAT, {1, 1, 1, 1}}});
  const std::vector<string> output_node_names = {"softmax"};
  ASSERT_TRUE(gt_.LoadGraphFromProto(TEST_GRAPH_TRANSFER_OPS_DEFINITIONS, def,
                                     input_node_info_list, output_node_names,
                                     EMPTY_OUTPUT_TENSOR_MAP)
                  .ok());
  SanityCheckNodes(gt_);
  const int const_node_count = gt_.GetConstNodeParams().size();
  ASSERT_EQ(2, const_node_count);
  const int op_node_count = gt_.GetOpNodeParams().size();
  ASSERT_EQ(3, op_node_count);
  const GraphTransferer::NodeTransferParams* params_conv =
      FindOpNodeParams(gt_, "conv");
  ASSERT_TRUE(params_conv != nullptr);
  const int id = params_conv->node_id;
  EXPECT_GE(id, 0);
  EXPECT_EQ("Conv2D", params_conv->type);
  EXPECT_EQ(3, params_conv->inputs_size);
  EXPECT_EQ(1, params_conv->outputs_size);
  EXPECT_EQ("NN_PAD_SAME", params_conv->padding);
}

TEST_F(GraphTransfererTest, LoadMaxPoolGraph) {
  GraphDef def = CreatePoolGraphDef();
  std::vector<GraphTransferer::InputNodeInfo> input_node_info_list;
  input_node_info_list.emplace_back(
      GraphTransferer::InputNodeInfo{"input", Tensor{DT_FLOAT, {1, 1, 1, 1}}});
  const std::vector<string> output_node_names = {"softmax"};
  ASSERT_TRUE(gt_.LoadGraphFromProto(TEST_GRAPH_TRANSFER_OPS_DEFINITIONS, def,
                                     input_node_info_list, output_node_names,
                                     EMPTY_OUTPUT_TENSOR_MAP)
                  .ok());
  SanityCheckNodes(gt_);
  const int const_node_count = gt_.GetConstNodeParams().size();
  ASSERT_EQ(2, const_node_count);
  const int op_node_count = gt_.GetOpNodeParams().size();
  ASSERT_EQ(3, op_node_count);
  const GraphTransferer::NodeTransferParams* params_max_pool =
      FindOpNodeParams(gt_, "maxpool");
  ASSERT_TRUE(params_max_pool != nullptr);
  const int id = params_max_pool->node_id;
  EXPECT_GE(id, 0);
  EXPECT_EQ("MaxPool", params_max_pool->type);
  EXPECT_EQ(3, params_max_pool->inputs_size);
  EXPECT_EQ(1, params_max_pool->outputs_size);
  EXPECT_EQ("NN_PAD_SAME", params_max_pool->padding);
}

TEST(HexagonOpsDefinitions, CheckOpsDefinitions) {
  const IGraphTransferOpsDefinitions& ops_definitions =
      HexagonOpsDefinitions::getInstance();
  const int total_ops_count = ops_definitions.GetTotalOpsCount();
  EXPECT_GT(total_ops_count, 0);
  const int input_op_id =
      ops_definitions.GetOpIdFor(IGraphTransferOpsDefinitions::INPUT_OP_NAME);
  EXPECT_GE(input_op_id, 0);
  EXPECT_EQ(input_op_id, ops_definitions.GetInputNodeOpId());
  const int output_op_id =
      ops_definitions.GetOpIdFor(IGraphTransferOpsDefinitions::OUTPUT_OP_NAME);
  EXPECT_GE(output_op_id, 0);
  EXPECT_EQ(output_op_id, ops_definitions.GetOutputNodeOpId());
}

TEST(GraphTransferer, LoadGraphFromProtoFile) {
  const IGraphTransferOpsDefinitions* ops_definitions =
      &TEST_GRAPH_TRANSFER_OPS_DEFINITIONS;
  string filename =
      io::JoinPath(testing::TensorFlowSrcRoot(),
                   "core/example/testdata/parse_example_graph_def.pbtxt");
  std::vector<GraphTransferer::InputNodeInfo> input_node_info_list = {};
  std::vector<string> output_node_names = {};
  bool is_text_proto = true;

  // Keep following comments for debugging purpose for now
  // filename = "v3_stripped_quantized_graph_opt.pb";
  // input_node_info_list.emplace_back(
  // GraphTransferer::InputNodeInfo{"Mul", Tensor{DT_FLOAT, {1,299,299,3}}});
  // output_node_names.emplace_back("softmax");
  // is_text_proto = false;
  // ops_definitions = &HexagonOpsDefinitions::getInstance();

  GraphTransferer::OutputTensorInfo output_tensor_info;
  GraphTransferer gt;
  gt.EnableStrictCheckMode(false);
  Status status = gt.LoadGraphFromProtoFile(
      *ops_definitions, filename, input_node_info_list, output_node_names,
      is_text_proto, true, &output_tensor_info);
  // TODO(satok): Uncomment following assert once we fix the loader problem
  // ASSERT_TRUE(status.ok()) << status;
}

}  // namespace tensorflow
