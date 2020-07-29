// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/ir/mkldnn/bidirectional_gru_mkldnn_fuse_pass.h"

#include <gtest/gtest.h>
#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle {
namespace framework {
namespace ir {

void SetOp(ProgramDesc* prog, const std::string& type, const std::string& name,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs, bool use_mkldnn = false) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType(type);
  op->SetAttr("name", name);
  if (type == "fusion_gru") {
    op->SetAttr("use_mkldnn", use_mkldnn);
    op->SetInput("X", {inputs[0]});
    op->SetInput("WeightX", {inputs[1]});
    op->SetInput("WeightH", {inputs[2]});
    op->SetInput("Bias", {inputs[3]});
    op->SetOutput("Hidden", outputs);
  } else if (type == "concat") {
    op->SetAttr("use_mkldnn", use_mkldnn);
    op->SetInput("X", inputs);
    op->SetOutput("Out", outputs);
  }
  op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
              static_cast<int>(OpRole::kForward));
}

ProgramDesc BuildProgramDesc(std::string activation) {
  ProgramDesc prog;
  for (auto& v :
       std::vector<std::string>({"gru_input", "gru1_WeightX", "gru1_WeightH", "gru1_Bias", "gru1_Hidden",
                                 "gru2_WeightX", "gru2_WeightH", "gru2_Bias", "gru2_Hidden", "concat_Out"})) {
    auto* var = prog.MutableBlock(0)->Var(v);
    var->SetType(proto::VarType::SELECTED_ROWS);
  }

  SetOp(&prog, "fusion_gru", "gru1",
        std::vector<std::string>({"gru_input", "gru1_WeightX", "gru1_WeightH", "gru1_Bias"}),
        std::vector<std::string>({"gru1_Hidden"}), true);
  SetOp(&prog, "fusion_gru", "gru2",
        std::vector<std::string>({"gru_input", "gru2_WeightX", "gru2_WeightH", "gru2_Bias"}),
        std::vector<std::string>({"gru2_Hidden"}), true);
  SetOp(&prog, "concat", "concat", std::vector<std::string>({"gru1_Hidden", "gru2_Hidden"}),
        std::vector<std::string>({"concat_Out"}));
  return prog;
}

void MainTest(std::string activation) {
  auto prog = BuildProgramDesc(activation);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  auto pass =
      PassRegistry::Instance().Get("bidirectional_gru_mkldnn_fuse_pass");

  int original_nodes_num = graph->Nodes().size();

  graph.reset(pass->Apply(graph.release()));

  int current_nodes_num = graph->Nodes().size();

  std::cout<<"Original nodes num: "<<original_nodes_num<<" Current nodes num: "<<current_nodes_num<<std::endl;

  // Remove 3 Nodes: CONV, activation, conv_out
  // Add 1 Node: ConvActivation
  // EXPECT_EQ(original_nodes_num - 2, current_nodes_num);

  // Assert conv_activation op in newly generated graph
  // int conv_activation_count = 0;

  // for (auto* node : graph->Nodes()) {
  //   if (node->IsOp() && node->Op()->Type() == "conv2d") {
  //     auto* op = node->Op();
  //     ASSERT_TRUE(op->HasAttr("use_mkldnn"));
  //     EXPECT_TRUE(BOOST_GET_CONST(bool, op->GetAttr("use_mkldnn")));
  //     auto op_name = BOOST_GET_CONST(std::string, op->GetAttr("name"));
  //     if (op->GetAttrIfExists<std::string>("fuse_activation") == activation) {
  //       ++conv_activation_count;
  //     }
  //     // check if only "conv1" convolution is fused
  //     if (op_name == "conv1") {
  //       ASSERT_TRUE(op->HasAttr("fuse_activation"));
  //     } else if (op_name == "conv2") {
  //       ASSERT_FALSE(op->HasAttr("fuse_activation"));
  //     }
  //   }
  // }
  // EXPECT_EQ(conv_activation_count, 1);
}

TEST(BidirectionalGRUFusePass, bidirectional_gru_mkldnn_fuse_pass) { MainTest("concat"); }

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(bidirectional_gru_mkldnn_fuse_pass);
