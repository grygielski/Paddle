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
#include <string>
#include <vector>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

void BidirectionalGRUFusePass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);

  GraphPatternDetector gpd;
  auto* gru_input = gpd.mutable_pattern()
                        ->NewNode(name_scope_ + "/gru_input")
                        ->AsInput()
                        ->assert_is_op_input("fusion_gru", "X");
  patterns::BidirectionalFusionGRU bidirectional_gru_pattern(
                                    gpd.mutable_pattern(), "bidirectional_gru");
  bidirectional_gru_pattern(gru_input, fuse_type());

  auto* scope = param_scope();

  int found_bidirectional_gru_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle GRU + GRU + " + fuse_type() + " fuse";

    // Ops
    GET_IR_NODE_FROM_SUBGRAPH(fusion_gru1, fusion_gru1, bidirectional_gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fusion_gru2, fusion_gru2, bidirectional_gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(post_op, post_op, bidirectional_gru_pattern);

    // GRU_1 Nodes
    GET_IR_NODE_FROM_SUBGRAPH(fusion_gru1_WeightX, fusion_gru1_WeightX, bidirectional_gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fusion_gru1_WeightH, fusion_gru1_WeightH, bidirectional_gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fusion_gru1_Bias, fusion_gru1_Bias, bidirectional_gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fusion_gru1_Hidden, fusion_gru1_Hidden, bidirectional_gru_pattern);

    // GRU_2 Nodes
    GET_IR_NODE_FROM_SUBGRAPH(fusion_gru2_WeightX, fusion_gru2_WeightX, bidirectional_gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fusion_gru2_WeightH, fusion_gru2_WeightH, bidirectional_gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fusion_gru2_Bias, fusion_gru2_Bias, bidirectional_gru_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fusion_gru2_Hidden, fusion_gru2_Hidden, bidirectional_gru_pattern);

    // Post_Op Output
    GET_IR_NODE_FROM_SUBGRAPH(post_op_output, post_op_output, bidirectional_gru_pattern);

    // auto fusion_gru1_bias_tensor = scope->FindVar(fusion_gru1_Bias->Name())->Get<framework::LoDTensor>();
    // auto* fusion_gru1_bias_data = fusion_gru1_bias_tensor.data<float>();

    // auto fusion_gru2_bias_tensor = scope->FindVar(fusion_gru2_Bias->Name())->Get<framework::LoDTensor>();
    // auto* fusion_gru2_bias_data = fusion_gru2_bias_tensor.data<float>();

    // auto fusion_gru1_weightx_tensor = scope->FindVar(fusion_gru1_WeightX->Name())->Get<framework::LoDTensor>();
    // auto* fusion_gru1_weightx_data = fusion_gru1_weightx_tensor.data<float>();

    // auto fusion_gru2_weightx_tensor = scope->FindVar(fusion_gru2_WeightX->Name())->Get<framework::LoDTensor>();
    // auto* fusion_gru2_weightx_data = fusion_gru2_weightx_tensor.data<float>();

    // Transform first fusion_gru into bidirectional gru with fuse_type() post operation.
    OpDesc* desc = fusion_gru1->Op();
    desc->SetAttr("bidirectional_type", fuse_type());
    desc->SetOutput("Hidden", std::vector<std::string>({post_op_output->Name()}));

    // Resize tensors and set proper values for weights and bias tensors
    auto* gru1_weightx_tensor = scope->FindVar(fusion_gru1_WeightX->Name())->GetMutable<framework::LoDTensor>();
    auto* gru2_weightx_tensor = scope->FindVar(fusion_gru2_WeightX->Name())->GetMutable<framework::LoDTensor>();

    auto* gru1_wx_data = gru1_weightx_tensor->data<float>();
    auto* gru2_wx_data = gru2_weightx_tensor->data<float>();

    auto wx_dims = gru1_weightx_tensor->dims();
    auto wx_numel = gru1_weightx_tensor->numel();
    auto* bidirectional_gru_wx_data = gru1_weightx_tensor->mutable_data<float>({2 * wx_dims[0], wx_dims[1]}, platform::CPUPlace());
    
    memcpy(bidirectional_gru_wx_data, gru1_wx_data, wx_numel * sizeof(float));
    memcpy(bidirectional_gru_wx_data + wx_numel, gru2_wx_data, wx_numel * sizeof(float));


    auto* gru1_weighth_tensor = scope->FindVar(fusion_gru1_WeightH->Name())->GetMutable<framework::LoDTensor>();
    auto* gru2_weighth_tensor = scope->FindVar(fusion_gru2_WeightH->Name())->GetMutable<framework::LoDTensor>();

    auto* gru1_wh_data = gru1_weighth_tensor->data<float>();
    auto* gru2_wh_data = gru2_weighth_tensor->data<float>();

    auto wh_dims = gru1_weighth_tensor->dims();
    auto wh_numel = gru1_weighth_tensor->numel();
    auto* bidirectional_gru_wh_data = gru1_weighth_tensor->mutable_data<float>({2 * wh_dims[0], wh_dims[1]}, platform::CPUPlace());
    
    memcpy(bidirectional_gru_wh_data, gru1_wh_data, wh_numel * sizeof(float));
    memcpy(bidirectional_gru_wh_data + wh_numel, gru2_wh_data, wh_numel * sizeof(float));


    auto* gru1_bias_tensor = scope->FindVar(fusion_gru1_Bias->Name())->GetMutable<framework::LoDTensor>();
    auto* gru2_bias_tensor = scope->FindVar(fusion_gru2_Bias->Name())->GetMutable<framework::LoDTensor>();

    auto* gru1_bias_data = gru1_bias_tensor->data<float>();
    auto* gru2_bias_data = gru2_bias_tensor->data<float>();

    auto bias_dims = gru1_bias_tensor->dims();
    auto bias_numel = gru1_bias_tensor->numel();
    auto* bidirectional_gru_bias_data = gru1_bias_tensor->mutable_data<float>({2 * bias_dims[0], bias_dims[1]}, platform::CPUPlace());
    
    memcpy(bidirectional_gru_bias_data, gru1_bias_data, bias_numel * sizeof(float));
    memcpy(bidirectional_gru_bias_data + bias_numel, gru2_bias_data, bias_numel * sizeof(float));

    GraphSafeRemoveNodes(graph, {post_op, fusion_gru2, fusion_gru2_WeightX, fusion_gru2_WeightH, fusion_gru2_Bias, fusion_gru2_Hidden, fusion_gru1_Hidden});

    IR_NODE_LINK_TO(fusion_gru1, post_op_output);
    found_bidirectional_gru_count++;
  };

  gpd(graph, handler);

  AddStatis(found_bidirectional_gru_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(bidirectional_gru_mkldnn_fuse_pass,
              paddle::framework::ir::BidirectionalGRUFusePass);

REGISTER_PASS(bidirectional_gru_sum_mkldnn_fuse_pass,
              paddle::framework::ir::BidirectionalGRUSumFusePass);
