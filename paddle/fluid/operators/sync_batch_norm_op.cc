/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/batch_norm_op.h"

namespace paddle {
namespace operators {
template <typename T>
class BatchNormGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  std::unique_ptr<T> Apply() const override {
    auto *op = new T();
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));

    op->SetInput("Scale", this->Input("Scale"));
    op->SetInput("Bias", this->Input("Bias"));
    op->SetInput("SavedMean", this->Output("SavedMean"));
    op->SetInput("SavedVariance", this->Output("SavedVariance"));

    // used when setting use_global_stats True during training
    if (boost::get<bool>(this->GetAttr("use_global_stats"))) {
      op->SetInput("Mean", this->Output("MeanOut"));
      op->SetInput("Variance", this->Output("VarianceOut"));
    }

    op->SetAttrMap(this->Attrs());

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Scale"), this->InputGrad("Scale"));
    op->SetOutput(framework::GradVarName("Bias"), this->InputGrad("Bias"));

    return std::unique_ptr<T>(op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sync_batch_norm, ops::BatchNormOp, ops::BatchNormOpMaker,
                  ops::BatchNormOpInferVarType,
                  ops::BatchNormGradMaker<paddle::framework::OpDesc>,
                  ops::BatchNormGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(sync_batch_norm_grad, ops::BatchNormGradOp);
