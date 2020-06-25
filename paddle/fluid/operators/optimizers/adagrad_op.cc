/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/optimizers/adagrad_op.h"
#include <vector>

#include <cmath>

#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
class AdagradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Param"), "Input", "Param", "Adagrad");
    OP_INOUT_CHECK(ctx->HasInput("Grad"), "Input", "Grad", "Adagrad");
    OP_INOUT_CHECK(ctx->HasInput("Moment"), "Input", "Moment", "Adagrad");
    OP_INOUT_CHECK(ctx->HasInput("LearningRate"), "Input", "LearningRate",
                   "Adagrad");
    OP_INOUT_CHECK(ctx->HasOutput("ParamOut"), "Output", "ParamOut", "Adagrad");
    OP_INOUT_CHECK(ctx->HasOutput("MomentOut"), "Output", "MomentOut",
                   "Adagrad");

    auto lr_dims = ctx->GetInputDim("LearningRate");
    PADDLE_ENFORCE_NE(framework::product(lr_dims), 0,
                      platform::errors::InvalidArgument(
                          "Maybe the Input variable LearningRate has not "
                          "been initialized. You may need to confirm "
                          "if you put exe.run(startup_program) "
                          "after optimizer.minimize function."));
    PADDLE_ENFORCE_EQ(framework::product(lr_dims), 1,
                      platform::errors::InvalidArgument(
                          "LearningRate should have one element"));
    auto param_dims = ctx->GetInputDim("Param");
    PADDLE_ENFORCE_EQ(
        param_dims, ctx->GetInputDim("Grad"),
        platform::errors::InvalidArgument("Param and Grad input of AdagradOp "
                                          "should have the same dimension."));
    PADDLE_ENFORCE_EQ(
        param_dims, ctx->GetInputDim("Moment"),
        platform::errors::InvalidArgument("Param and Moment input of AdagradOp "
                                          "should have the same dimension."));

    ctx->SetOutputDim("ParamOut", param_dims);
    ctx->SetOutputDim("MomentOut", param_dims);
  }
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
  #ifdef PADDLE_WITH_MKLDNN
    if (platform::CanMKLDNNBeUsed(ctx)) {
      library = framework::LibraryType::kMKLDNN;
      layout = framework::DataLayout::kMKLDNN;
    }
  #endif

    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(ctx, "Param"),
                                   ctx.GetPlace(), layout, library);
  }
};

class AdagradOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param", "(Tensor) Input parameter");
    AddInput("Grad", "(Tensor) Input gradient");
    AddInput("Moment", "(Tensor) Second moment");
    AddInput("LearningRate", "(Tensor) Learning rate");

    AddOutput("ParamOut", "(Tensor) Output parameter");
    AddOutput("MomentOut", "(Tensor) Output second moment");

    AddAttr<float>("epsilon",
                   "(float, default 1.0e-6) "
                   "Constant for numerical stability")
        .SetDefault(1.0e-6f);
    AddAttr<bool>("use_mkldnn",
                "(bool, default false) Only used in mkldnn kernel")
      .SetDefault(false);
    AddComment(R"DOC(

Adaptive Gradient Algorithm (Adagrad).

The update is done as follows:

$$moment\_out = moment + grad * grad \\
param\_out = param - \frac{learning\_rate * grad}{\sqrt{moment\_out} + \epsilon}
$$

The original paper(http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
does not have the epsilon attribute. It is added here in our implementation
as also proposed here: http://cs231n.github.io/neural-networks-3/#ada
for numerical stability to avoid the division by zero error.

)DOC");
  }
};

namespace {
size_t FindPos(const std::vector<int64_t>& rows, int64_t value) {
  return std::find(rows.begin(), rows.end(), value) - rows.begin();
}
}  // namespace

template <typename T>
struct SparseAdagradFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::SelectedRows& grad,
                  const framework::Tensor& learning_rate, T epsilon,
                  framework::Tensor* moment, framework::Tensor* param) {
    // 1. g_m.rows = set(g.rows)
    auto grad_width = grad.value().dims()[1];
    math::scatter::MergeAdd<platform::CPUDeviceContext, T> merge_func;
    auto grad_merge = merge_func(context, grad);
    auto& merge_rows = grad_merge.rows();
    auto* grad_merge_data = grad_merge.mutable_value()->template data<T>();

    // 2. m += g_m * g_m
    auto grad_square =
        SquareSelectedRows<platform::CPUDeviceContext, T>(context, grad_merge);

    math::SelectedRowsAddToTensor<platform::CPUDeviceContext, T> functor;
    functor(context, grad_square, moment);

    // 3. update parameter
    auto* lr = learning_rate.data<T>();
    auto* param_data = param->data<T>();
    auto* moment_data = moment->data<T>();

    for (size_t i = 0; i < merge_rows.size(); i++) {
      for (int64_t j = 0; j < grad_width; j++) {
        param_data[merge_rows[i] * grad_width + j] -=
            lr[0] * grad_merge_data[i * grad_width + j] /
            (std::sqrt(moment_data[merge_rows[i] * grad_width + j]) + epsilon);
      }
    }
  }
};

template struct SparseAdagradFunctor<platform::CPUDeviceContext, float>;
template struct SparseAdagradFunctor<platform::CPUDeviceContext, double>;

template <typename T>
class AdagradMKLDNNKernel : public paddle::framework::OpKernel<T> {
public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    // std::cout<<"@@@@@@@@@@@@@@@@@@@@ Working\n";

    auto& dev_ctx = ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    const auto* param = ctx.Input<Tensor>("Param");
    const auto* grad = ctx.Input<Tensor>("Grad");
    const auto* moment = ctx.Input<Tensor>("Moment");
    const auto *learning_rate = ctx.Input<Tensor>("LearningRate");

    auto* param_out = ctx.Output<Tensor>("ParamOut");
    auto* moment_out = ctx.Output<Tensor>("MomentOut");

    T epsilon = static_cast<T>(ctx.Attr<float>("epsilon"));

    auto* param_data = param->data<T>();
    auto* grad_data = grad->data<T>();
    auto* moment_data = moment->data<T>();
    auto* learning_rate_data = learning_rate->data<T>();

    auto* param_out_data = param_out->mutable_data<T>(ctx.GetPlace());
    auto* moment_out_data = moment_out->mutable_data<T>(ctx.GetPlace());

    auto numel = param->numel();

    dnnl::stream engine_stream(mkldnn_engine);
    if (dev_ctx.GetBlob("@grad_m") == nullptr) {
      auto common_md = dnnl::memory::desc({numel}, platform::MKLDNNGetDataType<T>(), MKLDNNMemoryFormat::a);

      // MOMENT_OUT CALCULATIONS
      memcpy(moment_out_data, moment_data, sizeof(T) * numel);
      auto grad_m = std::make_shared<dnnl::memory>(common_md, mkldnn_engine, platform::to_void_cast<T>(grad_data));
      dev_ctx.SetBlob("@grad_m", grad_m);

      auto m_out_m = std::make_shared<dnnl::memory>(common_md, mkldnn_engine, platform::to_void_cast<T>(moment_out_data));
      dev_ctx.SetBlob("@m_out_m", m_out_m);

      auto binary_d = dnnl::binary::desc(dnnl::algorithm::binary_mul, common_md, common_md, common_md);

      dnnl::post_ops binary_ops;
      binary_ops.append_sum(1.0f);
      dnnl::primitive_attr binary_attr;
      binary_attr.set_post_ops(binary_ops);

      auto binary_pd = dnnl::binary::primitive_desc(binary_d, binary_attr, mkldnn_engine);
      auto binary_prim = std::make_shared<dnnl::binary>(binary_pd);
      dev_ctx.SetBlob("@binary_prim", binary_prim);

      binary_prim->execute(engine_stream, {{DNNL_ARG_SRC_0, *grad_m},
                                          {DNNL_ARG_SRC_1, *grad_m},
                                          {DNNL_ARG_DST, *m_out_m}});
      engine_stream.wait();

      // PARAM_OUT CALCULATIONS
      memcpy(param_out_data, param_data, sizeof(T) * numel);

      auto intermediate_m = std::make_shared<dnnl::memory>(common_md, mkldnn_engine);
      dev_ctx.SetBlob("@intermediate_m", intermediate_m);

      auto w_out_m = std::make_shared<dnnl::memory>(common_md, mkldnn_engine, platform::to_void_cast<T>(param_out_data));
      dev_ctx.SetBlob("@w_out_m", w_out_m);

      auto eltwise_sqrt_d = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference, dnnl::algorithm::eltwise_sqrt,
                                                        common_md, 0.f, 0.f);
      auto eltwise_sqrt_pd = dnnl::eltwise_forward::primitive_desc(eltwise_sqrt_d, mkldnn_engine);
      auto eltwise_sqrt_prim = std::make_shared<dnnl::eltwise_forward>(eltwise_sqrt_pd);
      dev_ctx.SetBlob("@eltwise_sqrt_prim", eltwise_sqrt_prim);

      auto eltwise_linear_d = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference, dnnl::algorithm::eltwise_linear,
                                                          common_md, 1.f, epsilon);
      auto eltwise_linear_pd = dnnl::eltwise_forward::primitive_desc(eltwise_linear_d, mkldnn_engine);
      auto eltwise_linear_prim = std::make_shared<dnnl::eltwise_forward>(eltwise_linear_pd);
      dev_ctx.SetBlob("@eltwise_linear_prim", eltwise_linear_prim);

      auto eltwise_pow_d = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference, dnnl::algorithm::eltwise_pow,
                                                          common_md, -(learning_rate_data[0]), -1.f);
      auto eltwise_pow_pd = dnnl::eltwise_forward::primitive_desc(eltwise_pow_d, mkldnn_engine);
      auto eltwise_pow_prim = std::make_shared<dnnl::eltwise_forward>(eltwise_pow_pd);
      dev_ctx.SetBlob("@eltwise_pow_prim", eltwise_pow_prim);

      eltwise_sqrt_prim->execute(engine_stream, {{DNNL_ARG_SRC, *m_out_m},
                                              {DNNL_ARG_DST, *intermediate_m}});
      eltwise_linear_prim->execute(engine_stream, {{DNNL_ARG_SRC, *intermediate_m},
                                                  {DNNL_ARG_DST, *intermediate_m}});
      eltwise_pow_prim->execute(engine_stream, {{DNNL_ARG_SRC, *intermediate_m},
                                                {DNNL_ARG_DST, *intermediate_m}});
      binary_prim->execute(engine_stream, {{DNNL_ARG_SRC_0, *grad_m},
                                          {DNNL_ARG_SRC_1, *intermediate_m},
                                          {DNNL_ARG_DST, *w_out_m}});
      engine_stream.wait();
    } else {
      // MOMENT_OUT CALCULATIONS
      memcpy(moment_out_data, moment_data, sizeof(T) * numel);
      auto grad_m = std::static_pointer_cast<dnnl::memory>(dev_ctx.GetBlob("@grad_m"));
      grad_m->set_data_handle(platform::to_void_cast<T>(grad_data));
      auto m_out_m = std::static_pointer_cast<dnnl::memory>(dev_ctx.GetBlob("@m_out_m"));
      m_out_m->set_data_handle(platform::to_void_cast<T>(moment_out_data));

      auto binary_prim = std::static_pointer_cast<dnnl::binary>(dev_ctx.GetBlob("@binary_prim"));

      binary_prim->execute(engine_stream, {{DNNL_ARG_SRC_0, *grad_m},
                                          {DNNL_ARG_SRC_1, *grad_m},
                                          {DNNL_ARG_DST, *m_out_m}});
      engine_stream.wait();

      // PARAM_OUT CALCULATIONS
      memcpy(param_out_data, param_data, sizeof(T) * numel);
      auto intermediate_m = std::static_pointer_cast<dnnl::memory>(dev_ctx.GetBlob("@intermediate_m"));
      auto w_out_m = std::static_pointer_cast<dnnl::memory>(dev_ctx.GetBlob("@w_out_m"));
      w_out_m->set_data_handle(platform::to_void_cast<T>(param_out_data));

      auto eltwise_sqrt_prim = std::static_pointer_cast<dnnl::eltwise_forward>(dev_ctx.GetBlob("@eltwise_sqrt_prim"));
      auto eltwise_linear_prim = std::static_pointer_cast<dnnl::binary>(dev_ctx.GetBlob("@eltwise_linear_prim"));
      auto eltwise_pow_prim = std::static_pointer_cast<dnnl::binary>(dev_ctx.GetBlob("@eltwise_pow_prim"));

      eltwise_sqrt_prim->execute(engine_stream, {{DNNL_ARG_SRC, *m_out_m},
                                              {DNNL_ARG_DST, *intermediate_m}});
      eltwise_linear_prim->execute(engine_stream, {{DNNL_ARG_SRC, *intermediate_m},
                                                  {DNNL_ARG_DST, *intermediate_m}});
      eltwise_pow_prim->execute(engine_stream, {{DNNL_ARG_SRC, *intermediate_m},
                                              {DNNL_ARG_DST, *intermediate_m}});
      binary_prim->execute(engine_stream, {{DNNL_ARG_SRC_0, *grad_m},
                                          {DNNL_ARG_SRC_1, *intermediate_m},
                                          {DNNL_ARG_DST, *w_out_m}});
      engine_stream.wait();
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(adagrad, ops::AdagradOp, ops::AdagradOpMaker);
REGISTER_OP_CPU_KERNEL(
    adagrad, ops::AdagradOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::AdagradOpKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_KERNEL(adagrad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::AdagradMKLDNNKernel<float>);
