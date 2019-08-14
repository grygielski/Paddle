/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <unordered_map>
#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using mkldnn::memory;
using mkldnn::primitive;
using mkldnn::reorder;
using mkldnn::stream;
using platform::to_void_cast;
using platform::GetMKLDNNFormat;

inline void GetWeightsTz(std::vector<int64_t>& weights_tz,
                         int groups,  // NOLINT
                         bool is_conv3d) {
  if (groups > 1) {
    if (is_conv3d) {
      int output = weights_tz[0];
      int input = weights_tz[1];
      int dimension = weights_tz[2];
      int height = weights_tz[3];
      int width = weights_tz[4];
      weights_tz.resize(6);
      weights_tz[0] = groups;
      weights_tz[1] = output / groups;
      weights_tz[2] = input;
      weights_tz[3] = dimension;
      weights_tz[4] = height;
      weights_tz[5] = width;
    } else {
      int output = weights_tz[0];
      int input = weights_tz[1];
      int height = weights_tz[2];
      int width = weights_tz[3];
      weights_tz.resize(5);
      weights_tz[0] = groups;
      weights_tz[1] = output / groups;
      weights_tz[2] = input;
      weights_tz[3] = height;
      weights_tz[4] = width;
    }
  }
}

inline mkldnn::memory::format_tag GetWeightsFormat(
    mkldnn::memory::format_tag format, int groups, bool is_conv3d) {
  if (is_conv3d) {
    return (groups == 1) ? format : mkldnn::memory::format_tag::goidhw;
  } else {
    return (groups == 1) ? format : mkldnn::memory::format_tag::goihw;
  }
}

static mkldnn::memory::data_type GetDstType(bool is_int8,
                                            bool force_fp32_output,
                                            bool fuse_relu, bool fuse_brelu,
                                            bool fuse_residual_conn,
                                            const Tensor* residual_param) {
  auto dst_dt = mkldnn::memory::data_type::f32;  // uint8_t, int8_t, float
  if (is_int8) {
    dst_dt = (fuse_relu || fuse_brelu) ? mkldnn::memory::data_type::u8
                                       : mkldnn::memory::data_type::s8;
    if (force_fp32_output) {
      dst_dt = mkldnn::memory::data_type::f32;
    }
    if (fuse_residual_conn && residual_param) {
      auto residual_dt = framework::ToMKLDNNDataType(residual_param->type());
      if (dst_dt != residual_dt) dst_dt = residual_dt;
    }
  }
  return dst_dt;
}

template <typename T, typename K>
class ConvMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");
    bool is_INT8 =
        std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value;
    if (!is_INT8) {
      ComputeFP32(ctx);
    } else {
      bool fuse_relu = ctx.Attr<bool>("fuse_relu");
      bool fuse_residual_conn = ctx.Attr<bool>("fuse_residual_connection");
      bool fuse_brelu = ctx.Attr<bool>("fuse_brelu");
      bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");
      auto residual_param = ctx.Input<Tensor>("ResidualData");
      auto dst_dt = GetDstType(true, force_fp32_output, fuse_relu, fuse_brelu,
                               fuse_residual_conn, residual_param);
      if (dst_dt == mkldnn::memory::data_type::f32) {
        ComputeINT8<float>(ctx);
      } else if (dst_dt == mkldnn::memory::data_type::u8) {
        ComputeINT8<uint8_t>(ctx);
      } else if (dst_dt == mkldnn::memory::data_type::s8) {
        ComputeINT8<int8_t>(ctx);
      }
    }
  }

  void ComputeFP32(const paddle::framework::ExecutionContext& ctx) const {
    const bool is_test = ctx.Attr<bool>("is_test");

    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto* input = ctx.Input<Tensor>("Input");
    auto* filter = ctx.Input<Tensor>("Filter");
    auto* bias = ctx.HasInput("Bias") ? ctx.Input<Tensor>("Bias") : nullptr;
    auto* output = ctx.Output<Tensor>("Output");

    PADDLE_ENFORCE(input->layout() == DataLayout::kMKLDNN &&
                       input->format() != memory::format_tag::undef,
                   "Wrong layout/format set for Input tensor");
    PADDLE_ENFORCE(filter->layout() == DataLayout::kMKLDNN &&
                       filter->format() != memory::format_tag::undef,
                   "Wrong layout/format set for Filter tensor");
    PADDLE_ENFORCE(input->dims().size() == 4 || input->dims().size() == 5,
                   "Input must be with 4 or 5 dimensions, i.e. NCHW or NCDHW");
    PADDLE_ENFORCE(filter->dims().size() == 4 || filter->dims().size() == 5,
                   "Filter must be with 4 or 5 dimensions, i.e. OIHW or OIDHW");
    if (bias) {
      PADDLE_ENFORCE(bias->layout() == DataLayout::kMKLDNN &&
                         bias->format() != memory::format_tag::undef,
                     "Wrong layout/format set for Bias tensor");
      PADDLE_ENFORCE(bias->dims().size() == 1,
                     "Bias must only have 1 dimension, i.e. X");
    }

    // TODO(grygielski) change
    std::vector<int> strides_temp = ctx.Attr<std::vector<int>>("strides");
    std::vector<int64_t> strides(begin(strides_temp), end(strides_temp));

    // TODO(grygielski) change
    std::vector<int> paddings_temp = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int64_t> paddings(begin(paddings_temp), end(paddings_temp));

    // TODO(grygielski) change
    std::vector<int> dilations_temp = ctx.Attr<std::vector<int>>("dilations");
    std::vector<int64_t> dilations(begin(dilations_temp), end(dilations_temp));

    bool fuse_relu = ctx.Attr<bool>("fuse_relu");
    bool fuse_residual_conn = ctx.Attr<bool>("fuse_residual_connection");
    bool fuse_brelu = false;
    float fuse_brelu_threshold = 6.0;
    int groups = ctx.Attr<int>("groups");
    bool is_conv3d = strides.size() == 3U;
    if (!is_conv3d) {
      fuse_brelu = ctx.Attr<bool>("fuse_brelu");
      fuse_brelu_threshold = ctx.Attr<float>("fuse_brelu_threshold");
    }
    // TODO(tpatejko): add support for dilation
    PADDLE_ENFORCE(
        is_conv3d
            ? dilations.size() == 3 && dilations[0] == 1 && dilations[1] == 1 &&
                  dilations[2] == 1
            : dilations.size() == 2 && dilations[0] == 1 && dilations[1] == 1,
        "dilation in convolution is not implemented yet");

    const T* input_data = input->data<T>();
    const T* filter_data = filter->data<T>();

    // TODO(grygielski)
    std::vector<int64_t> src_tz = paddle::framework::vectorize(input->dims());
    std::vector<int64_t> weights_tz =
        paddle::framework::vectorize(filter->dims());

    int g = std::max(groups, 1);
    GetWeightsTz(weights_tz, g, is_conv3d);
    std::vector<int64_t> dst_tz = paddle::framework::vectorize(output->dims());

    // Get unique name for storing MKLDNN primitives
    const std::string key = platform::ConvMKLDNNHandler::GetHash(
        src_tz, weights_tz, fuse_relu, fuse_brelu, strides, paddings, dilations,
        groups, ctx.op().Input("Input") + ctx.op().Input("Filter"));

    std::vector<primitive> pipeline;

    auto src_format = input->format();
    mkldnn::memory::format_tag weights_format =
        GetWeightsFormat(filter->format(), g, is_conv3d);

    auto user_src_md = platform::MKLDNNMemDesc(
        {src_tz}, platform::MKLDNNGetDataType<T>(), src_format);
    auto user_weights_md = platform::MKLDNNMemDesc(
        {weights_tz}, platform::MKLDNNGetDataType<T>(), weights_format);

    /* create memory descriptor for convolution without specified format
     * ('any') which lets a primitive (convolution in this case) choose
     * the memory format preferred for best performance
     */
    std::string data_format = ctx.Attr<std::string>("data_format");
    auto chosen_memory_format =
        platform::data_format_to_memory_format(data_format);

    weights_format = mkldnn::memory::format_tag::any;
    // Check the format for user's special output
    if (chosen_memory_format != mkldnn::memory::format_tag::any) {
      if (is_conv3d) {
        chosen_memory_format =
            platform::MKLDNNFormatForSize(src_tz.size(), chosen_memory_format);
      }
    }
    auto src_md = platform::MKLDNNMemDesc(
        src_tz, platform::MKLDNNGetDataType<T>(), chosen_memory_format);
    auto weights_md = platform::MKLDNNMemDesc(
        weights_tz, platform::MKLDNNGetDataType<T>(), weights_format);
    std::vector<int64_t>
        bias_tz;  // TODO(mgallus): avoid empty vector creation.
                  // Currently used whenever bias is != nullptr.
    auto dst_md = platform::MKLDNNMemDesc(
        dst_tz, platform::MKLDNNGetDataType<T>(), chosen_memory_format);

    platform::ConvMKLDNNHandler handler(dev_ctx, mkldnn_engine, key);

    // create a conv primitive descriptor and save it for usage in backward
    std::shared_ptr<mkldnn::convolution_forward::primitive_desc> conv_pd;
    auto fwd_prop_kind = is_test ? mkldnn::prop_kind::forward_inference
                                 : mkldnn::prop_kind::forward_training;
    if (bias) {
      bias_tz = paddle::framework::vectorize(bias->dims());
      auto bias_md = platform::MKLDNNMemDesc(
          bias_tz, platform::MKLDNNGetDataType<T>(), memory::format_tag::x);
      conv_pd = handler.AcquireConvolutionPrimitiveDescriptor(
          src_md, weights_md, bias_md, dst_md, strides, paddings, mkldnn_engine,
          fuse_relu, fuse_residual_conn, fuse_brelu, fuse_brelu_threshold,
          fwd_prop_kind);
    } else {
      conv_pd = handler.AcquireConvolutionPrimitiveDescriptor(
          src_md, weights_md, boost::none, dst_md, strides, paddings,
          mkldnn_engine, fuse_relu, fuse_residual_conn, fuse_brelu,
          fuse_brelu_threshold, fwd_prop_kind);
    }

    std::shared_ptr<mkldnn::convolution_forward> conv_p;
    conv_p = handler.AcquireConvolution();

    // create mkldnn memory from input tensors (data/weights)
    auto user_src_memory_p =
        handler.AcquireSrcMemory(user_src_md, to_void_cast<T>(input_data));
    auto user_weights_memory_p = handler.AcquireWeightsMemory(
        user_weights_md, to_void_cast<T>(filter_data));

    // create reorder primitive if the input format is not the preferred one
    auto src_memory_p =
        handler.AcquireSrcMemoryFromPrimitive(user_src_memory_p, pipeline);
    auto weights_memory_p = handler.AcquireWeightsMemoryFromPrimitive(
        user_weights_memory_p, pipeline, is_test);

    std::shared_ptr<mkldnn::memory> dst_memory_p, user_residual_memory_p;

    if (fuse_residual_conn) {
      auto residual_param = ctx.Input<Tensor>("ResidualData");
      auto residual_param_data = residual_param->data<T>();

      PADDLE_ENFORCE(
          residual_param_data != nullptr,
          "Provide data if you want MKLDNN conv+elementwise_add fusion");
      PADDLE_ENFORCE_EQ(output->dims(), residual_param->dims(),
                        "Output and elementwise parameter need to have the "
                        "same dimension sizes");

      if (residual_param->format() != handler.GetDstFormat()) {
        auto output_data =
            output->mutable_data<T>(ctx.GetPlace(), handler.GetDstMemorySize());

        // TODO(grygielski)
        std::vector<int64_t> residual_data_tz =
            paddle::framework::vectorize(residual_param->dims());

        auto residual_data_type =
            paddle::framework::ToMKLDNNDataType(residual_param->type());

        auto user_residual_md = platform::MKLDNNMemDesc(
            residual_data_tz, residual_data_type, residual_param->format());
        user_residual_memory_p = handler.AcquireResidualDataMemory(
            user_residual_md, to_void_cast<T>(residual_param_data));

        dst_memory_p = handler.AcquireDstMemoryFromResidualDataMemory(
            user_residual_memory_p, to_void_cast<T>(output_data), pipeline);
      } else {
        output->ShareDataWith(*residual_param);
        auto output_data = output->mutable_data<T>(ctx.GetPlace());
        dst_memory_p =
            handler.AcquireDstMemoryFromPrimitive(to_void_cast<T>(output_data));
      }
    } else {
      auto output_data =
          output->mutable_data<T>(ctx.GetPlace(), handler.GetDstMemorySize());
      dst_memory_p =
          handler.AcquireDstMemoryFromPrimitive(to_void_cast<T>(output_data));
    }
    // create convolution op primitive
    std::shared_ptr<mkldnn::memory> user_bias_memory_p, bias_memory_p;
    if (bias) {
      const T* bias_data = bias->data<T>();
      auto user_bias_md = platform::MKLDNNMemDesc(
          {bias_tz}, platform::MKLDNNGetDataType<T>(), memory::format_tag::x);
      user_bias_memory_p =
          handler.AcquireBiasMemory(user_bias_md, to_void_cast<T>(bias_data));
      // auto user_bias_memory_p =
      // std::make_shared<mkldnn::memory>(user_bias_md, mkldnn_engine,
      // to_void_cast<T>(bias_data));
      bias_memory_p =
          handler.AcquireBiasMemoryFromPrimitive(user_bias_memory_p, pipeline);
      // conv_p = handler.AcquireConvolution();
    } else {
      // TODO(grygielski) redundant stuff
      // conv_p = handler.AcquireConvolution();
    }

    // TODO(grygielski)
    mkldnn::stream astream(mkldnn_engine);
    if (bias) {
      conv_p->execute(astream, {{MKLDNN_ARG_SRC, *src_memory_p},
                                {MKLDNN_ARG_WEIGHTS, *weights_memory_p},
                                {MKLDNN_ARG_BIAS, *bias_memory_p},
                                {MKLDNN_ARG_DST, *dst_memory_p}});
    } else {
      conv_p->execute(astream, {{MKLDNN_ARG_SRC, *src_memory_p},
                                {MKLDNN_ARG_WEIGHTS, *weights_memory_p},
                                {MKLDNN_ARG_DST, *dst_memory_p}});
    }
    astream.wait();

    output->set_layout(DataLayout::kMKLDNN);
    output->set_format(GetMKLDNNFormat(*dst_memory_p));
  }
  template <typename T_out>
  void ComputeINT8(const paddle::framework::ExecutionContext& ctx) const {
    const bool is_test = ctx.Attr<bool>("is_test");

    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto* input = ctx.Input<Tensor>("Input");
    auto* filter = ctx.Input<Tensor>("Filter");
    auto* bias = ctx.HasInput("Bias") ? ctx.Input<Tensor>("Bias") : nullptr;
    auto* output = ctx.Output<Tensor>("Output");

    PADDLE_ENFORCE(input->layout() == DataLayout::kMKLDNN &&
                       input->format() != memory::format_tag::undef,
                   "Wrong layout/format set for Input tensor");
    PADDLE_ENFORCE(filter->layout() == DataLayout::kMKLDNN &&
                       filter->format() != memory::format_tag::undef,
                   "Wrong layout/format set for Filter tensor");
    PADDLE_ENFORCE(input->dims().size() == 4 || input->dims().size() == 5,
                   "Input must be with 4 or 5 dimensions, i.e. NCHW or NCDHW");
    PADDLE_ENFORCE(filter->dims().size() == 4 || filter->dims().size() == 5,
                   "Filter must be with 4 or 5 dimensions, i.e. OIHW or OIDHW");
    if (bias) {
      PADDLE_ENFORCE(bias->layout() == DataLayout::kMKLDNN &&
                         bias->format() != memory::format_tag::undef,
                     "Wrong layout/format set for Bias tensor");
      PADDLE_ENFORCE(bias->dims().size() == 1,
                     "Bias must only have 1 dimension, i.e. X");
    }

    // TODO(grygielski) change
    std::vector<int> strides_temp = ctx.Attr<std::vector<int>>("strides");
    std::vector<int64_t> strides(begin(strides_temp), end(strides_temp));

    // TODO(grygielski) change
    std::vector<int> paddings_temp = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int64_t> paddings(begin(paddings_temp), end(paddings_temp));

    // TODO(grygielski) change
    std::vector<int> dilations_temp = ctx.Attr<std::vector<int>>("dilations");
    std::vector<int64_t> dilations(begin(dilations_temp), end(dilations_temp));

    int groups = ctx.Attr<int>("groups");
    bool fuse_relu = ctx.Attr<bool>("fuse_relu");
    bool fuse_residual_conn = ctx.Attr<bool>("fuse_residual_connection");
    bool fuse_brelu = ctx.Attr<bool>("fuse_brelu");
    float fuse_brelu_threshold = ctx.Attr<float>("fuse_brelu_threshold");
    bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");
    bool unsigned_output = fuse_relu || fuse_brelu;

    PADDLE_ENFORCE(!fuse_residual_conn || !force_fp32_output,
                   "residual fusion does not support force output with fp32");

    bool is_conv3d = strides.size() == 3U;
    // TODO(tpatejko): add support for dilation
    PADDLE_ENFORCE(
        is_conv3d
            ? dilations.size() == 3 && dilations[0] == 1 && dilations[1] == 1 &&
                  dilations[2] == 1
            : dilations.size() == 2 && dilations[0] == 1 && dilations[1] == 1,
        "dilation in convolution is not implemented yet");

    PADDLE_ENFORCE(is_conv3d != true, "int8 does not support conv3d currently");

    const T* input_data = input->data<T>();

    std::vector<int64_t> src_tz = paddle::framework::vectorize(input->dims());
    std::vector<int64_t> weights_tz =
        paddle::framework::vectorize(filter->dims());
    int g = std::max(groups, 1);

    GetWeightsTz(weights_tz, g, is_conv3d);
    std::vector<int64_t> dst_tz = paddle::framework::vectorize(output->dims());

    mkldnn::memory::data_type src_dt =
        paddle::framework::ToMKLDNNDataType(input->type());

    // Get unique name for storing MKLDNN primitives
    std::string key;
    key.reserve(MaxKeyLength);
    platform::ConvMKLDNNHandler::AppendKey(
        &key, src_tz, weights_tz, strides, paddings, dilations, groups, src_dt,
        input->format(), fuse_relu, fuse_residual_conn, fuse_brelu,
        ctx.op().Input("Input") + ctx.op().Input("Filter"));

    const std::string key_conv_pd = key + "@conv_pd";

    bool need_s8_to_u8 = false;
    std::shared_ptr<mkldnn::convolution_forward> conv_p;
    std::shared_ptr<mkldnn::memory> src_memory_p;
    std::shared_ptr<mkldnn::memory> user_src_memory_p;
    std::shared_ptr<mkldnn::memory> dst_memory_p;
    std::vector<primitive> pipeline;
    std::shared_ptr<mkldnn::convolution_forward::primitive_desc> conv_pd;
    std::shared_ptr<platform::ConvMKLDNNHandler> handler;

    // This is workaround for hacky implementation
    // of conv int8 mkl-dnn. Once conv fp32 and conv int8
    // are merged/unified, this will disappear
    std::string key_tid = "";
    if (platform::get_cur_mkldnn_session_id() ==
        platform::kMKLDNNSessionID_Default) {
      key_tid = "-t:" + platform::MKLDNNHandler::ThreadIDasStr();
    }

    auto prim_key = key + key_tid + "@conv_p";
    auto dst_key = key + key_tid + "@dst_mem_p";
    auto src_key = key + key_tid + "@src_mem_p";
    auto user_src_key = key + key_tid + "@user_src_mem_p";
    auto src_reorder_key = key + key_tid + "@src_mem_preorder_p";
    auto residual_reorder_key = key + key_tid + "@residual_data_mem_preorder_p";

    // TODO(grygielski) can't work like that anymore
    // conv_p = std::static_pointer_cast<mkldnn::convolution_forward>(
    //     dev_ctx.GetBlob(prim_key));

    // TODO(grygielski) MISSING MEMORY POINTERS IN ELSE
    // if (conv_p == nullptr || !is_test) {
    const K* filter_data = filter->data<K>();
    auto scale_in_data = ctx.Attr<float>("Scale_in");
    auto scale_in_eltwise_data = ctx.Attr<float>("Scale_in_eltwise");
    auto scale_weights_data = ctx.Attr<std::vector<float>>("Scale_weights");
    auto scale_out_data =
        force_fp32_output ? 1.0f : ctx.Attr<float>("Scale_out");
    float sum_scale =
        fuse_residual_conn ? scale_out_data / scale_in_eltwise_data : 1.0f;

    bool is_multi_channel = scale_weights_data.size() > 1;

    int count = is_multi_channel ? (g > 1 ? (weights_tz)[1] * (weights_tz)[0]
                                          : (weights_tz)[0])
                                 : 1;
    std::vector<float> output_shift_scale(count);
#pragma omp parallel for if (count > 1)
    for (int i = 0; i < count; i++) {
      if (scale_weights_data[i] == 0.0)
        output_shift_scale[i] = scale_out_data;  // weights data will contain 0
                                                 // in some models, then weights
                                                 // scale couldn't be calculated
      else
        output_shift_scale[i] =
            static_cast<float>(static_cast<double>(scale_out_data) /
                               (static_cast<double>(scale_in_data) *
                                static_cast<double>(scale_weights_data[i])));
    }

    auto user_src_md =
        platform::MKLDNNMemDesc({src_tz}, src_dt, input->format());
    auto user_weights_md =
        platform::MKLDNNMemDesc({weights_tz}, platform::MKLDNNGetDataType<K>(),
                                ((g) == 1) ? mkldnn::memory::format_tag::oihw
                                           : mkldnn::memory::format_tag::goihw);

    /* create memory descriptor for convolution without specified format
    * ('any') which lets a primitive (convolution in this case) choose
    * the memory format preferred for best performance
    */
    std::string data_format = ctx.Attr<std::string>("data_format");
    auto chosen_memory_format =
        platform::data_format_to_memory_format(data_format);

    std::vector<int64_t> bias_tz;

    auto src_md = platform::MKLDNNMemDesc(src_tz, src_dt, chosen_memory_format);
    auto weights_md = platform::MKLDNNMemDesc(weights_tz, memory::data_type::s8,
                                              chosen_memory_format);
    auto dst_md = platform::MKLDNNMemDesc(
        dst_tz, platform::MKLDNNGetDataType<T_out>(), chosen_memory_format);

    handler.reset(new platform::ConvMKLDNNHandler(dev_ctx, mkldnn_engine, key));
    // create a conv primitive descriptor and save it for usage in backward
    // TODO(lidanqing): We use relu post-op instead of brelu post-op cause
    // mkldnn v0.18 does not support INT8 brelu post-op. Use code in /**/ when
    // v0.20 is enabled
    auto propagation = is_test ? mkldnn::prop_kind::forward_scoring
                               : mkldnn::prop_kind::forward_training;

    if (bias) {
      bias_tz = paddle::framework::vectorize(bias->dims());
      auto bias_md = platform::MKLDNNMemDesc(bias_tz, memory::data_type::s32,
                                             mkldnn::memory::format_tag::x);
      conv_pd = handler->AcquireConvolutionPrimitiveDescriptor(
          src_md, weights_md, bias_md, dst_md, strides, paddings, mkldnn_engine,
          fuse_relu || fuse_brelu /*fuse_relu*/, fuse_residual_conn,
          false /*fuse_brelu*/, fuse_brelu_threshold, propagation,
          output_shift_scale, sum_scale);
    } else {
      conv_pd = handler->AcquireConvolutionPrimitiveDescriptor(
          src_md, weights_md, boost::none, dst_md, strides, paddings,
          mkldnn_engine, fuse_relu || fuse_brelu /*fuse_relu*/,
          fuse_residual_conn, false /*fuse_brelu*/, fuse_brelu_threshold,
          propagation, output_shift_scale, sum_scale);
    }

    // create mkldnn memory from input tensors (data/weights)
    user_src_memory_p =
        handler->AcquireSrcMemory(user_src_md, to_void_cast<T>(input_data));
    auto user_weights_memory_p = handler->AcquireWeightsMemory(
        user_weights_md, to_void_cast<K>(filter_data));

    // create reorder primitive if the input format is not the preferred one
    src_memory_p =
        handler->AcquireSrcMemoryFromPrimitive(user_src_memory_p, pipeline);

    std::shared_ptr<mkldnn::memory> weights_memory_p;
    int mask_reorder =
        is_multi_channel ? ((g != 1) ? (1 << 1) + (1 << 0) : 1 << 0) : 0;
    weights_memory_p = handler->AcquireWeightsMemoryFromPrimitive(
        user_weights_memory_p, pipeline, is_test, true, scale_weights_data,
        mask_reorder);

    // TODO(grygielski) test one
    dst_memory_p =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx.GetBlob(dst_key));

    if (fuse_residual_conn) {
      auto residual_param = ctx.Input<Tensor>("ResidualData");
      PADDLE_ENFORCE_EQ(output->dims(), residual_param->dims(),
                        "Output and elementwise parameter need to have the "
                        "same dimension sizes");
      auto residual_dt =
          paddle::framework::ToMKLDNNDataType(residual_param->type());
      if (residual_param->format() != handler->GetDstFormat()) {
        auto residual_data_tz =
            paddle::framework::vectorize(residual_param->dims());
        auto user_residual_md = platform::MKLDNNMemDesc(
            residual_data_tz, residual_dt, residual_param->format());
        dst_memory_p = platform::SetDstMemory<T_out>(
            ctx, output, residual_param, user_residual_md, handler, &pipeline);
      } else {
        output->ShareDataWith(*residual_param);
        dst_memory_p = platform::SetDstMemory<T_out>(ctx, output, handler);
      }
      need_s8_to_u8 =
          (platform::MKLDNNGetDataType<T_out>() == memory::data_type::s8) &&
          unsigned_output;
    } else {
      dst_memory_p = platform::SetDstMemory<T_out>(ctx, output, handler);
    }

    // create convolution op primitive
    auto scale_bias_key = key + "@scale_bias";
    std::shared_ptr<mkldnn::memory> bias_memory_p;
    if (bias) {
      const K* bias_data = bias->data<K>();
      auto user_bias_md = platform::MKLDNNMemDesc(
          {bias_tz}, platform::MKLDNNGetDataType<K>(), memory::format_tag::x);
      auto user_bias_memory_p =
          handler->AcquireBiasMemory(user_bias_md, to_void_cast<K>(bias_data));
      int mask_reorder = is_multi_channel ? 1 << 0 : 1;
      int count = is_multi_channel ? (g > 1 ? (weights_tz)[1] * (weights_tz)[0]
                                            : (weights_tz)[0])
                                   : 1;
      std::vector<float> scale_bias_data(count);
#pragma omp parallel for if (count > 1)
      for (int i = 0; i < count; i++) {
        scale_bias_data[i] = scale_in_data * scale_weights_data[i];
      }
      bias_memory_p = handler->AcquireBiasMemoryFromPrimitive(
          user_bias_memory_p, pipeline, is_test, true, scale_bias_data,
          mask_reorder);
      conv_p = handler->AcquireConvolution();
    } else {
      conv_p = handler->AcquireConvolution();
    }
    // push primitive to stream and wait until it's executed

    // pipeline.push_back(*conv_p);
    // TODO(grygielski) MISSING MEMORY POINTERS IN ELSE
    // } else {
    //   auto src_memory_reorder_p = std::static_pointer_cast<mkldnn::memory>(
    //       dev_ctx.GetBlob(src_reorder_key));
    //   src_memory_p =
    //       std::static_pointer_cast<mkldnn::memory>(dev_ctx.GetBlob(src_key));
    //   if (src_memory_reorder_p) {
    //     user_src_memory_p = std::static_pointer_cast<mkldnn::memory>(
    //         dev_ctx.GetBlob(user_src_key));
    //     user_src_memory_p->set_data_handle(to_void_cast<T>(input_data));
    //   } else if (src_memory_p) {
    //     src_memory_p->set_data_handle(to_void_cast<T>(input_data));
    //   }

    //   dst_memory_p =
    //       std::static_pointer_cast<mkldnn::memory>(dev_ctx.GetBlob(dst_key));
    //   conv_pd =
    //       std::static_pointer_cast<mkldnn::convolution_forward::primitive_desc>(
    //           dev_ctx.GetBlob(key_conv_pd));
    //   if (conv_pd) {
    //     handler.reset(new platform::ConvMKLDNNHandler(conv_pd, dev_ctx,
    //                                                   mkldnn_engine, key));
    //   }

    //   if (fuse_residual_conn) {
    //     auto residual_param = ctx.Input<Tensor>("ResidualData");
    //     output->ShareDataWith(*residual_param);
    //     need_s8_to_u8 =
    //         (platform::MKLDNNGetDataType<T_out>() == memory::data_type::s8)
    //         &&
    //         unsigned_output;
    //   }
    //   platform::SetDstMemoryHandler<T_out>(ctx, output, handler,
    //   dst_memory_p);

    //   // TODO(grygielski) memory is no longer privitive so can't be pushed
    //   into mkldnn::primitive vector
    //   if (src_memory_reorder_p) {
    //     //pipeline.push_back(*src_memory_reorder_p);
    //   }

    //   auto residual_reorder_p = std::static_pointer_cast<mkldnn::memory>(
    //       dev_ctx.GetBlob(residual_reorder_key));
    // // TODO(grygielski) memory is no longer privitive so can't be pushed into
    // mkldnn::primitive vector
    //   if (residual_reorder_p) {
    //     //pipeline.push_back(*residual_reorder_p);
    //   }
    //   pipeline.push_back(*conv_p);
    // }

    // TODO(grygielski)
    mkldnn::stream astream(mkldnn_engine);
    if (bias) {
      conv_p->execute(astream, {{MKLDNN_ARG_SRC, *src_memory_p},
                                {MKLDNN_ARG_WEIGHTS, *weights_memory_p},
                                {MKLDNN_ARG_BIAS, *bias_memory_p},
                                {MKLDNN_ARG_DST, *dst_memory_p}});
    } else {
      conv_p->execute(astream, {{MKLDNN_ARG_SRC, *src_memory_p},
                                {MKLDNN_ARG_WEIGHTS, *weights_memory_p},
                                {MKLDNN_ARG_DST, *dst_memory_p}});
    }
    astream.wait();
    // TODO(grygielski) INT8 conv temporarily disabled
    // stream(stream::kind::eager).submit(pipeline).wait();
    if (need_s8_to_u8) {
      output->mutable_data<uint8_t>(ctx.GetPlace());
    }
    output->set_layout(DataLayout::kMKLDNN);
    output->set_format(GetMKLDNNFormat(*dst_memory_p));
  }
};

template <typename T>
class ConvMKLDNNGradOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    const Tensor* input = ctx.Input<Tensor>("Input");
    const Tensor* filter = ctx.Input<Tensor>("Filter");
    const Tensor* output_grad =
        ctx.Input<Tensor>(framework::GradVarName("Output"));
    Tensor* input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    Tensor* filter_grad = ctx.Output<Tensor>(framework::GradVarName("Filter"));

    PADDLE_ENFORCE(input->layout() == DataLayout::kMKLDNN &&
                       input->format() != memory::format_tag::undef,
                   "Wrong layout/format set for Input tensor");
    PADDLE_ENFORCE(filter->layout() == DataLayout::kMKLDNN &&
                       filter->format() != memory::format_tag::undef,
                   "Wrong layout/format set for Filter tensor");
    PADDLE_ENFORCE(output_grad->layout() == DataLayout::kMKLDNN &&
                       output_grad->format() != memory::format_tag::undef,
                   "Wrong layout/format set for output_grad tensor");

    PADDLE_ENFORCE(
        !ctx.Attr<bool>("is_test"),
        "is_test attribute should be set to False in training phase.");

    if (!input_grad && !filter_grad) return;

    // TODO(grygielski) change
    std::vector<int> strides_temp = ctx.Attr<std::vector<int>>("strides");
    std::vector<int64_t> strides(begin(strides_temp), end(strides_temp));

    // TODO(grygielski) change
    std::vector<int> paddings_temp = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int64_t> paddings(begin(paddings_temp), end(paddings_temp));

    // TODO(grygielski) change
    std::vector<int> dilations_temp = ctx.Attr<std::vector<int>>("dilations");
    std::vector<int64_t> dilations(begin(dilations_temp), end(dilations_temp));

    int groups = ctx.Attr<int>("groups");

    bool is_conv3d = strides.size() == 3U;
    const T* input_data = input->data<T>();
    const T* filter_data = filter->data<T>();
    const T* output_grad_data = output_grad->data<T>();
    T* input_grad_data = nullptr;
    T* filter_grad_data = nullptr;

    std::vector<int64_t> src_tz = paddle::framework::vectorize(input->dims());
    std::vector<int64_t> weights_tz =
        paddle::framework::vectorize(filter->dims());
    int g = std::max(groups, 1);
    GetWeightsTz(weights_tz, g, is_conv3d);
    std::vector<int64_t> dst_tz =
        paddle::framework::vectorize(output_grad->dims());
    bool fuse_relu = ctx.Attr<bool>("fuse_relu");
    bool fuse_brelu = false;
    if (!is_conv3d) {
      fuse_brelu = ctx.Attr<bool>("fuse_brelu");
    }
    auto src_format = input->format();
    mkldnn::memory::format_tag weights_format =
        GetWeightsFormat(filter->format(), g, is_conv3d);

    // Get an unique name from "argument" name of "input" and "Filter" variable
    // as well as attributes of primitive to be created
    // This name will be used as key when saving info into device context
    const std::string key = platform::ConvMKLDNNHandler::GetHash(
        src_tz, weights_tz, fuse_relu, fuse_brelu, strides, paddings, dilations,
        groups, ctx.op().Input("Input") + ctx.op().Input("Filter"));

    const std::string key_conv_pd = key + "@conv_pd";
    std::vector<primitive> pipeline;

    // Create user memory descriptors
    auto user_src_md = platform::MKLDNNMemDesc(
        {src_tz}, platform::MKLDNNGetDataType<T>(), src_format);
    auto user_weights_md = platform::MKLDNNMemDesc(
        {weights_tz}, platform::MKLDNNGetDataType<T>(), weights_format);
    auto user_diff_dst_md = platform::MKLDNNMemDesc(
        {dst_tz}, platform::MKLDNNGetDataType<T>(), output_grad->format());

    /* create memory descriptor for conv backward without specified format
     * ('any') which lets a primitive (conv backward in this case) choose
     * the memory format preferred for best performance
     */
    std::string data_format = ctx.Attr<std::string>("data_format");
    auto chosen_memory_format =
        platform::data_format_to_memory_format(data_format);

    weights_format = mkldnn::memory::format_tag::any;
    // Check the format for user's special output
    if (chosen_memory_format != mkldnn::memory::format_tag::any) {
      if (is_conv3d) {
        chosen_memory_format =
            platform::MKLDNNFormatForSize(src_tz.size(), chosen_memory_format);
      }
    }

    auto src_md = platform::MKLDNNMemDesc(
        src_tz, platform::MKLDNNGetDataType<T>(), chosen_memory_format);
    auto diff_src_md = platform::MKLDNNMemDesc(
        src_tz, platform::MKLDNNGetDataType<T>(), chosen_memory_format);
    auto weights_md = platform::MKLDNNMemDesc(
        weights_tz, platform::MKLDNNGetDataType<T>(), weights_format);
    auto diff_weights_md = platform::MKLDNNMemDesc(
        weights_tz, platform::MKLDNNGetDataType<T>(), weights_format);
    auto diff_dst_md = platform::MKLDNNMemDesc(
        dst_tz, platform::MKLDNNGetDataType<T>(), chosen_memory_format);

    // Retrieve conv_pd from device context
    auto conv_pd =
        std::static_pointer_cast<mkldnn::convolution_forward::primitive_desc>(
            dev_ctx.GetBlob(key_conv_pd));
    PADDLE_ENFORCE(conv_pd != nullptr,
                   "Fail to find conv_pd in device context");

    // create backward convolution weights primitive descriptor
    auto conv_bwd_weights_desc = mkldnn::convolution_backward_weights::desc(
        mkldnn::algorithm::convolution_direct, src_md, diff_weights_md,
        diff_dst_md, strides, paddings, paddings);
    auto conv_bwd_weights_pd =
        std::make_shared<mkldnn::convolution_backward_weights::primitive_desc>(
            conv_bwd_weights_desc, mkldnn_engine, *conv_pd);

    // create backward convolution data primitive descriptor
    auto conv_bwd_data_desc = mkldnn::convolution_backward_data::desc(
        mkldnn::algorithm::convolution_direct, diff_src_md, weights_md,
        diff_dst_md, strides, paddings, paddings);
    auto conv_bwd_data_pd =
        std::make_shared<mkldnn::convolution_backward_data::primitive_desc>(
            conv_bwd_data_desc, mkldnn_engine, *conv_pd);

    platform::ConvMKLDNNHandler handler(conv_pd, conv_bwd_data_pd,
                                        conv_bwd_weights_pd, dev_ctx,
                                        mkldnn_engine, key);

    // create mkldnn memory from input tensors (data/weights)
    auto user_src_memory_p =
        handler.AcquireSrcMemory(user_src_md, to_void_cast<T>(input_data));
    auto user_weights_memory_p = handler.AcquireWeightsMemory(
        user_weights_md, to_void_cast<T>(filter_data));
    auto user_diff_dst_memory_p = handler.AcquireDiffDstMemory(
        user_diff_dst_md, to_void_cast<T>(output_grad_data));

    mkldnn::stream astream(mkldnn_engine);
    // create backward conv primitive for weights
    if (filter_grad) {
      auto src_memory_p = handler.AcquireSrcMemoryFromWeightsPrimitive(
          user_src_memory_p, pipeline);

      auto diff_dst_memory_4filter_p =
          handler.AcquireDiffDstMemoryFromWeightsPrimitive(
              user_diff_dst_memory_p, pipeline);

      const size_t size = handler.GetDiffWeightsMemorySize();
      filter_grad_data = filter_grad->mutable_data<T>(ctx.GetPlace(), size);

      auto diff_weights_memory_p =
          handler.AcquireDiffWeightsMemoryFromWeightsPrimitive(
              reinterpret_cast<void*>(filter_grad_data));

      auto conv_bwd_weights_p = handler.AcquireConvolutionBackwardWeights();

      // push primitive to stream and wait until it's executed
      // pipeline.push_back(*conv_bwd_weights_p);

      // TODO(grygielski) DOESNT SUPPORT REORDER
      // why no bias_diff?
      conv_bwd_weights_p->execute(
          astream, {{MKLDNN_ARG_SRC, *src_memory_p},
                    {MKLDNN_ARG_DIFF_DST, *diff_dst_memory_4filter_p},
                    {MKLDNN_ARG_DIFF_WEIGHTS, *diff_weights_memory_p}});
      astream.wait();

      filter_grad->set_layout(DataLayout::kMKLDNN);
      filter_grad->set_format(GetMKLDNNFormat(*diff_weights_memory_p));
    }

    if (input_grad) {
      auto weights_memory_p = handler.AcquireWeightsMemoryFromDataPrimitive(
          user_weights_memory_p, pipeline);

      auto diff_dst_memory_4data_p =
          handler.AcquireDiffDstMemoryFromDataPrimitive(user_diff_dst_memory_p,
                                                        pipeline);

      const size_t size = handler.GetDiffSourceMemorySize();
      input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace(), size);

      auto diff_src_memory_p = handler.AcquireDiffSrcMemoryFromDataPrimitive(
          reinterpret_cast<void*>(input_grad_data));

      auto conv_bwd_data_p = handler.AcquireConvolutionBackwardData();

      conv_bwd_data_p->execute(astream,
                               {{MKLDNN_ARG_WEIGHTS, *weights_memory_p},
                                {MKLDNN_ARG_DIFF_DST, *diff_dst_memory_4data_p},
                                {MKLDNN_ARG_DIFF_SRC, *diff_src_memory_p}});
      astream.wait();

      input_grad->set_layout(DataLayout::kMKLDNN);
      input_grad->set_format(GetMKLDNNFormat(*diff_src_memory_p));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv2d, MKLDNN,
                                    ::paddle::platform::CPUPlace, FP32,
                                    ops::kConvMKLDNNFP32,
                                    ops::ConvMKLDNNOpKernel<float, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv2d, MKLDNN,
                                    ::paddle::platform::CPUPlace, U8,
                                    ops::kConvMKLDNNINT8,
                                    ops::ConvMKLDNNOpKernel<uint8_t, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv2d, MKLDNN,
                                    ::paddle::platform::CPUPlace, S8,
                                    ops::kConvMKLDNNINT8,
                                    ops::ConvMKLDNNOpKernel<int8_t, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv2d_grad, MKLDNN,
                                    ::paddle::platform::CPUPlace, FP32,
                                    ops::kConvMKLDNNFP32,
                                    ops::ConvMKLDNNGradOpKernel<float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv3d, MKLDNN,
                                    ::paddle::platform::CPUPlace, FP32,
                                    ops::kConvMKLDNNFP32,
                                    ops::ConvMKLDNNOpKernel<float, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv3d_grad, MKLDNN,
                                    ::paddle::platform::CPUPlace, FP32,
                                    ops::kConvMKLDNNFP32,
                                    ops::ConvMKLDNNGradOpKernel<float>);
