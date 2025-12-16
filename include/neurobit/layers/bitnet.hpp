#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/sycl.hpp>
#include <vector>

namespace neurobit
{
namespace layers
{

namespace s = sycl;
using bfloat16 = s::ext::oneapi::bfloat16;
namespace matrix = s::ext::oneapi::experimental::matrix;

// BitNet b1.58 Layer Implementation
// Reference: "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
class BitLinear
{
  public:
    size_t in_dim;
    size_t out_dim;

    // Parameters
    s::buffer<float, 1> weight; // Shadow weights (FP32)
    s::buffer<float, 1> bias;   // Bias (FP32), optional

    // Gradients
    s::buffer<float, 1> grad_weight;
    s::buffer<float, 1> grad_bias;

    // Quantization Scales (Stored for Backward)
    s::buffer<float, 1> weight_scale; // beta [1] or [out_dim]
    s::buffer<float, 1> input_scale;  // gamma [1] or [batch]

    // Intermediate buffers for backward
    s::buffer<int8_t, 1> quant_weight;   // {-1, 0, 1}
    s::buffer<int8_t, 1> quant_weight_t; // transposed [out_dim, in_dim] for dX
    s::buffer<int8_t, 1> quant_input;    // 8-bit quantized input
    s::buffer<int8_t, 1> quant_input_t;  // transposed [in_dim, batch] for dW
    s::buffer<float, 1> buf_sum_abs;     // persistent reduction buffer (avoid per-call sync)

    bool use_bias;
    float eps = 1e-5f;
    bool weights_quantized = false;
    size_t cached_batch_size = 0;
    bool xmx_checked = false;
    bool xmx_supported = false;

    BitLinear(size_t in, size_t out, bool bias_enabled = false)
        : in_dim(in), out_dim(out), use_bias(bias_enabled), weight(s::range<1>(in * out)), bias(s::range<1>(out)),
          grad_weight(s::range<1>(in * out)), grad_bias(s::range<1>(out)),
          weight_scale(s::range<1>(1)), // Per-tensor scaling for simplicity first
          input_scale(s::range<1>(1)),  // Per-tensor scaling
          quant_weight(s::range<1>(in * out)), quant_weight_t(s::range<1>(in * out)), quant_input(s::range<1>(1)),
          quant_input_t(s::range<1>(1)), buf_sum_abs(s::range<1>(1))
    {
        // Initialize weights (Kaiming / Xavier)
        std::mt19937 gen(1234);
        float std_dev = std::sqrt(2.0f / in);
        std::normal_distribution<float> d(0.0f, std_dev);

        s::host_accessor w_acc(weight, s::write_only);
        for (size_t i = 0; i < in * out; ++i)
            w_acc[i] = d(gen);

        if (use_bias)
        {
            s::host_accessor b_acc(bias, s::write_only);
            for (size_t i = 0; i < out; ++i)
                b_acc[i] = 0.0f;
        }
    }

    void quantize_weights(s::queue &q, std::vector<s::event> *profile_events = nullptr)
    {
        size_t In = in_dim;
        size_t Out = out_dim;

        auto ev_red = q.submit([&](s::handler &h) {
            s::accessor w(weight, h, s::read_only);
            auto sum_reduction =
                s::reduction(buf_sum_abs, h, s::plus<float>(), {s::property::reduction::initialize_to_identity{}});
            h.parallel_for(s::range<1>(In * Out), sum_reduction,
                           [=](s::id<1> idx, auto &sum) { sum += s::fabs(w[idx]); });
        });
        if (profile_events)
            profile_events->push_back(ev_red);

        auto ev_beta = q.submit([&](s::handler &h) {
            s::accessor sum_abs_acc(buf_sum_abs, h, s::read_only);
            s::accessor w_s(weight_scale, h, s::write_only);
            h.single_task([=]() {
                float beta = sum_abs_acc[0] / (In * Out) + 1e-6f;
                w_s[0] = beta;
            });
        });
        if (profile_events)
            profile_events->push_back(ev_beta);

        auto ev_qw = q.submit([&](s::handler &h) {
            s::accessor w(weight, h, s::read_only);
            s::accessor w_s(weight_scale, h, s::read_only);
            s::accessor w_q(quant_weight, h, s::write_only);
            s::accessor w_qt(quant_weight_t, h, s::write_only);

            h.parallel_for(s::range<1>(In * Out), [=](s::id<1> idx) {
                size_t linear = idx[0];
                size_t r = linear / Out;
                size_t c = linear - r * Out;

                float beta = w_s[0];
                float scaled = w[linear] / beta;
                float rounded = s::round(scaled);
                if (rounded > 1.0f)
                    rounded = 1.0f;
                if (rounded < -1.0f)
                    rounded = -1.0f;
                int8_t qv = static_cast<int8_t>(rounded);
                w_q[linear] = qv;
                w_qt[c * In + r] = qv;
            });
        });
        if (profile_events)
            profile_events->push_back(ev_qw);

        weights_quantized = true;
    }

    // Forward Pass
    // X: [Batch, In] -> Y: [Batch, Out]
    void forward(s::queue &q, s::buffer<float, 1> &input, s::buffer<float, 1> &output, size_t batch_size,
                 std::vector<s::event> *profile_events = nullptr)
    {
        // Resize intermediate buffers if needed
        if (cached_batch_size != batch_size)
        {
            quant_input = s::buffer<int8_t, 1>(s::range<1>(batch_size * in_dim));
            quant_input_t = s::buffer<int8_t, 1>(s::range<1>(batch_size * in_dim));
            input_scale = s::buffer<float, 1>(s::range<1>(batch_size)); // Per-token scaling
            cached_batch_size = batch_size;
        }

        // Capture members locally to avoid 'this' capture in kernels
        size_t In = in_dim;
        size_t Out = out_dim;
        bool has_bias = use_bias;
        if (!weights_quantized)
            quantize_weights(q, profile_events);
        if (!xmx_checked)
        {
            auto device = q.get_device();
            auto sg_sizes = device.get_info<s::info::device::sub_group_sizes>();
            bool has_sg16 = std::find(sg_sizes.begin(), sg_sizes.end(), 16) != sg_sizes.end();
            xmx_supported = has_sg16 && device.has(s::aspect::ext_intel_matrix);
            xmx_checked = true;
        }

        // 1. Quantize Inputs (AbsMax) - BitNet b1.58 uses 8-bit activations
        // gamma = max(|X|) / 127
        // X_q = Clamp(Round(X / gamma), -127, 127)
        auto ev3 = q.submit([&](s::handler &h) {
            s::accessor x(input, h, s::read_only);
            s::accessor x_q(quant_input, h, s::write_only);
            s::accessor x_qt(quant_input_t, h, s::write_only);
            s::accessor x_s(input_scale, h, s::write_only);

            constexpr size_t kLocal = 256;
            s::local_accessor<float, 1> local_max{s::range<1>(kLocal), h};

            h.parallel_for(s::nd_range<2>{s::range<2>{batch_size, kLocal}, s::range<2>{1, kLocal}},
                           [=](s::nd_item<2> it) {
                               size_t b = it.get_global_id(0);
                               size_t lid = it.get_local_id(1);

                               float max_abs = 0.0f;
                               for (size_t i = lid; i < In; i += kLocal)
                               {
                                   float val = s::fabs(x[b * In + i]);
                                   if (val > max_abs)
                                       max_abs = val;
                               }

                               local_max[lid] = max_abs;
                               it.barrier(s::access::fence_space::local_space);

                               for (size_t stride = kLocal / 2; stride > 0; stride >>= 1)
                               {
                                   if (lid < stride)
                                   {
                                       float other = local_max[lid + stride];
                                       if (other > local_max[lid])
                                           local_max[lid] = other;
                                   }
                                   it.barrier(s::access::fence_space::local_space);
                               }

                               float gamma = local_max[0] / 127.0f + 1e-6f;
                               if (lid == 0)
                                   x_s[b] = gamma;

                               for (size_t i = lid; i < In; i += kLocal)
                               {
                                   float scaled = x[b * In + i] / gamma;
                                   float rounded = s::round(scaled);
                                   if (rounded > 127.0f)
                                       rounded = 127.0f;
                                   if (rounded < -127.0f)
                                       rounded = -127.0f;
                                   int8_t qv = static_cast<int8_t>(rounded);
                                   x_q[b * In + i] = qv;
                                   x_qt[i * batch_size + b] = qv;
                               }
                           });
        });
        if (profile_events)
            profile_events->push_back(ev3);

        s::event ev4;
        if (xmx_supported)
        {
            // 2. Matrix Multiplication via XMX (bf16 joint_matrix)
            // Y = (X_q @ W_q) * gamma * beta
            ev4 = q.submit([&](s::handler &h) {
                s::accessor x_q(quant_input, h, s::read_only);
                s::accessor w_q(quant_weight, h, s::read_only);
                s::accessor x_s(input_scale, h, s::read_only);
                s::accessor w_s(weight_scale, h, s::read_only);
                s::accessor y(output, h, s::write_only, s::no_init);
                s::accessor b(bias, h, s::read_only);

                constexpr size_t TM = 8;
                constexpr size_t TN = 16;
                constexpr size_t TK = 16;

                size_t num_m_blocks = (batch_size + TM - 1) / TM;
                size_t num_n_blocks = (Out + TN - 1) / TN;

                s::local_accessor<bfloat16, 2> slm_x{s::range<2>{TM, TK}, h};
                s::local_accessor<bfloat16, 2> slm_w{s::range<2>{TK, TN}, h};
                s::local_accessor<float, 2> slm_acc{s::range<2>{TM, TN}, h};

                h.parallel_for(
                    s::nd_range<2>{s::range<2>{num_m_blocks, num_n_blocks * 16}, s::range<2>{1, 16}},
                    [=](s::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                        auto sg = it.get_sub_group();
                        size_t m_base = it.get_group(0) * TM;
                        size_t n_base = it.get_group(1) * TN;
                        size_t lane = sg.get_local_id()[0];

                        matrix::joint_matrix<s::sub_group, float, matrix::use::accumulator, TM, TN> acc;
                        matrix::joint_matrix_fill(sg, acc, 0.0f);

                        for (size_t k_base = 0; k_base < In; k_base += TK)
                        {
                            for (size_t r = 0; r < TM; ++r)
                            {
                                size_t m = m_base + r;
                                size_t k = k_base + lane;
                                if (m < batch_size && k < In)
                                    slm_x[r][lane] = bfloat16(static_cast<float>(x_q[m * In + k]));
                                else
                                    slm_x[r][lane] = bfloat16(0.0f);
                            }

                            for (size_t c = 0; c < TN; ++c)
                            {
                                size_t k = k_base + lane;
                                size_t n = n_base + c;
                                if (k < In && n < Out)
                                    slm_w[lane][c] = bfloat16(static_cast<float>(w_q[k * Out + n]));
                                else
                                    slm_w[lane][c] = bfloat16(0.0f);
                            }

                            it.barrier(s::access::fence_space::local_space);

                            matrix::joint_matrix<s::sub_group, bfloat16, matrix::use::a, TM, TK,
                                                 matrix::layout::row_major>
                                mat_x;
                            matrix::joint_matrix<s::sub_group, bfloat16, matrix::use::b, TK, TN,
                                                 matrix::layout::row_major>
                                mat_w;

                            matrix::joint_matrix_load(sg, mat_x,
                                                      slm_x.template get_multi_ptr<s::access::decorated::no>(), TK);
                            matrix::joint_matrix_load(sg, mat_w,
                                                      slm_w.template get_multi_ptr<s::access::decorated::no>(), TN);
                            matrix::joint_matrix_mad(sg, acc, mat_x, mat_w, acc);

                            it.barrier(s::access::fence_space::local_space);
                        }

                        matrix::joint_matrix_store(sg, acc, slm_acc.template get_multi_ptr<s::access::decorated::no>(),
                                                   TN, matrix::layout::row_major);
                        it.barrier(s::access::fence_space::local_space);

                        float beta = w_s[0];
                        for (size_t idx = lane; idx < TM * TN; idx += 16)
                        {
                            size_t r = idx / TN;
                            size_t c = idx % TN;
                            size_t m = m_base + r;
                            size_t n = n_base + c;
                            if (m < batch_size && n < Out)
                            {
                                float res = slm_acc[r][c] * beta * x_s[m];
                                if (has_bias)
                                    res += b[n];
                                y[m * Out + n] = res;
                            }
                        }
                    });
            });
        }
        else
        {
            // Fallback: scalar dot
            ev4 = q.submit([&](s::handler &h) {
                s::accessor x_q(quant_input, h, s::read_only);
                s::accessor w_q(quant_weight, h, s::read_only);
                s::accessor x_s(input_scale, h, s::read_only);
                s::accessor w_s(weight_scale, h, s::read_only);
                s::accessor y(output, h, s::write_only, s::no_init);
                s::accessor b(bias, h, s::read_only);

                h.parallel_for(s::range<2>(batch_size, Out), [=](s::id<2> idx) {
                    size_t r = idx[0];
                    size_t c = idx[1];
                    float dot = 0.0f;
                    for (size_t k = 0; k < In; ++k)
                        dot += static_cast<float>(x_q[r * In + k]) * static_cast<float>(w_q[k * Out + c]);
                    float res = dot * w_s[0] * x_s[r];
                    if (has_bias)
                        res += b[c];
                    y[r * Out + c] = res;
                });
            });
        }
        if (profile_events)
            profile_events->push_back(ev4);
    }

    // Backward Pass
    // STE (Straight Through Estimator) for Weights
    void backward(s::queue &q, s::buffer<float, 1> &grad_output, s::buffer<float, 1> &grad_input, size_t batch_size,
                  std::vector<s::event> *profile_events = nullptr)
    {
        size_t In = in_dim;
        size_t Out = out_dim;
        bool has_bias = use_bias;
        if (!xmx_checked)
        {
            auto device = q.get_device();
            auto sg_sizes = device.get_info<s::info::device::sub_group_sizes>();
            bool has_sg16 = std::find(sg_sizes.begin(), sg_sizes.end(), 16) != sg_sizes.end();
            xmx_supported = has_sg16 && device.has(s::aspect::ext_intel_matrix);
            xmx_checked = true;
        }

        // Compute dX
        s::event ev0;
        if (xmx_supported)
        {
            ev0 = q.submit([&](s::handler &h) {
                s::accessor dy(grad_output, h, s::read_only);
                s::accessor w_qt(quant_weight_t, h, s::read_only);
                s::accessor w_s(weight_scale, h, s::read_only);
                s::accessor dx(grad_input, h, s::write_only, s::no_init);

                constexpr size_t TM = 8;
                constexpr size_t TN = 16;
                constexpr size_t TK = 16;

                size_t num_m_blocks = (batch_size + TM - 1) / TM;
                size_t num_n_blocks = (In + TN - 1) / TN;

                s::local_accessor<bfloat16, 2> slm_x{s::range<2>{TM, TK}, h};
                s::local_accessor<bfloat16, 2> slm_w{s::range<2>{TK, TN}, h};
                s::local_accessor<float, 2> slm_acc{s::range<2>{TM, TN}, h};

                h.parallel_for(
                    s::nd_range<2>{s::range<2>{num_m_blocks, num_n_blocks * 16}, s::range<2>{1, 16}},
                    [=](s::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                        auto sg = it.get_sub_group();
                        size_t m_base = it.get_group(0) * TM;
                        size_t n_base = it.get_group(1) * TN;
                        size_t lane = sg.get_local_id()[0];

                        matrix::joint_matrix<s::sub_group, float, matrix::use::accumulator, TM, TN> acc;
                        matrix::joint_matrix_fill(sg, acc, 0.0f);

                        for (size_t k_base = 0; k_base < Out; k_base += TK)
                        {
                            for (size_t r = 0; r < TM; ++r)
                            {
                                size_t m = m_base + r;
                                size_t k = k_base + lane;
                                if (m < batch_size && k < Out)
                                    slm_x[r][lane] = bfloat16(dy[m * Out + k]);
                                else
                                    slm_x[r][lane] = bfloat16(0.0f);
                            }

                            for (size_t c = 0; c < TN; ++c)
                            {
                                size_t k = k_base + lane;
                                size_t n = n_base + c;
                                if (k < Out && n < In)
                                    slm_w[lane][c] = bfloat16(static_cast<float>(w_qt[k * In + n]));
                                else
                                    slm_w[lane][c] = bfloat16(0.0f);
                            }

                            it.barrier(s::access::fence_space::local_space);

                            matrix::joint_matrix<s::sub_group, bfloat16, matrix::use::a, TM, TK,
                                                 matrix::layout::row_major>
                                mat_x;
                            matrix::joint_matrix<s::sub_group, bfloat16, matrix::use::b, TK, TN,
                                                 matrix::layout::row_major>
                                mat_w;

                            matrix::joint_matrix_load(sg, mat_x,
                                                      slm_x.template get_multi_ptr<s::access::decorated::no>(), TK);
                            matrix::joint_matrix_load(sg, mat_w,
                                                      slm_w.template get_multi_ptr<s::access::decorated::no>(), TN);
                            matrix::joint_matrix_mad(sg, acc, mat_x, mat_w, acc);

                            it.barrier(s::access::fence_space::local_space);
                        }

                        matrix::joint_matrix_store(sg, acc, slm_acc.template get_multi_ptr<s::access::decorated::no>(),
                                                   TN, matrix::layout::row_major);
                        it.barrier(s::access::fence_space::local_space);

                        float beta = w_s[0];
                        for (size_t idx = lane; idx < TM * TN; idx += 16)
                        {
                            size_t r = idx / TN;
                            size_t c = idx % TN;
                            size_t m = m_base + r;
                            size_t n = n_base + c;
                            if (m < batch_size && n < In)
                            {
                                dx[m * In + n] = slm_acc[r][c] * beta;
                            }
                        }
                    });
            });
        }
        else
        {
            ev0 = q.submit([&](s::handler &h) {
                s::accessor dy(grad_output, h, s::read_only);
                s::accessor w_q(quant_weight, h, s::read_only);
                s::accessor w_s(weight_scale, h, s::read_only);
                s::accessor dx(grad_input, h, s::write_only, s::no_init);

                h.parallel_for(s::range<2>(batch_size, In), [=](s::id<2> idx) {
                    size_t r = idx[0];
                    size_t c = idx[1];
                    float sum = 0.0f;
                    for (size_t k = 0; k < Out; ++k)
                        sum += dy[r * Out + k] * static_cast<float>(w_q[c * Out + k]); // Transpose W
                    dx[r * In + c] = sum * w_s[0];
                });
            });
        }
        if (profile_events)
            profile_events->push_back(ev0);

        // Compute dW
        s::event ev1;
        if (xmx_supported)
        {
            ev1 = q.submit([&](s::handler &h) {
                s::accessor dy(grad_output, h, s::read_only);
                s::accessor x_qt(quant_input_t, h, s::read_only);
                s::accessor x_s(input_scale, h, s::read_only);
                s::accessor dw(grad_weight, h, s::write_only, s::no_init);

                constexpr size_t TM = 8;
                constexpr size_t TN = 16;
                constexpr size_t TK = 16;

                size_t num_m_blocks = (In + TM - 1) / TM;
                size_t num_n_blocks = (Out + TN - 1) / TN;

                s::local_accessor<bfloat16, 2> slm_x{s::range<2>{TM, TK}, h};
                s::local_accessor<bfloat16, 2> slm_w{s::range<2>{TK, TN}, h};
                s::local_accessor<float, 2> slm_acc{s::range<2>{TM, TN}, h};

                h.parallel_for(
                    s::nd_range<2>{s::range<2>{num_m_blocks, num_n_blocks * 16}, s::range<2>{1, 16}},
                    [=](s::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                        auto sg = it.get_sub_group();
                        size_t m_base = it.get_group(0) * TM;
                        size_t n_base = it.get_group(1) * TN;
                        size_t lane = sg.get_local_id()[0];

                        matrix::joint_matrix<s::sub_group, float, matrix::use::accumulator, TM, TN> acc;
                        matrix::joint_matrix_fill(sg, acc, 0.0f);

                        for (size_t k_base = 0; k_base < batch_size; k_base += TK)
                        {
                            for (size_t r = 0; r < TM; ++r)
                            {
                                size_t m = m_base + r;
                                size_t k = k_base + lane;
                                if (m < In && k < batch_size)
                                    slm_x[r][lane] = bfloat16(static_cast<float>(x_qt[m * batch_size + k]));
                                else
                                    slm_x[r][lane] = bfloat16(0.0f);
                            }

                            for (size_t c = 0; c < TN; ++c)
                            {
                                size_t k = k_base + lane;
                                size_t n = n_base + c;
                                if (k < batch_size && n < Out)
                                    slm_w[lane][c] = bfloat16(dy[k * Out + n] * x_s[k]);
                                else
                                    slm_w[lane][c] = bfloat16(0.0f);
                            }

                            it.barrier(s::access::fence_space::local_space);

                            matrix::joint_matrix<s::sub_group, bfloat16, matrix::use::a, TM, TK,
                                                 matrix::layout::row_major>
                                mat_x;
                            matrix::joint_matrix<s::sub_group, bfloat16, matrix::use::b, TK, TN,
                                                 matrix::layout::row_major>
                                mat_w;

                            matrix::joint_matrix_load(sg, mat_x,
                                                      slm_x.template get_multi_ptr<s::access::decorated::no>(), TK);
                            matrix::joint_matrix_load(sg, mat_w,
                                                      slm_w.template get_multi_ptr<s::access::decorated::no>(), TN);
                            matrix::joint_matrix_mad(sg, acc, mat_x, mat_w, acc);

                            it.barrier(s::access::fence_space::local_space);
                        }

                        matrix::joint_matrix_store(sg, acc, slm_acc.template get_multi_ptr<s::access::decorated::no>(),
                                                   TN, matrix::layout::row_major);
                        it.barrier(s::access::fence_space::local_space);

                        for (size_t idx = lane; idx < TM * TN; idx += 16)
                        {
                            size_t r = idx / TN;
                            size_t c = idx % TN;
                            size_t m = m_base + r;
                            size_t n = n_base + c;
                            if (m < In && n < Out)
                                dw[m * Out + n] = slm_acc[r][c];
                        }
                    });
            });
        }
        else
        {
            ev1 = q.submit([&](s::handler &h) {
                s::accessor dy(grad_output, h, s::read_only);
                s::accessor x_q(quant_input, h, s::read_only);
                s::accessor x_s(input_scale, h, s::read_only);
                s::accessor dw(grad_weight, h, s::write_only, s::no_init);

                // Naive MM for dW: [In, Batch] @ [Batch, Out]
                h.parallel_for(s::range<2>(In, Out), [=](s::id<2> idx) {
                    size_t r = idx[0];
                    size_t c = idx[1];
                    float sum = 0.0f;
                    for (size_t b = 0; b < batch_size; ++b)
                    {
                        float term = static_cast<float>(x_q[b * In + r]) * dy[b * Out + c];
                        term *= x_s[b];
                        sum += term;
                    }
                    dw[r * Out + c] = sum;
                });
            });
        }
        if (profile_events)
            profile_events->push_back(ev1);

        if (has_bias)
        {
            auto ev2 = q.submit([&](s::handler &h) {
                s::accessor dy(grad_output, h, s::read_only);
                s::accessor db(grad_bias, h, s::write_only);
                h.parallel_for(s::range<1>(Out), [=](s::id<1> idx) {
                    size_t c = idx[0];
                    float sum = 0.0f;
                    for (size_t b = 0; b < batch_size; ++b)
                        sum += dy[b * Out + c];
                    db[c] = sum;
                });
            });
            if (profile_events)
                profile_events->push_back(ev2);
        }
    }

    // Optimizer Step (SGD for simplicity)
    void step(s::queue &q, float lr, float weight_decay = 0.0f, std::vector<s::event> *profile_events = nullptr)
    {
        size_t total_params = in_dim * out_dim;
        size_t Out = out_dim;
        bool has_bias = use_bias;

        auto ev0 = q.submit([&](s::handler &h) {
            s::accessor w(weight, h, s::read_write);
            s::accessor dw(grad_weight, h, s::read_only);

            h.parallel_for(s::range<1>(total_params), [=](s::id<1> idx) {
                size_t i = idx[0];
                float grad = dw[i];
                float val = w[i];

                // Weight Decay (L2)
                if (weight_decay > 0.0f)
                {
                    val -= weight_decay * lr * val;
                }

                // Update
                val -= lr * grad;

                w[i] = val;
            });
        });
        if (profile_events)
            profile_events->push_back(ev0);

        if (has_bias)
        {
            auto ev1 = q.submit([&](s::handler &h) {
                s::accessor b_acc(bias, h, s::read_write);
                s::accessor db_acc(grad_bias, h, s::read_only);
                h.parallel_for(s::range<1>(Out), [=](s::id<1> idx) { b_acc[idx[0]] -= lr * db_acc[idx[0]]; });
            });
            if (profile_events)
                profile_events->push_back(ev1);
        }

        quantize_weights(q, profile_events);
    }

    float mean_abs_weight_host()
    {
        s::host_accessor acc(weight_scale, s::read_only);
        if (acc.get_range()[0] == 0)
            return 0.0f;
        return acc[0];
    }
};

} // namespace layers
} // namespace neurobit
