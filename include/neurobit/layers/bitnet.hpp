#pragma once

#include <array>
#include <cmath>
#include <random>
#include <sycl/sycl.hpp>
#include <vector>

namespace neurobit
{
namespace layers
{

namespace s = sycl;

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
    s::buffer<int8_t, 1> quant_weight; // {-1, 0, 1}
    s::buffer<int8_t, 1> quant_input;  // 8-bit quantized input

    bool use_bias;
    float eps = 1e-5f;

    BitLinear(size_t in, size_t out, bool bias_enabled = false)
        : in_dim(in), out_dim(out), use_bias(bias_enabled), weight(s::range<1>(in * out)), bias(s::range<1>(out)),
          grad_weight(s::range<1>(in * out)), grad_bias(s::range<1>(out)),
          weight_scale(s::range<1>(1)),                                    // Per-tensor scaling for simplicity first
          input_scale(s::range<1>(1)),                                     // Per-tensor scaling
          quant_weight(s::range<1>(in * out)), quant_input(s::range<1>(1)) // Placeholder, resized in forward
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

    // Forward Pass
    // X: [Batch, In] -> Y: [Batch, Out]
    void forward(s::queue &q, s::buffer<float, 1> &input, s::buffer<float, 1> &output, size_t batch_size)
    {
        // Resize intermediate buffers if needed
        if (quant_input.get_range()[0] != batch_size * in_dim)
        {
            quant_input = s::buffer<int8_t, 1>(s::range<1>(batch_size * in_dim));
            input_scale = s::buffer<float, 1>(s::range<1>(batch_size)); // Per-token scaling
        }

        // Capture members locally to avoid 'this' capture in kernels
        size_t In = in_dim;
        size_t Out = out_dim;
        bool has_bias = use_bias;

        // 1. Quantize Weights (AbsMean)
        // beta = mean(|W|)
        // W_q = Clamp(Round(W / beta), -1, 1)
        {
            // Temporary buffer for reduction result
            std::array<float, 1> sum_abs_init{0.0f};
            s::buffer<float, 1> buf_sum_abs(sum_abs_init.data(), s::range<1>(1));

            // Step 1: Parallel Reduction for sum(|W|)
            q.submit([&](s::handler &h) {
                s::accessor w(weight, h, s::read_only);
                auto sum_reduction = s::reduction(buf_sum_abs, h, s::plus<float>());

                h.parallel_for(s::range<1>(In * Out), sum_reduction,
                               [=](s::id<1> idx, auto &sum) { sum += s::fabs(w[idx]); });
            });

            // Step 2: Compute Beta (Single Task - very fast)
            q.submit([&](s::handler &h) {
                s::accessor sum_abs_acc(buf_sum_abs, h, s::read_only);
                s::accessor w_s(weight_scale, h, s::write_only);

                h.single_task([=]() {
                    float beta = sum_abs_acc[0] / (In * Out) + 1e-6f;
                    w_s[0] = beta;
                });
            });

            // Step 3: Quantize Weights (Parallel)
            q.submit([&](s::handler &h) {
                s::accessor w(weight, h, s::read_only);
                s::accessor w_s(weight_scale, h, s::read_only); // Read computed beta
                s::accessor w_q(quant_weight, h, s::write_only);

                h.parallel_for(s::range<1>(In * Out), [=](s::id<1> idx) {
                    float beta = w_s[0];
                    float scaled = w[idx] / beta;
                    float rounded = s::round(scaled);
                    if (rounded > 1.0f)
                        rounded = 1.0f;
                    if (rounded < -1.0f)
                        rounded = -1.0f;
                    w_q[idx] = static_cast<int8_t>(rounded);
                });
            });
        }

        // 2. Quantize Inputs (AbsMax) - BitNet b1.58 uses 8-bit activations
        // gamma = max(|X|) / 127
        // X_q = Clamp(Round(X / gamma), -127, 127)
        q.submit([&](s::handler &h) {
            s::accessor x(input, h, s::read_only);
            s::accessor x_q(quant_input, h, s::write_only);
            s::accessor x_s(input_scale, h, s::write_only);

            h.parallel_for(s::range<1>(batch_size), [=](s::id<1> idx) {
                size_t b = idx[0];
                float max_abs = 0.0f;
                // Find Max per token
                for (size_t i = 0; i < In; ++i)
                {
                    float val = s::fabs(x[b * In + i]);
                    if (val > max_abs)
                        max_abs = val;
                }
                float gamma = max_abs / 127.0f + 1e-6f;
                x_s[b] = gamma;

                for (size_t i = 0; i < In; ++i)
                {
                    float scaled = x[b * In + i] / gamma;
                    float rounded = s::round(scaled);
                    if (rounded > 127.0f)
                        rounded = 127.0f;
                    if (rounded < -127.0f)
                        rounded = -127.0f;
                    x_q[b * In + i] = static_cast<int8_t>(rounded);
                }
            });
        });

        // 3. Matrix Multiplication (Int8 * Int8 -> FP32 Accum -> Scale)
        // Y = (X_q @ W_q) * gamma * beta
        q.submit([&](s::handler &h) {
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
                {
                    dot += static_cast<float>(x_q[r * In + k]) * static_cast<float>(w_q[k * Out + c]);
                }

                // Dequantize: dot * beta * gamma
                float res = dot * w_s[0] * x_s[r];

                if (has_bias)
                    res += b[c];
                y[r * Out + c] = res;
            });
        });
    }

    // Backward Pass
    // STE (Straight Through Estimator) for Weights
    void backward(s::queue &q, s::buffer<float, 1> &grad_output, s::buffer<float, 1> &grad_input, size_t batch_size)
    {
        size_t In = in_dim;
        size_t Out = out_dim;
        bool has_bias = use_bias;

        // Compute dX
        q.submit([&](s::handler &h) {
            s::accessor dy(grad_output, h, s::read_only);
            s::accessor w_q(quant_weight, h, s::read_only);
            s::accessor w_s(weight_scale, h, s::read_only);
            s::accessor dx(grad_input, h, s::write_only, s::no_init);

            h.parallel_for(s::range<2>(batch_size, In), [=](s::id<2> idx) {
                size_t r = idx[0];
                size_t c = idx[1];
                float sum = 0.0f;
                for (size_t k = 0; k < Out; ++k)
                {
                    sum += dy[r * Out + k] * static_cast<float>(w_q[c * Out + k]); // Transpose W
                }
                // STE: dX ≈ dY * (beta * W_q)  (gamma cancels via dX_q/dX ≈ 1/gamma)
                dx[r * In + c] = sum * w_s[0];
            });
        });

        // Compute dW
        q.submit([&](s::handler &h) {
            s::accessor dy(grad_output, h, s::read_only);
            s::accessor x_q(quant_input, h, s::read_only);
            s::accessor x_s(input_scale, h, s::read_only);
            s::accessor w_s(weight_scale, h, s::read_only);            // Add this accessor
            s::accessor dw(grad_weight, h, s::write_only, s::no_init); // Accumulate? Usually we reset.

            // Naive MM for dW: [In, Batch] @ [Batch, Out]
            h.parallel_for(s::range<2>(In, Out), [=](s::id<2> idx) {
                size_t r = idx[0];
                size_t c = idx[1];
                float sum = 0.0f;
                // Sum over batch
                for (size_t b = 0; b < batch_size; ++b)
                {
                    // dW += X.T * dY
                    // Here we use Quantized X and rescale with input scale
                    float term = static_cast<float>(x_q[b * In + r]) * dy[b * Out + c];
                    term *= x_s[b];
                    sum += term;
                }
                // STE: W_q = round(W / beta) => dW_q/dW ≈ 1/beta, and dY/dW_q includes beta,
                // so beta cancels; effective dW scale is driven by dequantized X (gamma).
                dw[r * Out + c] = sum;
            });
        });

        if (has_bias)
        {
            q.submit([&](s::handler &h) {
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
        }
    }

    // Optimizer Step (SGD for simplicity)
    void step(s::queue &q, float lr, float weight_decay = 0.0f)
    {
        size_t total_params = in_dim * out_dim;
        size_t Out = out_dim;
        bool has_bias = use_bias;

        q.submit([&](s::handler &h) {
            s::accessor w(weight, h, s::read_write);
            s::accessor dw(grad_weight, h, s::read_only);

            h.parallel_for(s::range<1>(total_params), [=](s::id<1> idx) {
                size_t i = idx[0];
                float grad = dw[i];
                float val = w[i];

                // Weight Decay (L1 Regularization for Sparsity)
                if (weight_decay > 0.0f)
                {
                    float s = (val > 0.0f) ? 1.0f : ((val < 0.0f) ? -1.0f : 0.0f);
                    val -= weight_decay * lr * s;
                }

                // Update
                val -= lr * grad;

                w[i] = val;
            });
        });

        if (has_bias)
        {
            q.submit([&](s::handler &h) {
                s::accessor b_acc(bias, h, s::read_write);
                s::accessor db_acc(grad_bias, h, s::read_only);
                h.parallel_for(s::range<1>(Out), [=](s::id<1> idx) { b_acc[idx[0]] -= lr * db_acc[idx[0]]; });
            });
        }
    }
};

} // namespace layers
} // namespace neurobit
