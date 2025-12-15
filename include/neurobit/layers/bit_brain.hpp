#pragma once

#include "neurobit/components/glial.hpp"
#include "neurobit/components/hippocampus.hpp"
#include "neurobit/core/types.hpp"
#include "neurobit/layers/activation.hpp" // M-DSiLU
#include "neurobit/layers/bitnet.hpp"     // BitNet b1.58 (Trainable)
#include "neurobit/layers/ssm_scan.hpp"
#include <sycl/sycl.hpp>
#include <vector>

namespace neurobit
{
namespace layers
{
class BitBrainLayer
{
  public:
    struct Config
    {
        size_t batch_size = 1;
        size_t seq_len = 1;
        size_t input_dim;
        size_t hidden_dim;
        size_t output_dim;

        components::GlialConfig glial_config;
        float vigilance = 0.5f; // For M-DSiLU

        // Biomimetic cross-module adaptation
        float glial_target_novelty_gain = 0.0f; // target_sparsity += gain * (1 - familiarity)
        float glial_priming_rate = 0.0f;        // blend current threshold toward hippocampus suggested_threshold
    };

    BitBrainLayer(s::queue &q, const Config &cfg)
        : cfg_(cfg), queue_(q), proj_in_(cfg.input_dim, cfg.hidden_dim, false),
          ssm_core_({cfg.batch_size, cfg.seq_len, cfg.hidden_dim}), activation_(cfg.vigilance),
          proj_out_(cfg.hidden_dim, cfg.output_dim, false), glial_(cfg.glial_config),
          hippocampus_({// Config for hippocampus
                        .batch_size = 1,
                        .input_dim = cfg.input_dim,
                        .hidden_dim = cfg.hidden_dim})
    {
        // Allocate intermediate buffers for Backprop
        buf_H_proj = new s::buffer<float, 1>(s::range<1>(cfg.batch_size * cfg.hidden_dim));
        buf_H_ssm = new s::buffer<float, 1>(s::range<1>(cfg.batch_size * cfg.hidden_dim));
        buf_H_act = new s::buffer<float, 1>(s::range<1>(cfg.batch_size * cfg.hidden_dim));

        // Gradients
        grad_H_act = new s::buffer<float, 1>(s::range<1>(cfg.batch_size * cfg.hidden_dim));
        grad_H_ssm = new s::buffer<float, 1>(s::range<1>(cfg.batch_size * cfg.hidden_dim));
        grad_H_proj = new s::buffer<float, 1>(s::range<1>(cfg.batch_size * cfg.hidden_dim));

        // SSM params (fixed/learned?)
        // For non-sequential tasks like image classification batches, we want minimal history interference.
        // Setting A to 0.0 makes S[t] = X[t], effectively bypassing the recurrent state.
        std::vector<float> h_A(cfg.hidden_dim, 0.0f);
        buf_SSM_A = new s::buffer<float, 1>(h_A.data(), s::range<1>(cfg.hidden_dim));
        buf_SSM_State = new s::buffer<float, 1>(s::range<1>(cfg.batch_size * cfg.hidden_dim));
    }

    ~BitBrainLayer()
    {
        delete buf_H_proj;
        delete buf_H_ssm;
        delete buf_H_act;
        delete grad_H_act;
        delete grad_H_ssm;
        delete grad_H_proj;
        delete buf_SSM_A;
        delete buf_SSM_State;
    }

    // Forward Pass
    void forward(s::buffer<float, 1> &input, s::buffer<float, 1> &output, s::buffer<float, 1> &w_fast,
                 s::buffer<float, 1> &w_slow)
    {
        size_t batch = cfg_.batch_size;
        size_t hidden = cfg_.hidden_dim;

        // Reset SSM state for non-sequential tasks (e.g., image classification batches)
        queue_.submit([&](s::handler &h) {
            s::accessor acc(*buf_SSM_State, h, s::write_only, s::no_init);
            h.parallel_for(s::range<1>(batch * hidden), [=](s::id<1> idx) { acc[idx] = 0.0f; });
        });

        // 1. Hippocampus: Track the current working threshold so suggested_threshold is on the right scale.
        {
            float th = glial_.get_threshold();
            float learned = hippocampus_.get_learned_threshold();
            hippocampus_.set_learned_threshold(0.99f * learned + 0.01f * th);
        }

        // 2. Hippocampus: Compute Familiarity (+ suggested threshold)
        auto recall = hippocampus_.compute_familiarity(queue_, input, w_fast, w_slow);
        float familiarity = recall.familiarity_score;

        // 2.5 Glial: context-dependent target sparsity (novelty-driven exploration)
        if (cfg_.glial_target_novelty_gain != 0.0f)
        {
            float target = cfg_.glial_config.target_sparsity + cfg_.glial_target_novelty_gain * (1.0f - familiarity);
            target = s::clamp(target, 0.05f, 0.95f);
            glial_.set_target_sparsity(target);
        }

        // 2.6 Glial: threshold priming toward hippocampus suggested threshold
        if (cfg_.glial_priming_rate > 0.0f)
        {
            float mix = s::clamp(cfg_.glial_priming_rate * (1.0f - familiarity), 0.0f, 1.0f);
            float cur = glial_.get_threshold();
            float primed = (1.0f - mix) * cur + mix * recall.suggested_threshold;
            glial_.set_threshold(primed);
        }

        // 3. Input Projection (BitLinear)
        proj_in_.forward(queue_, input, *buf_H_proj, batch);

        // 4. SSM Scan (Temporal)
        ssm_core_.forward_single(queue_, *buf_H_proj, *buf_SSM_A, *buf_H_ssm, *buf_SSM_State);

        // 3.5 LayerNorm on SSM output to stabilize signal for Glial/Activation
        // layer_norm(queue_, *buf_H_ssm, batch, hidden, 1e-5f);

        // 5. Activation (M-DSiLU)
        float threshold = glial_.get_threshold();
        activation_.forward(queue_, *buf_H_ssm, *buf_H_act, batch * hidden, threshold, familiarity);

        // 6. Output Projection (BitLinear)
        proj_out_.forward(queue_, *buf_H_act, output, batch);

        // 7. Glial Regulation (based on activation sparsity)
        {
            // Define "active" by whether pre-activation exceeds the effective threshold.
            // This avoids counting small-magnitude negative SiLU outputs as "active".
            float eff_th = threshold + cfg_.vigilance * (1.0f - familiarity);
            s::host_accessor acc(*buf_H_ssm, s::read_only);
            int active = 0;
            for (size_t i = 0; i < batch * hidden; ++i)
            {
                if (acc[i] > eff_th)
                    active++;
            }
            glial_.regulate(active, batch * hidden);
        }

        // Store context for backward
        last_familiarity_ = familiarity;
        last_threshold_ = threshold;
    }

    // Backward Pass
    void backward(s::buffer<float, 1> &grad_output, s::buffer<float, 1> &grad_input)
    {
        size_t batch = cfg_.batch_size;
        size_t hidden = cfg_.hidden_dim;

        // 1. Output Projection
        proj_out_.backward(queue_, grad_output, *grad_H_act, batch);

        // 2. Activation (M-DSiLU)
        activation_.backward(queue_, *grad_H_act, *buf_H_ssm, *grad_H_ssm, batch * hidden, last_threshold_,
                             last_familiarity_);

        // 2.5 LayerNorm Backward
        // layer_norm_backward(queue_, *grad_H_ssm, *buf_H_ssm, *grad_H_ssm, batch, hidden, 1e-5f); // grad_H_ssm is
        // input_grad for SSM

        // 3. SSM (Simplified Backward: Pass-through or approx?)
        queue_.submit([&](s::handler &h) {
            s::accessor in(*grad_H_ssm, h, s::read_only);
            s::accessor out(*grad_H_proj, h, s::write_only, s::no_init);
            h.copy(in, out);
        });

        // 4. Input Projection
        proj_in_.backward(queue_, *grad_H_proj, grad_input, batch);
    }

    // Update
    void step(float lr, float weight_decay = 0.0f)
    {
        proj_in_.step(queue_, lr, weight_decay);
        proj_out_.step(queue_, lr, weight_decay);
    }

    // Accessors
    components::GlialCell &get_glial()
    {
        return glial_;
    }
    components::Hippocampus &get_hippocampus()
    {
        return hippocampus_;
    }

  private:
    Config cfg_;
    s::queue &queue_;

    // Sub-modules
    BitLinear proj_in_;
    SSMScan ssm_core_;
    MDSiLU activation_;
    BitLinear proj_out_;

    components::GlialCell glial_;
    components::Hippocampus hippocampus_;

    // Buffers
    s::buffer<float, 1> *buf_H_proj;
    s::buffer<float, 1> *buf_H_ssm;
    s::buffer<float, 1> *buf_H_act; // Output of Activation (Input to L2)

    s::buffer<float, 1> *buf_SSM_A;
    s::buffer<float, 1> *buf_SSM_State;

    // Gradients
    s::buffer<float, 1> *grad_H_act;
    s::buffer<float, 1> *grad_H_ssm;
    s::buffer<float, 1> *grad_H_proj;

    // Context
    float last_familiarity_ = 0.0f;
    float last_threshold_ = 0.0f;

    // Private helper for LayerNorm (declarations)
    void layer_norm(s::queue &q, s::buffer<float, 1> &buf, size_t batch_size_arg, size_t dim_arg, float epsilon);
    void layer_norm_backward(s::queue &q, s::buffer<float, 1> &grad_output, s::buffer<float, 1> &input,
                             s::buffer<float, 1> &grad_input, size_t batch_size_arg, size_t dim_arg, float epsilon);
};

// Definitions of LayerNorm member functions outside the class definition
void BitBrainLayer::layer_norm(s::queue &q, s::buffer<float, 1> &buf, size_t batch_size_arg, size_t dim_arg,
                               float epsilon)
{
    q.submit([&](s::handler &h) {
        s::accessor acc(buf, h, s::read_write);
        h.parallel_for(s::range<1>(batch_size_arg), [=](s::id<1> idx_b) {
            size_t b = idx_b[0];
            float sum = 0.0f;
            float sum_sq = 0.0f;
            for (size_t i = 0; i < dim_arg; ++i)
            {
                float v = acc[b * dim_arg + i];
                sum += v;
                sum_sq += v * v;
            }
            float mean = sum / dim_arg;
            float var = sum_sq / dim_arg - mean * mean;
            float std_dev = s::rsqrt(var + epsilon);

            for (size_t i = 0; i < dim_arg; ++i)
            {
                acc[b * dim_arg + i] = (acc[b * dim_arg + i] - mean) * std_dev;
            }
        });
    });
}

void BitBrainLayer::layer_norm_backward(s::queue &q, s::buffer<float, 1> &grad_output, s::buffer<float, 1> &input,
                                        s::buffer<float, 1> &grad_input, size_t batch_size_arg, size_t dim_arg,
                                        float epsilon)
{
    q.submit([&](s::handler &h) {
        s::accessor d_out(grad_output, h, s::read_only);
        s::accessor x(input, h, s::read_only);
        s::accessor d_in(grad_input, h, s::write_only, s::no_init);
        h.parallel_for(s::range<1>(batch_size_arg), [=](s::id<1> idx_b) {
            size_t b = idx_b[0];
            float sum = 0.0f;
            float sum_sq = 0.0f;
            for (size_t i = 0; i < dim_arg; ++i)
            {
                float v = x[b * dim_arg + i];
                sum += v;
                sum_sq += v * v;
            }
            float mean = sum / dim_arg;
            float var = sum_sq / dim_arg - mean * mean;
            float std_dev_inv = s::rsqrt(var + epsilon);

            for (size_t i = 0; i < dim_arg; ++i)
            {
                float x_centered = x[b * dim_arg + i] - mean;
                float grad_x_norm = d_out[b * dim_arg + i] * std_dev_inv;

                d_in[b * dim_arg + i] = grad_x_norm;
            }
        });
    });
}
} // namespace layers
} // namespace neurobit
