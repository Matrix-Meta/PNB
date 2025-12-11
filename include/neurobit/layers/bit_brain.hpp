#pragma once

#include "neurobit/core/types.hpp"
#include "neurobit/layers/bit_linear.hpp"
#include "neurobit/layers/ssm_scan.hpp"
#include "neurobit/layers/spike_neuron.hpp"
#include "neurobit/components/glial.hpp"
#include "neurobit/components/hippocampus.hpp"
#include <sycl/sycl.hpp>
#include <random>

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
                int max_think_steps = 50;
                int min_think_steps = 3;
                int learn_after_steps = 3;

                components::GlialConfig glial_config;
                OUNoiseConfig ou_config;

                float hebbian_lr = 0.1f;
                float consolidation_rate = 0.1f;
                float decay_rate = 0.02f;
                float familiarity_threshold = 0.3f;
                float priming_strength = 0.5f;
                float injection_scale = 0.1f;

                float ffi_strength = 0.5f;

                bool use_usm = true;
                size_t work_group_size = 256;
            };

            struct ForwardResult
            {
                int reasoning_steps;
                float final_threshold;
                float final_firing_rate;
                float familiarity_score;
                bool is_familiar;
                bool learned;
            };

            BitBrainLayer(s::queue& q, const Config &cfg)
                : cfg_(cfg),
                  queue_(q),
                  proj_in_(q, cfg.input_dim, cfg.hidden_dim),
                  ssm_core_({cfg.batch_size, cfg.seq_len, cfg.hidden_dim}),
                  snn_core_({cfg.batch_size, cfg.seq_len, cfg.hidden_dim,
                             cfg.glial_config.initial_threshold, 0.5f, true, cfg.ou_config}),
                  proj_out_(q, cfg.hidden_dim, cfg.output_dim),
                  glial_(cfg.glial_config),
                  hippocampus_({
                      .batch_size = 1,
                      .input_dim = cfg.input_dim,
                      .hidden_dim = cfg.hidden_dim,
                      .learning_rate = cfg.hebbian_lr,
                      .consolidation_rate = cfg.consolidation_rate,
                      .decay_rate = cfg.decay_rate,
                      .weight_clip = 1.0f,
                      .familiarity_threshold = cfg.familiarity_threshold,
                      .priming_strength = cfg.priming_strength,
                      .injection_scale = cfg.injection_scale,
                      .normalize_injection = true,
                  }),
                  rng_(std::random_device{}())
            {
            }

            ForwardResult forward_reasoning(
                s::queue &q,
                s::buffer<float, 1> &buf_X,
                s::buffer<int8_t, 1> &buf_Win,
                s::buffer<float, 1> &buf_Assm,
                s::buffer<int8_t, 1> &buf_Wout,
                s::buffer<float, 1> &buf_Y,
                s::buffer<float, 1> &buf_W_fast,
                s::buffer<float, 1> &buf_W_slow,
                s::buffer<float, 1> &buf_SSM_state,
                s::buffer<float, 1> &buf_SNN_v,
                s::buffer<float, 1> &buf_SNN_ou,
                s::buffer<float, 1> &buf_H_proj,
                s::buffer<float, 1> &buf_H_ssm,
                s::buffer<float, 1> &buf_H_snn_z)
            {
                ForwardResult result{0, 0.0f, 0.0f, 0.0f, false, false};
                
                // 1. Familiarity
                auto recall = hippocampus_.compute_familiarity(q, buf_X, buf_W_fast, buf_W_slow);
                result.familiarity_score = recall.familiarity_score;
                result.is_familiar = recall.is_familiar;

                float initial_threshold = recall.suggested_threshold;
                glial_.set_threshold(initial_threshold);

                // 2. Input Projection (Direct Float -> BF16 inside)
                proj_in_.forward_single(q, buf_X, buf_Win, buf_H_proj);

                // 3. SSM
                ssm_core_.forward_single(q, buf_H_proj, buf_Assm, buf_H_ssm, buf_SSM_state);

                // 4. Injection
                if (recall.familiarity_score > 0.1f) {
                    hippocampus_.inject_memory(q, buf_X, buf_W_fast, buf_W_slow,
                                               buf_H_ssm, recall.familiarity_score);
                }

                q.wait();

                // 5. Reasoning Loop
                result = forward_reasoning_loop(q, buf_H_ssm, buf_SNN_v, buf_H_snn_z, buf_SNN_ou);
                result.familiarity_score = recall.familiarity_score;
                result.is_familiar = recall.is_familiar;

                // 6. Learning
                if (result.reasoning_steps > cfg_.learn_after_steps) {
                    hippocampus_.learn(q, buf_X, buf_H_snn_z, buf_W_fast, result.final_threshold);
                    result.learned = true;
                }

                // 7. Output Projection (Direct Float -> BF16 inside)
                proj_out_.forward_single(q, buf_H_snn_z, buf_Wout, buf_Y);
                q.wait();

                return result;
            }

            void sleep(s::queue &q, s::buffer<float, 1> &buf_W_fast, s::buffer<float, 1> &buf_W_slow) {
                hippocampus_.consolidate(q, buf_W_fast, buf_W_slow);
                q.wait();
            }

            void reset_state(s::queue &q, s::buffer<float, 1> &buf_SSM_state, s::buffer<float, 1> &buf_SNN_v, s::buffer<float, 1> &buf_SNN_ou) {
                ssm_core_.reset_state(q, buf_SSM_state);
                snn_core_.reset_state(q, buf_SNN_v, buf_SNN_ou);
                q.wait();
            }

            float get_current_threshold() const { return glial_.get_threshold(); }
            float get_learned_threshold() const { return hippocampus_.get_learned_threshold(); }
            float get_last_familiarity() const { return hippocampus_.get_last_familiarity(); }
            const components::GlialCell &get_glial() const { return glial_; }
            components::GlialCell &get_glial() { return glial_; }
            const Config &config() const { return cfg_; }

        private:
            Config cfg_;
            s::queue& queue_;
            BitLinearXMX<bfloat16> proj_in_;
            SSMScan ssm_core_;
            SpikeNeuron snn_core_;
            BitLinearXMX<bfloat16> proj_out_;
            components::GlialCell glial_;
            components::Hippocampus hippocampus_;
            std::mt19937 rng_;

            ForwardResult forward_reasoning_loop(
                s::queue &q,
                s::buffer<float, 1> &buf_H_ssm,
                s::buffer<float, 1> &buf_SNN_v,
                s::buffer<float, 1> &buf_H_snn_z,
                s::buffer<float, 1> &buf_SNN_ou)
            {
                ForwardResult result{0, 0.0f, 0.0f, 0.0f, false, false};
                const size_t hidden_dim = cfg_.hidden_dim;

                int *p_activity = s::malloc_shared<int>(1, q);
                int *p_stable = s::malloc_shared<int>(1, q);
                float *p_threshold = s::malloc_shared<float>(1, q);
                float *p_firing_rate = s::malloc_shared<float>(1, q);

                *p_activity = 0;
                *p_stable = 0;
                *p_threshold = glial_.get_threshold();
                *p_firing_rate = 0.0f;

                const float target_rate = cfg_.glial_config.target_sparsity;
                const float stability_tol = cfg_.glial_config.stability_tolerance;
                const float ffi_strength = cfg_.ffi_strength;
                const int max_steps = cfg_.max_think_steps;
                const int min_steps = cfg_.min_think_steps;
                const float ou_theta = cfg_.ou_config.theta;
                const float ou_mu = cfg_.ou_config.mu;
                const float ou_sigma = cfg_.ou_config.sigma;
                const float ou_dt = cfg_.ou_config.dt;
                const bool ou_enabled = cfg_.ou_config.enabled;

                uint32_t rng_seed = rng_();
                int final_step = 0;

                for (int step = 0; step < max_steps; ++step)
                {
                    *p_activity = 0;

                    q.submit([&](s::handler &h) {
                        s::accessor acc_H{buf_H_ssm, h, s::read_only};
                        s::accessor acc_V{buf_SNN_v, h, s::read_write};
                        s::accessor acc_Z{buf_H_snn_z, h, s::write_only, s::no_init};
                        s::accessor acc_OU{buf_SNN_ou, h, s::read_write};

                        s::local_accessor<int, 1> local_act{s::range<1>{256}, h};
                        size_t global_size = ((hidden_dim + 255) / 256) * 256;

                        h.parallel_for(
                            s::nd_range<1>{s::range<1>{global_size}, s::range<1>{256}},
                            [=](s::nd_item<1> it) {
                                size_t i = it.get_global_id(0);
                                size_t lid = it.get_local_id(0);
                                int my_spike = 0;

                                if (i < hidden_dim) {
                                    float input = acc_H[i];
                                    float energy = s::fabs(input);
                                    float noise = 0.0f;
                                    if (ou_enabled) {
                                        uint32_t seed = rng_seed + step * 1000 + i;
                                        seed = seed * 1664525u + 1013904223u;
                                        float u1 = (seed & 0xFFFFFF) / 16777216.0f + 1e-6f;
                                        seed = seed * 1664525u + 1013904223u;
                                        float u2 = (seed & 0xFFFFFF) / 16777216.0f;
                                        float z0 = s::sqrt(-2.0f * s::log(u1)) * s::cos(2.0f * 3.14159265f * u2);
                                        float ou_state = acc_OU[i];
                                        ou_state += ou_theta * (ou_mu - ou_state) * ou_dt + ou_sigma * s::sqrt(ou_dt) * z0;
                                        acc_OU[i] = ou_state;
                                        noise = ou_state;
                                    }
                                    float threshold = *p_threshold + ffi_strength * (energy / static_cast<float>(hidden_dim));
                                    float v = acc_V[i];
                                    v = v * 0.9f + input + noise;
                                    float spike = (v >= threshold) ? 1.0f : 0.0f;
                                    if (spike > 0.5f) {
                                        v = 0.0f;
                                        my_spike = 1;
                                    }
                                    acc_V[i] = v;
                                    acc_Z[i] = spike;
                                }
                                local_act[lid] = my_spike;
                                it.barrier(s::access::fence_space::local_space);
                                for (size_t s = 128; s > 0; s >>= 1) {
                                    if (lid < s) local_act[lid] += local_act[lid + s];
                                    it.barrier(s::access::fence_space::local_space);
                                }
                                if (lid == 0) {
                                    s::atomic_ref<int, s::memory_order::relaxed, s::memory_scope::device>
                                        atomic_act(*p_activity);
                                    atomic_act += local_act[0];
                                }
                            }); 
                    });

                    q.submit([&](s::handler &h) {
                        h.single_task([=]() {
                            int act = *p_activity;
                            float rate = static_cast<float>(act) / static_cast<float>(hidden_dim);
                            float error = rate - target_rate;
                            float th = *p_threshold;
                            th += 1.0f * error;
                            if (th < 0.1f) th = 0.1f;
                            if (th > 50.0f) th = 50.0f;
                            *p_threshold = th;
                            *p_firing_rate = rate;
                            float abs_error = (error > 0) ? error : -error;
                            *p_stable = (abs_error < stability_tol) ? 1 : 0;
                        }); 
                    });

                    if (step >= min_steps - 1 && (step % 2 == 0 || step == max_steps - 1)) {
                        q.wait();
                        final_step = step + 1;
                        if (*p_stable) break;
                    }
                }

                q.wait();
                glial_.set_threshold(*p_threshold);
                glial_.update_sparsity(*p_firing_rate);
                result.reasoning_steps = final_step > 0 ? final_step : max_steps;
                result.final_threshold = *p_threshold;
                result.final_firing_rate = *p_firing_rate;

                s::free(p_activity, q);
                s::free(p_stable, q);
                s::free(p_threshold, q);
                s::free(p_firing_rate, q);

                return result;
            }
        };
    }
}