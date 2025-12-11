#pragma once

#include "neurobit/core/types.hpp"
#include <sycl/sycl.hpp>
#include <vector>
#include <cmath>

namespace neurobit
{
    namespace layers
    {

        // OU (Ornstein-Uhlenbeck) 噪聲配置
        struct OUNoiseConfig
        {
            float theta = 0.15f;    // 回復速率
            float mu = 0.0f;        // 長期均值
            float sigma = 0.3f;     // 波動強度
            float dt = 1.0f;        // 時間步長
            bool enabled = true;
        };

        class SpikeNeuron
        {
        public:
            struct Config
            {
                size_t batch_size;
                size_t seq_len;
                size_t state_dim;
                float v_threshold = 1.0f;
                float v_decay = 0.5f;       // alpha: 膜電位衰減
                float v_reset_soft = true;  // 軟重置 vs 硬重置
                OUNoiseConfig ou_config;
            };

            SpikeNeuron(const Config &config) : cfg_(config) {}

            // 支援動態調整閾值 (Glial Control)
            void set_threshold(float new_th) { cfg_.v_threshold = new_th; }
            float get_threshold() const { return cfg_.v_threshold; }

            // 設置噪聲強度 (Glial Control)
            void set_noise_gain(float gain) { noise_gain_ = gain; }
            float get_noise_gain() const { return noise_gain_; }

            // 單步前向 (seq_len=1 最佳化版本)
            // buf_X: Input [B*D]
            // buf_V: Membrane potential state [B*D] (read/write)
            // buf_Z: Spike output [B*D]
            // buf_OU: OU noise state [B*D] (read/write)
            // buf_Activity: Spike counter [1]
            // random_seed: 用於生成隨機數的種子
            void forward_single(s::queue &q,
                                s::buffer<float, 1> &buf_X,
                                s::buffer<float, 1> &buf_V,
                                s::buffer<float, 1> &buf_Z,
                                s::buffer<float, 1> &buf_OU,
                                s::buffer<int, 1> &buf_Activity,
                                uint32_t random_seed = 12345)
            {
                const size_t B = cfg_.batch_size;
                const size_t D = cfg_.state_dim;
                const float th = cfg_.v_threshold;
                const float alpha = cfg_.v_decay;
                const bool soft_reset = cfg_.v_reset_soft;
                const float noise_gain = noise_gain_;

                // OU 參數
                const float ou_theta = cfg_.ou_config.theta;
                const float ou_mu = cfg_.ou_config.mu;
                const float ou_sigma = cfg_.ou_config.sigma;
                const float ou_dt = cfg_.ou_config.dt;
                const bool ou_enabled = cfg_.ou_config.enabled;

                q.submit([&](s::handler &h)
                         {
                    s::accessor acc_X{buf_X, h, s::read_only};
                    s::accessor acc_V{buf_V, h, s::read_write};
                    s::accessor acc_Z{buf_Z, h, s::write_only, s::no_init};
                    s::accessor acc_OU{buf_OU, h, s::read_write};
                    s::accessor acc_Act{buf_Activity, h, s::read_write};

                    s::range<1> global_range{B * D};

                    h.parallel_for(global_range, [=](s::id<1> idx) {
                        size_t i = idx[0];

                        // 簡單 LCG 隨機數生成 (per work-item)
                        uint32_t rng_state = random_seed + static_cast<uint32_t>(i) * 1664525u;
                        auto rand_uniform = [&]() -> float {
                            rng_state = rng_state * 1664525u + 1013904223u;
                            return static_cast<float>(rng_state) / 4294967296.0f;
                        };
                        auto rand_normal = [&]() -> float {
                            // Box-Muller 變換
                            float u1 = rand_uniform() + 1e-10f;
                            float u2 = rand_uniform();
                            return s::sqrt(-2.0f * s::log(u1)) * s::cos(2.0f * 3.14159265f * u2);
                        };

                        // 1. OU 噪聲更新
                        float ou_val = acc_OU[i];
                        if (ou_enabled) {
                            float dW = rand_normal() * s::sqrt(ou_dt);
                            ou_val = ou_val + ou_theta * (ou_mu - ou_val) * ou_dt + ou_sigma * dW;
                            acc_OU[i] = ou_val;
                        }

                        // 2. 膜電位更新
                        float v = acc_V[i];
                        float input_current = acc_X[i];
                        v = alpha * v + input_current + noise_gain * ou_val;

                        // 3. 脈衝生成
                        float spike = 0.0f;
                        if (v >= th) {
                            spike = 1.0f;
                            // 軟重置 vs 硬重置
                            v = soft_reset ? (v - th) : 0.0f;

                            // 原子計數
                            s::atomic_ref<int, s::memory_order::relaxed, s::memory_scope::device>
                                atomic_counter(acc_Act[0]);
                            atomic_counter += 1;
                        }

                        acc_V[i] = v;
                        acc_Z[i] = spike;
                    }); });
            }

            // 完整序列前向 (保留用於批量處理)
            void forward(s::queue &q,
                         s::buffer<float, 1> &buf_X,  // [B*L*D]
                         s::buffer<float, 1> &buf_V,  // [B*D] persistent state
                         s::buffer<float, 1> &buf_Z,  // [B*L*D] output
                         s::buffer<float, 1> &buf_OU, // [B*D] OU state
                         s::buffer<int, 1> &buf_Activity,
                         uint32_t random_seed = 12345)
            {
                const size_t B = cfg_.batch_size;
                const size_t L = cfg_.seq_len;
                const size_t D = cfg_.state_dim;
                const float th = cfg_.v_threshold;
                const float alpha = cfg_.v_decay;
                const bool soft_reset = cfg_.v_reset_soft;
                const float noise_gain = noise_gain_;

                const float ou_theta = cfg_.ou_config.theta;
                const float ou_mu = cfg_.ou_config.mu;
                const float ou_sigma = cfg_.ou_config.sigma;
                const float ou_dt = cfg_.ou_config.dt;
                const bool ou_enabled = cfg_.ou_config.enabled;

                q.submit([&](s::handler &h)
                         {
                    s::accessor acc_X{buf_X, h, s::read_only};
                    s::accessor acc_V{buf_V, h, s::read_write};
                    s::accessor acc_Z{buf_Z, h, s::write_only, s::no_init};
                    s::accessor acc_OU{buf_OU, h, s::read_write};
                    s::accessor acc_Act{buf_Activity, h, s::read_write};

                    s::range<2> global_range{B, D};

                    h.parallel_for(global_range, [=](s::id<2> idx) {
                        size_t b = idx[0];
                        size_t d = idx[1];
                        size_t state_idx = b * D + d;

                        // RNG 初始化
                        uint32_t rng_state = random_seed + static_cast<uint32_t>(state_idx) * 1664525u;
                        auto rand_uniform = [&]() -> float {
                            rng_state = rng_state * 1664525u + 1013904223u;
                            return static_cast<float>(rng_state) / 4294967296.0f;
                        };
                        auto rand_normal = [&]() -> float {
                            float u1 = rand_uniform() + 1e-10f;
                            float u2 = rand_uniform();
                            return s::sqrt(-2.0f * s::log(u1)) * s::cos(2.0f * 3.14159265f * u2);
                        };

                        float v = acc_V[state_idx];
                        float ou_val = acc_OU[state_idx];
                        int local_spikes = 0;

                        for (size_t t = 0; t < L; ++t) {
                            size_t x_idx = b * L * D + t * D + d;

                            // OU 更新
                            if (ou_enabled) {
                                float dW = rand_normal() * s::sqrt(ou_dt);
                                ou_val = ou_val + ou_theta * (ou_mu - ou_val) * ou_dt + ou_sigma * dW;
                            }

                            // 膜電位更新
                            float input_current = acc_X[x_idx];
                            v = alpha * v + input_current + noise_gain * ou_val;

                            // 脈衝生成
                            float spike = 0.0f;
                            if (v >= th) {
                                spike = 1.0f;
                                v = soft_reset ? (v - th) : 0.0f;
                                local_spikes++;
                            }

                            acc_Z[x_idx] = spike;
                        }

                        // 保存狀態
                        acc_V[state_idx] = v;
                        acc_OU[state_idx] = ou_val;

                        // 批量更新計數器
                        if (local_spikes > 0) {
                            s::atomic_ref<int, s::memory_order::relaxed, s::memory_scope::device>
                                atomic_counter(acc_Act[0]);
                            atomic_counter += local_spikes;
                        }
                    }); });
            }

            // 重置狀態
            void reset_state(s::queue &q,
                             s::buffer<float, 1> &buf_V,
                             s::buffer<float, 1> &buf_OU)
            {
                const size_t total = cfg_.batch_size * cfg_.state_dim;
                q.submit([&](s::handler &h)
                         {
                    s::accessor acc_V{buf_V, h, s::write_only, s::no_init};
                    s::accessor acc_OU{buf_OU, h, s::write_only, s::no_init};
                    h.parallel_for(s::range<1>{total}, [=](s::id<1> idx) {
                        acc_V[idx] = 0.0f;
                        acc_OU[idx] = 0.0f;
                    }); });
            }

        private:
            Config cfg_;
            float noise_gain_ = 1.0f;
        };

    } // namespace layers
} // namespace neurobit
