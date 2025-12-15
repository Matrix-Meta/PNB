#pragma once

#include "neurobit/core/types.hpp"
#include <cmath>
#include <sycl/sycl.hpp>
#include <vector>

namespace neurobit
{
namespace layers
{

struct OUNoiseConfig
{
    float theta = 0.15f;
    float mu = 0.0f;
    float sigma = 0.3f;
    float dt = 1.0f;
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
        float v_decay = 0.5f;
        float v_reset_soft = true;
        OUNoiseConfig ou_config;
    };

    SpikeNeuron(s::queue &q, const Config &config) : cfg_(config)
    {
    }

    void set_threshold(float new_th)
    {
        cfg_.v_threshold = new_th;
    }
    float get_threshold() const
    {
        return cfg_.v_threshold;
    }

    void set_noise_gain(float gain)
    {
        noise_gain_ = gain;
    }
    float get_noise_gain() const
    {
        return noise_gain_;
    }

    // Lightweight Philox RNG-based Spike Neuron
    void forward_single(s::queue &q, s::buffer<float, 1> &buf_X, s::buffer<float, 1> &buf_V, s::buffer<float, 1> &buf_Z,
                        s::buffer<float, 1> &buf_OU, s::buffer<int, 1> &buf_Activity, uint32_t step_seed = 12345)
    {
        const size_t B = cfg_.batch_size;
        const size_t D = cfg_.state_dim;
        const float th = cfg_.v_threshold;
        const float alpha = cfg_.v_decay;
        const bool soft_reset = cfg_.v_reset_soft;
        const float noise_gain = noise_gain_;

        const float ou_theta = cfg_.ou_config.theta;
        const float ou_mu = cfg_.ou_config.mu;
        const float ou_dt = cfg_.ou_config.dt;
        const bool ou_enabled = cfg_.ou_config.enabled;
        const float ou_sigma = cfg_.ou_config.sigma;

        size_t total_items = B * D;
        size_t work_group_size = 256;
        size_t padded_global_items = (total_items + work_group_size - 1) / work_group_size * work_group_size;

        s::range<1> global_range{padded_global_items};
        s::range<1> local_range{work_group_size};

        q.submit([&](s::handler &h) {
            s::accessor acc_X{buf_X, h, s::read_only};
            s::accessor acc_V{buf_V, h, s::read_write};
            s::accessor acc_Z{buf_Z, h, s::write_only, s::no_init};
            s::accessor acc_OU{buf_OU, h, s::read_write};
            s::accessor acc_Act{buf_Activity, h, s::read_write};

            h.parallel_for(s::nd_range<1>{global_range, local_range}, [=](s::nd_item<1> it) [[sycl::reqd_sub_group_size(
                                                                          16)]] {
                size_t i = it.get_global_id(0);
                auto sg = it.get_sub_group();
                int my_spike_count = 0;

                if (i < total_items)
                {
                    float ou_val = acc_OU[i];

                    if (ou_enabled)
                    {
                        // High-performance Philox-like hash for uniform noise
                        uint32_t state = step_seed + i;
                        state ^= state >> 16;
                        state *= 0x85ebca6b;
                        state ^= state >> 13;
                        state *= 0xc2b2ae35;
                        state ^= state >> 16;

                        // Uniform [-1, 1]
                        float z0 = (static_cast<float>(state) / 4294967296.0f) * 2.0f - 1.0f;

                        // Approximate Normal (Variance matching)
                        z0 *= 1.732f;

                        ou_val =
                            ou_val + ou_theta * (ou_mu - ou_val) * ou_dt + ou_sigma * noise_gain * s::sqrt(ou_dt) * z0;
                        acc_OU[i] = ou_val;
                    }

                    float v = acc_V[i];
                    float input_current = acc_X[i];
                    v = alpha * v + input_current + noise_gain * ou_val;

                    float spike = 0.0f;
                    if (v >= th)
                    {
                        spike = 1.0f;
                        v = soft_reset ? (v - th) : 0.0f;
                        my_spike_count = 1;
                    }

                    acc_V[i] = v;
                    acc_Z[i] = spike;
                }

                int sg_spikes = s::reduce_over_group(sg, my_spike_count, s::plus<int>());

                if (sg.get_local_id()[0] == 0 && sg_spikes > 0)
                {
                    s::atomic_ref<int, s::memory_order::relaxed, s::memory_scope::device> atomic_counter(acc_Act[0]);
                    atomic_counter += sg_spikes;
                }
            });
        });
    }

    // Keep full forward for compatibility (omitted optimization for brevity)
    void forward(s::queue &q, s::buffer<float, 1> &buf_X, s::buffer<float, 1> &buf_V, s::buffer<float, 1> &buf_Z,
                 s::buffer<float, 1> &buf_OU, s::buffer<int, 1> &buf_Activity, uint32_t random_seed = 12345)
    {
        forward_single(q, buf_X, buf_V, buf_Z, buf_OU, buf_Activity, random_seed);
    }

    void reset_state(s::queue &q, s::buffer<float, 1> &buf_V, s::buffer<float, 1> &buf_OU)
    {
        const size_t total = cfg_.batch_size * cfg_.state_dim;
        q.submit([&](s::handler &h) {
            s::accessor acc_V{buf_V, h, s::write_only, s::no_init};
            s::accessor acc_OU{buf_OU, h, s::write_only, s::no_init};
            h.parallel_for(s::range<1>{total}, [=](s::id<1> idx) {
                acc_V[idx] = 0.0f;
                acc_OU[idx] = 0.0f;
            });
        });
    }

  private:
    Config cfg_;
    float noise_gain_ = 1.0f;
};

} // namespace layers
} // namespace neurobit
