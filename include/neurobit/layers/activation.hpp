#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sycl/sycl.hpp>
#include <vector>

namespace neurobit
{
namespace layers
{

namespace s = sycl;

/**
 * M-DSiLU (Memory-modulated Dynamic SiLU)
 *
 * Upgraded Formula (5-in-1):
 * - Grouped thresholds: eff_th = theta_base + th_group[g] + vigilance * novelty(familiarity)
 * - Plastic novelty: novelty = f(1 - familiarity; p, sharpness)
 * - Multiplicative gating: y = SiLU(x) * sigmoid((x - eff_th)/tau)
 * - Leaky negative tail:  + alpha * min(0, x - eff_th)
 * - Temperature/slope: tau controls gate softness
 *
 * - theta_base: Provided by Glial Cell (Homeostasis)
 * - familiarity: Provided by Hippocampus (0.0 = Novel, 1.0 = Familiar)
 * - vigilance: Hyperparameter controlling how much novelty suppresses activity
 *
 * Characteristics:
 * - Smooth, non-monotonic (like SiLU)
 * - Dynamic sparsity (controlled by theta)
 * - Context-aware (controlled by familiarity)
 */
class MDSiLU
{
  public:
    float vigilance;
    float tau = 1.0f;
    float alpha = 0.02f;
    float novelty_power = 1.0f;     // p
    float novelty_sharpness = 1.0f; // saturating sharpness (>0)

    float group_offset_lr = 0.05f;
    float group_offset_decay = 0.999f;
    float group_offset_max_abs = 1.0f;

    MDSiLU(float vigilance_factor = 0.5f) : vigilance(vigilance_factor)
    {
        s::host_accessor acc(fallback_group_offsets_, s::write_only);
        acc[0] = 0.0f;
    }

    void configure_threshold_groups(size_t hidden_dim, size_t group_size)
    {
        hidden_dim_ = std::max<size_t>(1, hidden_dim);
        group_size_ = std::max<size_t>(1, group_size);
        group_count_ = (hidden_dim_ + group_size_ - 1) / group_size_;

        group_offsets_host_.assign(group_count_, 0.0f);
        group_offsets_dev_ = std::make_unique<s::buffer<float, 1>>(s::range<1>(group_count_));
        {
            s::host_accessor acc(*group_offsets_dev_, s::write_only);
            for (size_t i = 0; i < group_count_; ++i)
                acc[i] = 0.0f;
        }
    }

    void sync_group_offsets_from_device()
    {
        size_t gc = group_count();
        group_offsets_host_.assign(gc, 0.0f);
        if (group_offsets_dev_)
        {
            s::host_accessor acc(*group_offsets_dev_, s::read_only);
            for (size_t g = 0; g < gc; ++g)
                group_offsets_host_[g] = acc[g];
        }
        else
        {
            s::host_accessor acc(fallback_group_offsets_, s::read_only);
            group_offsets_host_[0] = acc[0];
        }
    }

    void copy_group_offsets_host(std::vector<float> &out, bool refresh_from_device = true)
    {
        if (refresh_from_device)
            sync_group_offsets_from_device();
        out = group_offsets_host_;
    }

    void set_group_offsets_host(const std::vector<float> &values)
    {
        size_t gc = group_count();
        group_offsets_host_.assign(gc, 0.0f);
        for (size_t g = 0; g < gc && g < values.size(); ++g)
            group_offsets_host_[g] = values[g];

        if (group_offsets_dev_)
        {
            s::host_accessor acc(*group_offsets_dev_, s::write_only);
            for (size_t g = 0; g < gc; ++g)
                acc[g] = group_offsets_host_[g];
        }
        else
        {
            s::host_accessor acc(fallback_group_offsets_, s::write_only);
            acc[0] = group_offsets_host_[0];
        }
    }

    size_t group_count() const
    {
        return std::max<size_t>(1, group_count_);
    }

    size_t group_size() const
    {
        return std::max<size_t>(1, group_size_);
    }

    size_t group_index(size_t feature_index) const
    {
        size_t gs = std::max<size_t>(1, group_size_);
        return (feature_index / gs);
    }

    float novelty_component_host(float familiarity) const
    {
        float fam = std::clamp(familiarity, 0.0f, 1.0f);
        float novelty = 1.0f - fam;
        float p = std::clamp(novelty_power, 0.1f, 8.0f);
        float sharp = std::clamp(novelty_sharpness, 0.0f, 32.0f);

        float novelty_pow = std::pow(std::max(novelty, 0.0f), p);
        if (sharp <= 0.0f)
            return novelty_pow;

        float denom = 1.0f - std::exp(-sharp);
        if (denom <= 1e-6f)
            return novelty_pow;

        float shaped = (1.0f - std::exp(-sharp * novelty_pow)) / denom;
        return std::clamp(shaped, 0.0f, 1.0f);
    }

    void get_effective_thresholds_host(float base_threshold, float familiarity, std::vector<float> &out) const
    {
        out.assign(group_count(), 0.0f);
        float novelty_shift = vigilance * novelty_component_host(familiarity);
        for (size_t g = 0; g < out.size(); ++g)
        {
            float off = (g < group_offsets_host_.size()) ? group_offsets_host_[g] : 0.0f;
            out[g] = base_threshold + off + novelty_shift;
        }
    }

    void homeostatic_update_groups(const std::vector<uint32_t> &active_counts,
                                   const std::vector<uint32_t> &total_counts, float target_active_rate)
    {
        if (!group_offsets_dev_)
            return;

        size_t gc = group_count();
        if (active_counts.size() != gc || total_counts.size() != gc || group_offsets_host_.size() != gc)
            return;

        float lr = std::clamp(group_offset_lr, 0.0f, 1.0f);
        float decay = std::clamp(group_offset_decay, 0.0f, 1.0f);
        float max_abs = std::max(group_offset_max_abs, 0.0f);
        float target = std::clamp(target_active_rate, 0.0f, 1.0f);

        float mean = 0.0f;
        for (size_t g = 0; g < gc; ++g)
        {
            float total = static_cast<float>(std::max<uint32_t>(1u, total_counts[g]));
            float rate = static_cast<float>(active_counts[g]) / total;
            float err = rate - target; // too active -> increase threshold offset

            float v = group_offsets_host_[g];
            v = decay * v + lr * err;
            v = std::clamp(v, -max_abs, max_abs);
            group_offsets_host_[g] = v;
            mean += v;
        }
        mean /= static_cast<float>(gc);
        for (size_t g = 0; g < gc; ++g)
            group_offsets_host_[g] -= mean;

        s::host_accessor acc(*group_offsets_dev_, s::write_only);
        for (size_t g = 0; g < gc; ++g)
            acc[g] = group_offsets_host_[g];
    }

    void homeostatic_update_groups_device(s::queue &q, const uint32_t *active_counts, const uint32_t *total_counts,
                                          const float *target_active_rate,
                                          std::vector<s::event> *profile_events = nullptr)
    {
        if (!group_offsets_dev_ || !active_counts || !total_counts || !target_active_rate)
            return;

        size_t gc = group_count();
        float lr = std::clamp(group_offset_lr, 0.0f, 1.0f);
        float decay = std::clamp(group_offset_decay, 0.0f, 1.0f);
        float max_abs = std::max(group_offset_max_abs, 0.0f);

        auto ev0 = q.submit([&](s::handler &h) {
            s::accessor g_off(*group_offsets_dev_, h, s::read_write);
            h.parallel_for(s::range<1>(gc), [=](s::id<1> idx) {
                size_t g = idx[0];
                uint32_t tot = total_counts[g];
                uint32_t act = active_counts[g];
                float rate = (tot > 0) ? (static_cast<float>(act) / static_cast<float>(tot)) : 0.0f;
                float err = rate - (*target_active_rate);

                float off = g_off[g];
                off = off * decay + lr * err;
                if (max_abs > 0.0f)
                    off = s::clamp(off, -max_abs, max_abs);
                g_off[g] = off;
            });
        });
        if (profile_events)
            profile_events->push_back(ev0);
    }

    void forward(s::queue &q, s::buffer<float, 1> &input, s::buffer<float, 1> &output, size_t size,
                 const float *base_threshold, const float *familiarity, uint32_t *group_active_counts = nullptr,
                 uint32_t *total_active = nullptr, std::vector<s::event> *profile_events = nullptr)
    {
        float v = vigilance;
        float t = std::clamp(tau, 0.05f, 8.0f);
        float a = std::clamp(alpha, 0.0f, 1.0f);
        float np = novelty_power;
        float ns = novelty_sharpness;

        size_t hd = std::max<size_t>(1, hidden_dim_);
        size_t gs = std::max<size_t>(1, group_size_);
        size_t gc = std::max<size_t>(1, group_count_);

        s::buffer<float, 1> &group_off = group_offsets_dev_ ? *group_offsets_dev_ : fallback_group_offsets_;
        auto ev = q.submit([&](s::handler &h) {
            s::accessor in(input, h, s::read_only);
            s::accessor out(output, h, s::write_only, s::no_init);
            s::accessor g_off(group_off, h, s::read_only);

            h.parallel_for(s::range<1>(size), [=](s::id<1> idx) {
                size_t i = idx[0];
                size_t f = (i % hd);
                size_t g = (f / gs);
                if (g >= gc)
                    g = gc - 1;

                float th = base_threshold ? *base_threshold : 0.0f;
                float fam = familiarity ? *familiarity : 0.0f;
                fam = s::clamp(fam, 0.0f, 1.0f);
                float novelty = 1.0f - fam;
                float p = s::clamp(np, 0.1f, 8.0f);
                float sharp = s::clamp(ns, 0.0f, 32.0f);

                float novelty_pow = s::pow(s::fmax(novelty, 0.0f), p);
                float novelty_shaped = novelty_pow;
                if (sharp > 0.0f)
                {
                    float denom = 1.0f - s::exp(-sharp);
                    if (denom > 1e-6f)
                    {
                        novelty_shaped = (1.0f - s::exp(-sharp * novelty_pow)) / denom;
                        novelty_shaped = s::clamp(novelty_shaped, 0.0f, 1.0f);
                    }
                }

                float novelty_shift = v * s::clamp(novelty_shaped, 0.0f, 1.0f);
                float eff_th = th + g_off[g] + novelty_shift;

                float x_shifted = in[idx] - eff_th;
                float gate = sigmoid_device(x_shifted / t);

                float x_sig = sigmoid_device(in[idx]);
                float silu = in[idx] * x_sig;

                float leaky = a * s::fmin(0.0f, x_shifted);
                out[idx] = silu * gate + leaky;

                if (group_active_counts || total_active)
                {
                    bool is_active = in[idx] > eff_th;
                    if (is_active)
                    {
                        if (group_active_counts)
                        {
                            s::atomic_ref<uint32_t, s::memory_order::relaxed, s::memory_scope::device,
                                          s::access::address_space::global_space>
                                atomic_group(group_active_counts[g]);
                            atomic_group.fetch_add(1u);
                        }
                        if (total_active)
                        {
                            s::atomic_ref<uint32_t, s::memory_order::relaxed, s::memory_scope::device,
                                          s::access::address_space::global_space>
                                atomic_total(*total_active);
                            atomic_total.fetch_add(1u);
                        }
                    }
                }
            });
        });
        if (profile_events)
            profile_events->push_back(ev);
    }

    void backward(s::queue &q, s::buffer<float, 1> &grad_output, s::buffer<float, 1> &input,
                  s::buffer<float, 1> &grad_input, size_t size, const float *base_threshold, const float *familiarity,
                  std::vector<s::event> *profile_events = nullptr)
    {
        float v = vigilance;
        float t = std::clamp(tau, 0.05f, 8.0f);
        float inv_t = 1.0f / t;
        float a = std::clamp(alpha, 0.0f, 1.0f);
        float np = novelty_power;
        float ns = novelty_sharpness;

        size_t hd = std::max<size_t>(1, hidden_dim_);
        size_t gs = std::max<size_t>(1, group_size_);
        size_t gc = std::max<size_t>(1, group_count_);

        s::buffer<float, 1> &group_off = group_offsets_dev_ ? *group_offsets_dev_ : fallback_group_offsets_;
        auto ev = q.submit([&](s::handler &h) {
            s::accessor dy(grad_output, h, s::read_only);
            s::accessor x(input, h, s::read_only);
            s::accessor dx(grad_input, h, s::write_only, s::no_init);
            s::accessor g_off(group_off, h, s::read_only);

            h.parallel_for(s::range<1>(size), [=](s::id<1> idx) {
                size_t i = idx[0];
                size_t f = (i % hd);
                size_t g = (f / gs);
                if (g >= gc)
                    g = gc - 1;

                float th = base_threshold ? *base_threshold : 0.0f;
                float fam = familiarity ? *familiarity : 0.0f;
                fam = s::clamp(fam, 0.0f, 1.0f);
                float novelty = 1.0f - fam;
                float p = s::clamp(np, 0.1f, 8.0f);
                float sharp = s::clamp(ns, 0.0f, 32.0f);

                float novelty_pow = s::pow(s::fmax(novelty, 0.0f), p);
                float novelty_shaped = novelty_pow;
                if (sharp > 0.0f)
                {
                    float denom = 1.0f - s::exp(-sharp);
                    if (denom > 1e-6f)
                    {
                        novelty_shaped = (1.0f - s::exp(-sharp * novelty_pow)) / denom;
                        novelty_shaped = s::clamp(novelty_shaped, 0.0f, 1.0f);
                    }
                }

                float novelty_shift = v * s::clamp(novelty_shaped, 0.0f, 1.0f);
                float eff_th = th + g_off[g] + novelty_shift;
                float x_shifted = x[idx] - eff_th;

                float gate = sigmoid_device(x_shifted * inv_t);
                float d_gate = gate * (1.0f - gate) * inv_t;

                float x_sig = sigmoid_device(x[idx]);
                float silu = x[idx] * x_sig;
                float d_silu = x_sig * (1.0f + x[idx] * (1.0f - x_sig));

                float d_leaky = (x_shifted < 0.0f) ? a : 0.0f;

                float d_out = d_silu * gate + silu * d_gate + d_leaky;
                dx[idx] = dy[idx] * d_out;
            });
        });
        if (profile_events)
            profile_events->push_back(ev);
    }

  private:
    static inline float sigmoid_device(float x)
    {
        if (x >= 0.0f)
        {
            float z = s::exp(-x);
            return 1.0f / (1.0f + z);
        }
        float z = s::exp(x);
        return z / (1.0f + z);
    }

    size_t hidden_dim_ = 1;
    size_t group_size_ = 1;
    size_t group_count_ = 1;

    std::vector<float> group_offsets_host_;
    std::unique_ptr<s::buffer<float, 1>> group_offsets_dev_;

    // Used when configure_threshold_groups() isn't called; keep a valid accessor path.
    s::buffer<float, 1> fallback_group_offsets_{s::range<1>(1)};
};

} // namespace layers
} // namespace neurobit
