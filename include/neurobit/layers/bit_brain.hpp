#pragma once

#include "neurobit/components/glial.hpp"
#include "neurobit/components/hippocampus.hpp"
#include "neurobit/components/neuromodulator.hpp"
#include "neurobit/core/types.hpp"
#include "neurobit/layers/activation.hpp" // M-DSiLU
#include "neurobit/layers/bitnet.hpp"     // BitNet b1.58 (Trainable)
#include "neurobit/layers/ssm_scan.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
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
        float md_silu_tau = 1.0f;
        float md_silu_alpha = 0.02f;
        float md_silu_novelty_power = 1.0f;
        float md_silu_novelty_sharpness = 1.0f;
        size_t md_silu_threshold_group_size = 32;
        float md_silu_group_offset_lr = 0.05f;
        float md_silu_group_offset_decay = 0.999f;
        float md_silu_group_offset_max_abs = 1.0f;

        // Biomimetic dynamic adaptation (rule + learnable)
        bool enable_neuromodulator = true;
        components::NeuromodulatorConfig neuromodulator_config;
    };

    struct DeviceGlialState
    {
        float current_threshold = 0.0f;
        float current_lr = 0.0f;
        float prev_error = 0.0f;
        float error_integral = 0.0f;
        float threshold_velocity = 0.0f;
        float sparsity_ema = 0.0f;
        float error_ema = 0.0f;
        float last_change_ratio = 1.0f;
        int stable_count = 0;
        int is_first_call = 1;
        float current_noise_gain = 0.0f;
        float last_sparsity = 0.0f;
    };

    BitBrainLayer(s::queue &q, const Config &cfg)
        : cfg_(cfg), queue_(q), proj_in_(cfg.input_dim, cfg.hidden_dim, true),
          ssm_core_({cfg.batch_size, cfg.seq_len, cfg.hidden_dim}), activation_(cfg.vigilance),
          proj_mid1_(cfg.hidden_dim, cfg.hidden_dim, true), proj_mid2_(cfg.hidden_dim, cfg.hidden_dim, true),
          proj_out_(cfg.hidden_dim, cfg.output_dim, true), glial_(cfg.glial_config),
          hippocampus_({// Config for hippocampus
                        .batch_size = 1,
                        .input_dim = cfg.input_dim,
                        .hidden_dim = cfg.hidden_dim}),
          neuromodulator_(cfg.neuromodulator_config)
    {
        activation_.tau = cfg_.md_silu_tau;
        activation_.alpha = cfg_.md_silu_alpha;
        activation_.novelty_power = cfg_.md_silu_novelty_power;
        activation_.novelty_sharpness = cfg_.md_silu_novelty_sharpness;
        activation_.group_offset_lr = cfg_.md_silu_group_offset_lr;
        activation_.group_offset_decay = cfg_.md_silu_group_offset_decay;
        activation_.group_offset_max_abs = cfg_.md_silu_group_offset_max_abs;
        activation_.configure_threshold_groups(cfg_.hidden_dim, cfg_.md_silu_threshold_group_size);

        usm_context_ = queue_.get_context();
        size_t gc = activation_.group_count();
        group_count_ = gc;
        group_active_counts_ = s::malloc_shared<uint32_t>(gc, queue_);
        group_total_counts_ = s::malloc_shared<uint32_t>(gc, queue_);
        total_active_count_ = s::malloc_shared<uint32_t>(1, queue_);
        target_sparsity_ = s::malloc_shared<float>(1, queue_);
        glial_state_ = s::malloc_shared<DeviceGlialState>(1, queue_);
        glial_threshold_snapshot_ = s::malloc_shared<float>(1, queue_);
        effective_batch_size_ = s::malloc_shared<uint32_t>(1, queue_);

        for (size_t g = 0; g < gc; ++g)
        {
            group_active_counts_[g] = 0u;
            size_t feature_start = g * activation_.group_size();
            size_t feature_count = 0;
            if (feature_start < cfg_.hidden_dim)
                feature_count = std::min(activation_.group_size(), cfg_.hidden_dim - feature_start);
            group_total_counts_[g] = static_cast<uint32_t>(cfg_.batch_size * feature_count * 3);
        }
        *total_active_count_ = 0u;
        *target_sparsity_ = std::clamp(cfg_.glial_config.target_sparsity, 0.0f, 1.0f);
        *glial_threshold_snapshot_ = cfg_.glial_config.initial_threshold;
        *effective_batch_size_ = static_cast<uint32_t>(cfg_.batch_size);

        glial_state_->current_threshold = cfg_.glial_config.initial_threshold;
        glial_state_->current_lr = cfg_.glial_config.initial_lr;
        glial_state_->prev_error = 0.0f;
        glial_state_->error_integral = 0.0f;
        glial_state_->threshold_velocity = 0.0f;
        glial_state_->sparsity_ema = 0.0f;
        glial_state_->error_ema = 0.0f;
        glial_state_->last_change_ratio = 1.0f;
        glial_state_->stable_count = 0;
        glial_state_->is_first_call = 1;
        glial_state_->current_noise_gain = cfg_.glial_config.noise_gain_high;
        glial_state_->last_sparsity = 0.0f;

        // Allocate intermediate buffers for Backprop
        buf_H_proj = new s::buffer<float, 1>(s::range<1>(cfg.batch_size * cfg.hidden_dim));
        buf_H_ssm = new s::buffer<float, 1>(s::range<1>(cfg.batch_size * cfg.hidden_dim));
        buf_H_act = new s::buffer<float, 1>(s::range<1>(cfg.batch_size * cfg.hidden_dim));
        buf_ln_mean = new s::buffer<float, 1>(s::range<1>(cfg.batch_size));
        buf_ln_rstd = new s::buffer<float, 1>(s::range<1>(cfg.batch_size));
        buf_H_mid1 = new s::buffer<float, 1>(s::range<1>(cfg.batch_size * cfg.hidden_dim));
        buf_H_mid1_act = new s::buffer<float, 1>(s::range<1>(cfg.batch_size * cfg.hidden_dim));
        buf_ln_mean1 = new s::buffer<float, 1>(s::range<1>(cfg.batch_size));
        buf_ln_rstd1 = new s::buffer<float, 1>(s::range<1>(cfg.batch_size));
        buf_H_mid2 = new s::buffer<float, 1>(s::range<1>(cfg.batch_size * cfg.hidden_dim));
        buf_H_mid2_act = new s::buffer<float, 1>(s::range<1>(cfg.batch_size * cfg.hidden_dim));
        buf_ln_mean2 = new s::buffer<float, 1>(s::range<1>(cfg.batch_size));
        buf_ln_rstd2 = new s::buffer<float, 1>(s::range<1>(cfg.batch_size));

        // Gradients
        grad_H_act = new s::buffer<float, 1>(s::range<1>(cfg.batch_size * cfg.hidden_dim));
        grad_H_ssm_norm = new s::buffer<float, 1>(s::range<1>(cfg.batch_size * cfg.hidden_dim));
        grad_H_ssm = new s::buffer<float, 1>(s::range<1>(cfg.batch_size * cfg.hidden_dim));
        grad_H_proj = new s::buffer<float, 1>(s::range<1>(cfg.batch_size * cfg.hidden_dim));
        grad_H_mid1_act = new s::buffer<float, 1>(s::range<1>(cfg.batch_size * cfg.hidden_dim));
        grad_H_mid1_norm = new s::buffer<float, 1>(s::range<1>(cfg.batch_size * cfg.hidden_dim));
        grad_H_mid1 = new s::buffer<float, 1>(s::range<1>(cfg.batch_size * cfg.hidden_dim));
        grad_H_mid2_act = new s::buffer<float, 1>(s::range<1>(cfg.batch_size * cfg.hidden_dim));
        grad_H_mid2_norm = new s::buffer<float, 1>(s::range<1>(cfg.batch_size * cfg.hidden_dim));
        grad_H_mid2 = new s::buffer<float, 1>(s::range<1>(cfg.batch_size * cfg.hidden_dim));

        // SSM params (fixed/learned?)
        // For non-sequential tasks like image classification batches, we want minimal history interference.
        // Setting A to 0.0 makes S[t] = X[t], effectively bypassing the recurrent state.
        std::vector<float> h_A(cfg.hidden_dim, 0.0f);
        buf_SSM_A = new s::buffer<float, 1>(h_A.data(), s::range<1>(cfg.hidden_dim));
        buf_SSM_State = new s::buffer<float, 1>(s::range<1>(cfg.batch_size * cfg.hidden_dim));
    }

    ~BitBrainLayer()
    {
        if (group_active_counts_)
            s::free(group_active_counts_, usm_context_);
        if (group_total_counts_)
            s::free(group_total_counts_, usm_context_);
        if (total_active_count_)
            s::free(total_active_count_, usm_context_);
        if (target_sparsity_)
            s::free(target_sparsity_, usm_context_);
        if (glial_state_)
            s::free(glial_state_, usm_context_);
        if (glial_threshold_snapshot_)
            s::free(glial_threshold_snapshot_, usm_context_);
        if (effective_batch_size_)
            s::free(effective_batch_size_, usm_context_);

        delete buf_H_proj;
        delete buf_H_ssm;
        delete buf_H_act;
        delete grad_H_act;
        delete grad_H_ssm_norm;
        delete grad_H_ssm;
        delete grad_H_proj;
        delete buf_SSM_A;
        delete buf_SSM_State;
        delete buf_ln_mean;
        delete buf_ln_rstd;
        delete buf_H_mid1;
        delete buf_H_mid1_act;
        delete buf_ln_mean1;
        delete buf_ln_rstd1;
        delete buf_H_mid2;
        delete buf_H_mid2_act;
        delete buf_ln_mean2;
        delete buf_ln_rstd2;
        delete grad_H_mid1_act;
        delete grad_H_mid1_norm;
        delete grad_H_mid1;
        delete grad_H_mid2_act;
        delete grad_H_mid2_norm;
        delete grad_H_mid2;
    }

    void set_feedback(float loss, float uncertainty, float accuracy, bool training)
    {
        last_loss_ = loss;
        last_uncertainty_ = uncertainty;
        last_accuracy_ = accuracy;
        last_training_ = training;
    }

    float get_glial_threshold_host() const
    {
        return glial_state_ ? glial_state_->current_threshold : glial_.get_threshold();
    }

    float get_last_active_rate_host() const
    {
        return glial_state_ ? glial_state_->last_sparsity : last_active_rate_;
    }

    float get_target_sparsity_host() const
    {
        return target_sparsity_ ? *target_sparsity_ : glial_.get_target_sparsity();
    }

    float get_last_familiarity_host() const
    {
        return last_familiarity_host_;
    }

    struct WeightAbsStats
    {
        float proj_in = 0.0f;
        float proj_mid1 = 0.0f;
        float proj_mid2 = 0.0f;
        float proj_out = 0.0f;
        float mean_all = 0.0f;
    };

    WeightAbsStats get_weight_abs_stats_host()
    {
        WeightAbsStats s;
        s.proj_in = proj_in_.mean_abs_weight_host();
        s.proj_mid1 = proj_mid1_.mean_abs_weight_host();
        s.proj_mid2 = proj_mid2_.mean_abs_weight_host();
        s.proj_out = proj_out_.mean_abs_weight_host();
        s.mean_all = 0.25f * (s.proj_in + s.proj_mid1 + s.proj_mid2 + s.proj_out);
        return s;
    }

    struct CheckpointState
    {
        uint32_t version = 1;

        size_t batch_size = 0;
        size_t seq_len = 0;
        size_t input_dim = 0;
        size_t hidden_dim = 0;
        size_t output_dim = 0;

        DeviceGlialState glial_state{};
        float target_sparsity = 0.0f;
        float glial_threshold_snapshot = 0.0f;

        float hippocampus_learned_threshold = 0.0f;
        std::array<float, 4> neuromodulator_w = {0.0f, 0.0f, 0.0f, 0.0f};
        components::NeuromodulatorControl neuromodulator_last{};

        float md_vigilance = 0.0f;
        float md_tau = 0.0f;
        float md_alpha = 0.0f;
        float md_novelty_power = 0.0f;
        float md_novelty_sharpness = 0.0f;
        size_t md_group_size = 0;
        size_t md_group_count = 0;
        std::vector<float> md_group_offsets;

        bool proj_in_use_bias = false;
        bool proj_mid1_use_bias = false;
        bool proj_mid2_use_bias = false;
        bool proj_out_use_bias = false;

        std::vector<float> proj_in_weight;
        std::vector<float> proj_in_bias;
        std::vector<float> proj_mid1_weight;
        std::vector<float> proj_mid1_bias;
        std::vector<float> proj_mid2_weight;
        std::vector<float> proj_mid2_bias;
        std::vector<float> proj_out_weight;
        std::vector<float> proj_out_bias;

        std::vector<float> w_fast;
        std::vector<float> w_slow;

        std::vector<float> ssm_A;
        std::vector<float> ssm_state;
    };

    void export_checkpoint_host(CheckpointState &out, s::buffer<float, 1> &w_fast, s::buffer<float, 1> &w_slow)
    {
        out = CheckpointState{};
        out.batch_size = cfg_.batch_size;
        out.seq_len = cfg_.seq_len;
        out.input_dim = cfg_.input_dim;
        out.hidden_dim = cfg_.hidden_dim;
        out.output_dim = cfg_.output_dim;

        if (glial_state_)
            out.glial_state = *glial_state_;
        if (target_sparsity_)
            out.target_sparsity = *target_sparsity_;
        if (glial_threshold_snapshot_)
            out.glial_threshold_snapshot = *glial_threshold_snapshot_;

        out.hippocampus_learned_threshold = hippocampus_.get_learned_threshold();
        out.neuromodulator_w = neuromodulator_.learned_weights();
        out.neuromodulator_last = neuromodulator_.last_control();

        out.md_vigilance = activation_.vigilance;
        out.md_tau = activation_.tau;
        out.md_alpha = activation_.alpha;
        out.md_novelty_power = activation_.novelty_power;
        out.md_novelty_sharpness = activation_.novelty_sharpness;
        out.md_group_size = activation_.group_size();
        out.md_group_count = activation_.group_count();
        activation_.copy_group_offsets_host(out.md_group_offsets, true);

        auto copy_f32 = [](s::buffer<float, 1> &buf, std::vector<float> &dst) {
            size_t n = buf.get_range().size();
            dst.resize(n);
            s::host_accessor acc(buf, s::read_only);
            for (size_t i = 0; i < n; ++i)
                dst[i] = acc[i];
        };

        out.proj_in_use_bias = proj_in_.use_bias;
        out.proj_mid1_use_bias = proj_mid1_.use_bias;
        out.proj_mid2_use_bias = proj_mid2_.use_bias;
        out.proj_out_use_bias = proj_out_.use_bias;

        copy_f32(proj_in_.weight, out.proj_in_weight);
        copy_f32(proj_in_.bias, out.proj_in_bias);
        copy_f32(proj_mid1_.weight, out.proj_mid1_weight);
        copy_f32(proj_mid1_.bias, out.proj_mid1_bias);
        copy_f32(proj_mid2_.weight, out.proj_mid2_weight);
        copy_f32(proj_mid2_.bias, out.proj_mid2_bias);
        copy_f32(proj_out_.weight, out.proj_out_weight);
        copy_f32(proj_out_.bias, out.proj_out_bias);

        copy_f32(w_fast, out.w_fast);
        copy_f32(w_slow, out.w_slow);

        if (buf_SSM_A)
            copy_f32(*buf_SSM_A, out.ssm_A);
        if (buf_SSM_State)
            copy_f32(*buf_SSM_State, out.ssm_state);
    }

    void set_effective_batch_size(size_t effective_batch)
    {
        if (!effective_batch_size_)
            return;
        effective_batch = std::max<size_t>(1, std::min(effective_batch, cfg_.batch_size));
        *effective_batch_size_ = static_cast<uint32_t>(effective_batch);

        if (!group_total_counts_)
            return;
        size_t gc = group_count_;
        for (size_t g = 0; g < gc; ++g)
        {
            size_t feature_start = g * activation_.group_size();
            size_t feature_count = 0;
            if (feature_start < cfg_.hidden_dim)
                feature_count = std::min(activation_.group_size(), cfg_.hidden_dim - feature_start);
            group_total_counts_[g] = static_cast<uint32_t>(effective_batch * feature_count * 3);
        }
    }

    void sync_host_feedback()
    {
        last_active_rate_ = get_last_active_rate_host();
        if (familiarity_ptr_)
            last_familiarity_host_ = std::clamp(*familiarity_ptr_, 0.0f, 1.0f);
    }

    // Forward Pass
    void forward(s::buffer<float, 1> &input, s::buffer<float, 1> &output, s::buffer<float, 1> &w_fast,
                 s::buffer<float, 1> &w_slow, std::vector<s::event> *profile_events = nullptr)
    {
        forward_impl(input, output, w_fast, w_slow, profile_events, true);
    }

    void forward_inference(s::buffer<float, 1> &input, s::buffer<float, 1> &output, s::buffer<float, 1> &w_fast,
                           s::buffer<float, 1> &w_slow, std::vector<s::event> *profile_events = nullptr)
    {
        forward_impl(input, output, w_fast, w_slow, profile_events, false);
    }

    // Backward Pass
    void backward(s::buffer<float, 1> &grad_output, s::buffer<float, 1> &grad_input,
                  std::vector<s::event> *profile_events = nullptr)
    {
        size_t batch = cfg_.batch_size;
        if (effective_batch_size_)
            batch = std::min(batch, static_cast<size_t>(*effective_batch_size_));
        size_t hidden = cfg_.hidden_dim;

        // 1. Output Projection
        proj_out_.backward(queue_, grad_output, *grad_H_mid2_act, batch, profile_events);

        // 2. Extra Hidden 2 backward
        activation_.backward(queue_, *grad_H_mid2_act, *buf_H_mid2, *grad_H_mid2_norm, batch * hidden,
                             glial_threshold_snapshot_, familiarity_ptr_, profile_events);
        layer_norm_backward(queue_, *grad_H_mid2_norm, *buf_H_mid2, *buf_ln_rstd2, *grad_H_mid2, batch, hidden, 1e-5f,
                            profile_events);
        proj_mid2_.backward(queue_, *grad_H_mid2, *grad_H_mid1_act, batch, profile_events);

        // 3. Extra Hidden 1 backward
        activation_.backward(queue_, *grad_H_mid1_act, *buf_H_mid1, *grad_H_mid1_norm, batch * hidden,
                             glial_threshold_snapshot_, familiarity_ptr_, profile_events);
        layer_norm_backward(queue_, *grad_H_mid1_norm, *buf_H_mid1, *buf_ln_rstd1, *grad_H_mid1, batch, hidden, 1e-5f,
                            profile_events);
        proj_mid1_.backward(queue_, *grad_H_mid1, *grad_H_act, batch, profile_events);

        // 4. Activation (M-DSiLU) on SSM output
        activation_.backward(queue_, *grad_H_act, *buf_H_ssm, *grad_H_ssm_norm, batch * hidden,
                             glial_threshold_snapshot_, familiarity_ptr_, profile_events);

        // 4.5 LayerNorm Backward
        layer_norm_backward(queue_, *grad_H_ssm_norm, *buf_H_ssm, *buf_ln_rstd, *grad_H_ssm, batch, hidden, 1e-5f,
                            profile_events);

        // 5. SSM (Simplified Backward: Pass-through)
        auto ev_copy = queue_.submit([&](s::handler &h) {
            s::accessor in(*grad_H_ssm, h, s::read_only);
            s::accessor out(*grad_H_proj, h, s::write_only, s::no_init);
            h.copy(in, out);
        });
        if (profile_events)
            profile_events->push_back(ev_copy);

        // 6. Input Projection
        proj_in_.backward(queue_, *grad_H_proj, grad_input, batch, profile_events);
    }

    // Update
    void step(float lr, float weight_decay = 0.0f, std::vector<s::event> *profile_events = nullptr)
    {
        proj_in_.step(queue_, lr, weight_decay, profile_events);
        proj_mid1_.step(queue_, lr, weight_decay, profile_events);
        proj_mid2_.step(queue_, lr, weight_decay, profile_events);
        proj_out_.step(queue_, lr, weight_decay, profile_events);

        if (last_training_)
        {
            activation_.homeostatic_update_groups_device(queue_, group_active_counts_, group_total_counts_,
                                                         target_sparsity_, profile_events);
        }
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
    void forward_impl(s::buffer<float, 1> &input, s::buffer<float, 1> &output, s::buffer<float, 1> &w_fast,
                      s::buffer<float, 1> &w_slow, std::vector<s::event> *profile_events, bool training)
    {
        size_t batch = cfg_.batch_size;
        if (effective_batch_size_)
            batch = std::min(batch, static_cast<size_t>(*effective_batch_size_));
        size_t hidden = cfg_.hidden_dim;
        size_t total_neurons = batch * hidden * 3;

        // Reset SSM state for non-sequential tasks (e.g., image classification batches)
        auto ev_reset = queue_.submit([&](s::handler &h) {
            s::accessor acc(*buf_SSM_State, h, s::write_only, s::no_init);
            h.parallel_for(s::range<1>(batch * hidden), [=](s::id<1> idx) { acc[idx] = 0.0f; });
        });
        if (profile_events)
            profile_events->push_back(ev_reset);

        // 1. Hippocampus: enqueue familiarity (device-side scalar; read on device by activation)
        familiarity_ptr_ = hippocampus_.enqueue_familiarity(queue_, input, w_fast, w_slow, profile_events);

        // 1.1 Stage device-side scalars for this iteration (snapshot threshold + reset activity counters).
        float staged_target = std::clamp(glial_.get_target_sparsity(), 0.0f, 1.0f);
        if (training && cfg_.enable_neuromodulator)
        {
            float delayed_fam = last_familiarity_host_;
            auto control = neuromodulator_.update(components::NeuromodulatorObservation{
                .active_rate = last_active_rate_,
                .familiarity = delayed_fam,
                .uncertainty = last_uncertainty_,
                .loss = last_loss_,
                .accuracy = last_accuracy_,
                .training = last_training_,
            });
            staged_target = control.target_sparsity;
            glial_.set_target_sparsity(staged_target);
            activation_.vigilance = control.vigilance;
            activation_.tau = control.md_silu_tau;
            activation_.alpha = control.md_silu_alpha;
            activation_.novelty_power = control.md_silu_novelty_power;
            activation_.novelty_sharpness = control.md_silu_novelty_sharpness;
        }
        staged_target = std::clamp(staged_target, 0.0f, 1.0f);

        auto ev_init = queue_.submit([&](s::handler &h) {
            DeviceGlialState *st = glial_state_;
            float *th_snap = glial_threshold_snapshot_;
            float *target = target_sparsity_;
            uint32_t *total_active = total_active_count_;
            uint32_t *group_active = group_active_counts_;
            size_t gc = group_count_;
            h.single_task([=]() {
                *target = staged_target;
                *th_snap = st->current_threshold;
                *total_active = 0u;
                for (size_t g = 0; g < gc; ++g)
                    group_active[g] = 0u;
            });
        });
        if (profile_events)
            profile_events->push_back(ev_init);

        // 3. Input Projection (BitLinear)
        proj_in_.forward(queue_, input, *buf_H_proj, batch, profile_events);

        // 4. SSM Scan (Temporal)
        ssm_core_.forward_single(queue_, *buf_H_proj, *buf_SSM_A, *buf_H_ssm, *buf_SSM_State, profile_events);

        // 3.5 LayerNorm on SSM output to stabilize signal for Glial/Activation
        layer_norm(queue_, *buf_H_ssm, *buf_ln_mean, *buf_ln_rstd, batch, hidden, 1e-5f, profile_events);

        // 5. Activation (M-DSiLU)
        activation_.forward(queue_, *buf_H_ssm, *buf_H_act, batch * hidden, glial_threshold_snapshot_, familiarity_ptr_,
                            group_active_counts_, total_active_count_, profile_events);

        // 5.1 Extra Hidden 1 (BitLinear + LayerNorm + M-DSiLU)
        proj_mid1_.forward(queue_, *buf_H_act, *buf_H_mid1, batch, profile_events);
        layer_norm(queue_, *buf_H_mid1, *buf_ln_mean1, *buf_ln_rstd1, batch, hidden, 1e-5f, profile_events);
        activation_.forward(queue_, *buf_H_mid1, *buf_H_mid1_act, batch * hidden, glial_threshold_snapshot_,
                            familiarity_ptr_, group_active_counts_, total_active_count_, profile_events);

        // 5.2 Extra Hidden 2 (BitLinear + LayerNorm + M-DSiLU)
        proj_mid2_.forward(queue_, *buf_H_mid1_act, *buf_H_mid2, batch, profile_events);
        layer_norm(queue_, *buf_H_mid2, *buf_ln_mean2, *buf_ln_rstd2, batch, hidden, 1e-5f, profile_events);
        activation_.forward(queue_, *buf_H_mid2, *buf_H_mid2_act, batch * hidden, glial_threshold_snapshot_,
                            familiarity_ptr_, group_active_counts_, total_active_count_, profile_events);

        // 6. Output Projection (BitLinear)
        proj_out_.forward(queue_, *buf_H_mid2_act, output, batch, profile_events);

        if (training)
        {
            // 7. Device-side Glial threshold update (for next batch)
            auto cfg = cfg_.glial_config;
            auto ev_glial = queue_.submit([&](s::handler &h) {
                DeviceGlialState *st = glial_state_;
                uint32_t *total_active = total_active_count_;
                float *target = target_sparsity_;
                h.single_task([=]() {
                    float current_sparsity = static_cast<float>(*total_active) / static_cast<float>(total_neurons);
                    if (st->is_first_call)
                    {
                        st->sparsity_ema = current_sparsity;
                        st->error_ema = current_sparsity - *target;
                    }
                    else
                    {
                        constexpr float kEmaAlpha = 0.05f;
                        st->sparsity_ema = (1.0f - kEmaAlpha) * st->sparsity_ema + kEmaAlpha * current_sparsity;
                        float instant_error = st->sparsity_ema - *target;
                        st->error_ema = (1.0f - kEmaAlpha) * st->error_ema + kEmaAlpha * instant_error;
                    }

                    float error = st->error_ema;
                    float abs_error = s::fabs(error);
                    float prev_abs_error = s::fabs(st->prev_error);
                    float error_delta = error - st->prev_error;

                    if (cfg.adaptive_lr && !st->is_first_call)
                    {
                        if (error * st->prev_error < 0.0f)
                        {
                            st->current_lr *= cfg.lr_shrink;
                        }
                        else if (abs_error > prev_abs_error + cfg.stability_tolerance)
                        {
                            st->current_lr *= cfg.lr_shrink;
                        }
                        else if (abs_error > cfg.stability_tolerance)
                        {
                            st->current_lr *= cfg.lr_growth;
                        }
                        st->current_lr = s::clamp(st->current_lr, cfg.min_lr, cfg.max_lr);
                    }

                    st->is_first_call = 0;

                    constexpr float kIntegralLeak = 0.995f;
                    constexpr float kIntegralClamp = 2.0f;
                    st->error_integral =
                        s::clamp(st->error_integral * kIntegralLeak + error, -kIntegralClamp, kIntegralClamp);

                    constexpr float kErrorScale = 0.15f;
                    float error_sat = s::tanh(error / kErrorScale);

                    constexpr float kKp = 1.0f;
                    constexpr float kKi = 0.05f;
                    constexpr float kKd = 0.1f;

                    float delta = st->current_lr * (kKp * error_sat + kKi * st->error_integral) - kKd * error_delta;

                    constexpr float kMomentum = 0.9f;
                    st->threshold_velocity = kMomentum * st->threshold_velocity + (1.0f - kMomentum) * delta;

                    float old_th = st->current_threshold;
                    st->current_threshold += st->threshold_velocity;
                    st->current_threshold = s::clamp(st->current_threshold, cfg.min_threshold, cfg.max_threshold);

                    float denom = s::fmax(s::fabs(old_th), 1e-6f);
                    st->last_change_ratio = s::fabs(st->current_threshold - old_th) / denom;

                    if (st->last_change_ratio < cfg.stability_tolerance && abs_error < cfg.stability_tolerance)
                        st->stable_count++;
                    else
                        st->stable_count = 0;

                    float target_eps = s::fmax(*target, 1e-6f);
                    if (abs_error > target_eps)
                        st->current_noise_gain = cfg.noise_gain_high;
                    else
                    {
                        float t = abs_error / target_eps;
                        st->current_noise_gain = cfg.noise_gain_low + t * (cfg.noise_gain_high - cfg.noise_gain_low);
                    }

                    st->prev_error = error;
                    st->last_sparsity = st->sparsity_ema;
                });
            });
            if (profile_events)
                profile_events->push_back(ev_glial);
        }
    }

    Config cfg_;
    s::queue &queue_;

    // Sub-modules
    BitLinear proj_in_;
    SSMScan ssm_core_;
    MDSiLU activation_;
    BitLinear proj_mid1_;
    BitLinear proj_mid2_;
    BitLinear proj_out_;

    components::GlialCell glial_;
    components::Hippocampus hippocampus_;
    components::Neuromodulator neuromodulator_;

    // Buffers
    s::buffer<float, 1> *buf_H_proj;
    s::buffer<float, 1> *buf_H_ssm;
    s::buffer<float, 1> *buf_H_act; // Output of Activation (Input to L2)
    s::buffer<float, 1> *buf_ln_mean;
    s::buffer<float, 1> *buf_ln_rstd;
    s::buffer<float, 1> *buf_H_mid1;
    s::buffer<float, 1> *buf_H_mid1_act;
    s::buffer<float, 1> *buf_ln_mean1;
    s::buffer<float, 1> *buf_ln_rstd1;
    s::buffer<float, 1> *buf_H_mid2;
    s::buffer<float, 1> *buf_H_mid2_act;
    s::buffer<float, 1> *buf_ln_mean2;
    s::buffer<float, 1> *buf_ln_rstd2;

    s::buffer<float, 1> *buf_SSM_A;
    s::buffer<float, 1> *buf_SSM_State;

    // Gradients
    s::buffer<float, 1> *grad_H_act;
    s::buffer<float, 1> *grad_H_ssm_norm;
    s::buffer<float, 1> *grad_H_ssm;
    s::buffer<float, 1> *grad_H_proj;
    s::buffer<float, 1> *grad_H_mid1_act;
    s::buffer<float, 1> *grad_H_mid1_norm;
    s::buffer<float, 1> *grad_H_mid1;
    s::buffer<float, 1> *grad_H_mid2_act;
    s::buffer<float, 1> *grad_H_mid2_norm;
    s::buffer<float, 1> *grad_H_mid2;

    // Context (host-side feedback; typically refreshed via sync_host_feedback() after queue sync)
    float last_active_rate_ = 0.0f;
    float last_familiarity_host_ = 0.0f;
    float last_uncertainty_ = 0.0f;
    float last_loss_ = 0.0f;
    float last_accuracy_ = 0.0f;
    bool last_training_ = false;

    // Device-side counters/state (USM shared)
    s::context usm_context_{};
    size_t group_count_ = 0;
    uint32_t *group_active_counts_ = nullptr;
    uint32_t *group_total_counts_ = nullptr;
    uint32_t *total_active_count_ = nullptr;
    float *target_sparsity_ = nullptr;
    DeviceGlialState *glial_state_ = nullptr;
    float *glial_threshold_snapshot_ = nullptr;
    uint32_t *effective_batch_size_ = nullptr;
    const float *familiarity_ptr_ = nullptr;

    // Private helper for LayerNorm (declarations)
    void layer_norm(s::queue &q, s::buffer<float, 1> &buf, s::buffer<float, 1> &mean, s::buffer<float, 1> &rstd,
                    size_t batch_size_arg, size_t dim_arg, float epsilon,
                    std::vector<s::event> *profile_events = nullptr);
    void layer_norm_backward(s::queue &q, s::buffer<float, 1> &grad_output, s::buffer<float, 1> &x_hat,
                             s::buffer<float, 1> &rstd, s::buffer<float, 1> &grad_input, size_t batch_size_arg,
                             size_t dim_arg, float epsilon, std::vector<s::event> *profile_events = nullptr);
};

// Definitions of LayerNorm member functions outside the class definition
void BitBrainLayer::layer_norm(s::queue &q, s::buffer<float, 1> &buf, s::buffer<float, 1> &mean,
                               s::buffer<float, 1> &rstd, size_t batch_size_arg, size_t dim_arg, float epsilon,
                               std::vector<s::event> *profile_events)
{
    auto ev0 = q.submit([&](s::handler &h) {
        s::accessor acc(buf, h, s::read_write);
        s::accessor m(mean, h, s::write_only, s::no_init);
        s::accessor rs(rstd, h, s::write_only, s::no_init);
        constexpr size_t kLocal = 256;
        s::local_accessor<float, 1> local_sum{s::range<1>(kLocal), h};
        s::local_accessor<float, 1> local_sq{s::range<1>(kLocal), h};

        h.parallel_for(s::nd_range<2>{s::range<2>{batch_size_arg, kLocal}, s::range<2>{1, kLocal}},
                       [=](s::nd_item<2> it) {
                           size_t b = it.get_global_id(0);
                           size_t lid = it.get_local_id(1);

                           float sum = 0.0f;
                           float sum_sq = 0.0f;
                           for (size_t i = lid; i < dim_arg; i += kLocal)
                           {
                               float v = acc[b * dim_arg + i];
                               sum += v;
                               sum_sq += v * v;
                           }

                           local_sum[lid] = sum;
                           local_sq[lid] = sum_sq;
                           it.barrier(s::access::fence_space::local_space);

                           for (size_t stride = kLocal / 2; stride > 0; stride >>= 1)
                           {
                               if (lid < stride)
                               {
                                   local_sum[lid] += local_sum[lid + stride];
                                   local_sq[lid] += local_sq[lid + stride];
                               }
                               it.barrier(s::access::fence_space::local_space);
                           }

                           float mean_v = local_sum[0] / static_cast<float>(dim_arg);
                           float var = local_sq[0] / static_cast<float>(dim_arg) - mean_v * mean_v;
                           float inv_std = s::rsqrt(var + epsilon);

                           if (lid == 0)
                           {
                               m[b] = mean_v;
                               rs[b] = inv_std;
                           }

                           for (size_t i = lid; i < dim_arg; i += kLocal)
                           {
                               acc[b * dim_arg + i] = (acc[b * dim_arg + i] - mean_v) * inv_std;
                           }
                       });
    });
    if (profile_events)
        profile_events->push_back(ev0);
}

void BitBrainLayer::layer_norm_backward(s::queue &q, s::buffer<float, 1> &grad_output, s::buffer<float, 1> &x_hat,
                                        s::buffer<float, 1> &rstd, s::buffer<float, 1> &grad_input,
                                        size_t batch_size_arg, size_t dim_arg, float epsilon,
                                        std::vector<s::event> *profile_events)
{
    auto ev0 = q.submit([&](s::handler &h) {
        s::accessor d_out(grad_output, h, s::read_only);
        s::accessor x(x_hat, h, s::read_only);
        s::accessor rs(rstd, h, s::read_only);
        s::accessor d_in(grad_input, h, s::write_only, s::no_init);
        constexpr size_t kLocal = 256;
        s::local_accessor<float, 1> local_sum{s::range<1>(kLocal), h};
        s::local_accessor<float, 1> local_sum_x{s::range<1>(kLocal), h};

        h.parallel_for(s::nd_range<2>{s::range<2>{batch_size_arg, kLocal}, s::range<2>{1, kLocal}},
                       [=](s::nd_item<2> it) {
                           size_t b = it.get_global_id(0);
                           size_t lid = it.get_local_id(1);
                           (void)epsilon;

                           float inv_std = rs[b];
                           float sum_dy = 0.0f;
                           float sum_dy_xhat = 0.0f;
                           for (size_t i = lid; i < dim_arg; i += kLocal)
                           {
                               float dy = d_out[b * dim_arg + i];
                               float xh = x[b * dim_arg + i];
                               sum_dy += dy;
                               sum_dy_xhat += dy * xh;
                           }

                           local_sum[lid] = sum_dy;
                           local_sum_x[lid] = sum_dy_xhat;
                           it.barrier(s::access::fence_space::local_space);

                           for (size_t stride = kLocal / 2; stride > 0; stride >>= 1)
                           {
                               if (lid < stride)
                               {
                                   local_sum[lid] += local_sum[lid + stride];
                                   local_sum_x[lid] += local_sum_x[lid + stride];
                               }
                               it.barrier(s::access::fence_space::local_space);
                           }

                           float mean_dy = local_sum[0] / static_cast<float>(dim_arg);
                           float mean_dy_xhat = local_sum_x[0] / static_cast<float>(dim_arg);

                           for (size_t i = lid; i < dim_arg; i += kLocal)
                           {
                               float dy = d_out[b * dim_arg + i];
                               float xh = x[b * dim_arg + i];
                               d_in[b * dim_arg + i] = inv_std * (dy - mean_dy - xh * mean_dy_xhat);
                           }
                       });
    });
    if (profile_events)
        profile_events->push_back(ev0);
}
} // namespace layers
} // namespace neurobit
