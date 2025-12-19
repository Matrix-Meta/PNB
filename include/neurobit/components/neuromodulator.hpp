/*
 * Copyright 2025 Project Neuro-Bit Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace neurobit
{
namespace components
{

struct NeuromodulatorConfig
{
    bool enable_rule = true;
    bool enable_learned = true;
    bool freeze_inference = true;

    float active_min = 0.05f;
    float active_max = 0.20f;

    float base_target = 0.15f;
    float novelty_gain = 0.05f;     // (1 - familiarity) -> allow more activity
    float uncertainty_gain = 0.03f; // uncertainty -> allow more activity

    float base_vigilance = 0.4f;
    float novelty_vigilance_gain = 0.05f; // novelty -> more suppression (higher effective threshold shift)

    float noise_gain_low = 0.0f;
    float noise_gain_high = 1.0f;

    float learned_delta_max = 0.03f;
    float learned_lr = 0.01f;
    float learned_kp = 0.2f; // desired_delta ~= kp*(active_mid - active)

    // M-DSiLU dynamic parameters
    float md_silu_tau_base = 1.0f;
    float md_silu_tau_min = 0.2f;
    float md_silu_tau_max = 3.0f;
    float md_silu_tau_active_gain = 2.0f;      // (active_mid - active) -> tau
    float md_silu_tau_uncertainty_gain = 0.5f; // uncertainty -> tau

    float md_silu_alpha_base = 0.02f;
    float md_silu_alpha_min = 0.0f;
    float md_silu_alpha_max = 0.2f;
    float md_silu_alpha_uncertainty_gain = 0.08f; // uncertainty -> leaky tail

    float md_silu_novelty_power_base = 1.0f;
    float md_silu_novelty_power_min = 0.5f;
    float md_silu_novelty_power_max = 3.0f;
    float md_silu_novelty_power_uncertainty_gain = 1.0f; // uncertainty -> p

    float md_silu_novelty_sharpness_base = 1.0f;
    float md_silu_novelty_sharpness_min = 0.0f;
    float md_silu_novelty_sharpness_max = 8.0f;
    float md_silu_novelty_sharpness_uncertainty_gain = 0.8f; // uncertainty -> lower sharpness
};

struct NeuromodulatorObservation
{
    float active_rate = 0.0f; // [0,1]
    float familiarity = 0.0f; // [0,1]
    float uncertainty = 0.0f; // [0,1] (e.g., normalized entropy)
    float loss = 0.0f;        // optional
    float accuracy = 0.0f;    // optional
    bool training = false;
};

struct NeuromodulatorControl
{
    float target_sparsity = 0.15f;
    float vigilance = 0.4f;
    float noise_gain = 0.0f;

    float md_silu_tau = 1.0f;
    float md_silu_alpha = 0.02f;
    float md_silu_novelty_power = 1.0f;
    float md_silu_novelty_sharpness = 1.0f;
};

class Neuromodulator
{
  public:
    explicit Neuromodulator(const NeuromodulatorConfig &cfg) : cfg_(cfg)
    {
    }

    std::array<float, 4> learned_weights() const
    {
        return w_;
    }

    void set_learned_weights(const std::array<float, 4> &w)
    {
        w_ = w;
    }

    NeuromodulatorControl update(const NeuromodulatorObservation &obs)
    {
        NeuromodulatorObservation o = obs;
        o.active_rate = std::clamp(o.active_rate, 0.0f, 1.0f);
        o.familiarity = std::clamp(o.familiarity, 0.0f, 1.0f);
        o.uncertainty = std::clamp(o.uncertainty, 0.0f, 1.0f);

        float novelty = 1.0f - o.familiarity;
        float active_mid = 0.5f * (cfg_.active_min + cfg_.active_max);

        float target = cfg_.base_target;
        if (cfg_.enable_rule)
        {
            target += cfg_.novelty_gain * novelty;
            target += cfg_.uncertainty_gain * o.uncertainty;
        }

        float delta = 0.0f;
        if (cfg_.enable_learned)
        {
            std::array<float, 4> features = {1.0f, novelty, o.uncertainty, (active_mid - o.active_rate)};
            float dot = 0.0f;
            for (size_t i = 0; i < features.size(); ++i)
                dot += w_[i] * features[i];
            delta = std::tanh(dot) * cfg_.learned_delta_max;

            bool can_learn = o.training || !cfg_.freeze_inference;
            if (can_learn)
            {
                float desired_delta = std::clamp(cfg_.learned_kp * (active_mid - o.active_rate),
                                                 -cfg_.learned_delta_max, cfg_.learned_delta_max);
                float err = desired_delta - delta;
                for (size_t i = 0; i < features.size(); ++i)
                    w_[i] += cfg_.learned_lr * err * features[i];
            }
        }

        target += delta;
        target = std::clamp(target, cfg_.active_min, cfg_.active_max);

        float vigilance = cfg_.base_vigilance;
        vigilance += cfg_.novelty_vigilance_gain * novelty;
        vigilance = std::clamp(vigilance, 0.0f, 2.0f);

        float noise_gain = cfg_.noise_gain_low + o.uncertainty * (cfg_.noise_gain_high - cfg_.noise_gain_low);
        noise_gain = std::clamp(noise_gain, 0.0f, 10.0f);

        float md_tau = cfg_.md_silu_tau_base;
        if (cfg_.enable_rule)
        {
            md_tau += cfg_.md_silu_tau_active_gain * (active_mid - o.active_rate);
            md_tau += cfg_.md_silu_tau_uncertainty_gain * o.uncertainty;
        }
        md_tau = std::clamp(md_tau, cfg_.md_silu_tau_min, cfg_.md_silu_tau_max);

        float md_alpha = cfg_.md_silu_alpha_base;
        if (cfg_.enable_rule)
            md_alpha += cfg_.md_silu_alpha_uncertainty_gain * o.uncertainty;
        md_alpha = std::clamp(md_alpha, cfg_.md_silu_alpha_min, cfg_.md_silu_alpha_max);

        float md_p = cfg_.md_silu_novelty_power_base;
        if (cfg_.enable_rule)
            md_p += cfg_.md_silu_novelty_power_uncertainty_gain * o.uncertainty;
        md_p = std::clamp(md_p, cfg_.md_silu_novelty_power_min, cfg_.md_silu_novelty_power_max);

        float md_sharp = cfg_.md_silu_novelty_sharpness_base;
        if (cfg_.enable_rule)
            md_sharp -= cfg_.md_silu_novelty_sharpness_uncertainty_gain * o.uncertainty;
        md_sharp = std::clamp(md_sharp, cfg_.md_silu_novelty_sharpness_min, cfg_.md_silu_novelty_sharpness_max);

        last_control_ = NeuromodulatorControl{.target_sparsity = target,
                                              .vigilance = vigilance,
                                              .noise_gain = noise_gain,
                                              .md_silu_tau = md_tau,
                                              .md_silu_alpha = md_alpha,
                                              .md_silu_novelty_power = md_p,
                                              .md_silu_novelty_sharpness = md_sharp};
        return last_control_;
    }

    const NeuromodulatorControl &last_control() const
    {
        return last_control_;
    }

    void set_last_control(const NeuromodulatorControl &c)
    {
        last_control_ = c;
    }

    const NeuromodulatorConfig &config() const
    {
        return cfg_;
    }

  private:
    NeuromodulatorConfig cfg_;
    std::array<float, 4> w_ = {0.0f, 0.0f, 0.0f, 0.0f};
    NeuromodulatorControl last_control_{};
};

} // namespace components
} // namespace neurobit
