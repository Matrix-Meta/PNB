#pragma once
#include <algorithm>
#include <cmath>

namespace neurobit
{
namespace components
{

struct GlialConfig
{
    float initial_threshold = 1.0f;
    float target_sparsity = 0.1f;
    float min_threshold = 0.1f;
    float max_threshold = 10.0f;

    // 自適應學習率
    bool adaptive_lr = true;
    float initial_lr = 1.0f;
    float max_lr = 1000.0f;
    float min_lr = 0.01f;
    float lr_growth = 1.2f;
    float lr_shrink = 0.5f;

    // 思考終止條件
    float stability_tolerance = 0.01f; // 閾值變化小於 1% 視為穩定
    int min_stable_steps = 2;          // 連續穩定步數才算真正穩定

    // 噪聲控制
    float noise_gain_high = 1.0f; // 高不確定度時的噪聲
    float noise_gain_low = 0.2f;  // 低不確定度時的噪聲
};

class GlialCell
{
  public:
    GlialCell(const GlialConfig &cfg)
        : cfg_(cfg), current_threshold_(cfg.initial_threshold), current_lr_(cfg.initial_lr), prev_error_(0.0f),
          last_change_ratio_(1.0f), stable_count_(0), is_first_call_(true), current_noise_gain_(cfg.noise_gain_high)
    {
    }

    // 核心調節函數
    float regulate(int total_spikes, size_t total_neurons)
    {
        if (total_neurons == 0)
            return current_threshold_;

        // Biomimetic homeostasis:
        // - Use an EMA firing-rate (astrocyte-like smoothing) to avoid jitter.
        // - Use a PI-like controller with anti-windup and momentum for stable threshold dynamics.
        // - Adapt the learning rate based on error dynamics (overshoot / stagnation).
        float current_sparsity = static_cast<float>(total_spikes) / static_cast<float>(total_neurons);
        if (is_first_call_)
        {
            sparsity_ema_ = current_sparsity;
            error_ema_ = current_sparsity - cfg_.target_sparsity;
        }
        else
        {
            constexpr float kEmaAlpha = 0.05f;
            sparsity_ema_ = (1.0f - kEmaAlpha) * sparsity_ema_ + kEmaAlpha * current_sparsity;
            float instant_error = sparsity_ema_ - cfg_.target_sparsity;
            error_ema_ = (1.0f - kEmaAlpha) * error_ema_ + kEmaAlpha * instant_error;
        }

        float error = error_ema_;
        float abs_error = std::abs(error);
        float prev_abs_error = std::abs(prev_error_);
        float error_delta = error - prev_error_;

        // Meta-Regulation (Adaptive LR)
        if (cfg_.adaptive_lr && !is_first_call_)
        {
            // Overshoot (sign flip) -> shrink; sustained error -> grow (up to clamp).
            if (error * prev_error_ < 0.0f)
            {
                current_lr_ *= cfg_.lr_shrink;
            }
            else if (abs_error > prev_abs_error + cfg_.stability_tolerance)
            {
                current_lr_ *= cfg_.lr_shrink;
            }
            else if (abs_error > cfg_.stability_tolerance)
            {
                current_lr_ *= cfg_.lr_growth;
            }
            current_lr_ = std::clamp(current_lr_, cfg_.min_lr, cfg_.max_lr);
        }

        // 首次調用後標記為非首次
        is_first_call_ = false;

        // PI controller with anti-windup (clamped integral) and a mild damping term.
        constexpr float kIntegralLeak = 0.995f;
        constexpr float kIntegralClamp = 2.0f;
        error_integral_ = std::clamp(error_integral_ * kIntegralLeak + error, -kIntegralClamp, kIntegralClamp);

        // Saturate proportional response to avoid large jumps when error is big.
        constexpr float kErrorScale = 0.15f;
        float error_sat = std::tanh(error / kErrorScale);

        constexpr float kKp = 1.0f;
        constexpr float kKi = 0.05f;
        constexpr float kKd = 0.1f;

        float delta = current_lr_ * (kKp * error_sat + kKi * error_integral_) - kKd * error_delta;

        // Momentum-like smoothing (glial calcium dynamics proxy)
        constexpr float kMomentum = 0.9f;
        threshold_velocity_ = kMomentum * threshold_velocity_ + (1.0f - kMomentum) * delta;

        float old_th = current_threshold_;
        current_threshold_ += threshold_velocity_;

        // 限制閾值範圍
        current_threshold_ = std::clamp(current_threshold_, cfg_.min_threshold, cfg_.max_threshold);

        // 計算變化率 (作為穩定度指標)
        float denom = std::max(std::abs(old_th), 1e-6f);
        if (denom > 0.0f)
        {
            last_change_ratio_ = std::abs(current_threshold_ - old_th) / denom;
        }
        else
        {
            // 防止除零: 使用絕對變化量
            last_change_ratio_ = std::abs(current_threshold_ - old_th);
        }

        // 更新穩定計數
        if (last_change_ratio_ < cfg_.stability_tolerance && abs_error < cfg_.stability_tolerance)
        {
            stable_count_++;
        }
        else
        {
            stable_count_ = 0;
        }

        // 根據誤差調整噪聲
        if (abs_error > cfg_.target_sparsity)
        {
            current_noise_gain_ = cfg_.noise_gain_high;
        }
        else
        {
            // 線性插值
            float t = abs_error / cfg_.target_sparsity;
            current_noise_gain_ = cfg_.noise_gain_low + t * (cfg_.noise_gain_high - cfg_.noise_gain_low);
        }

        prev_error_ = error;
        last_sparsity_ = sparsity_ema_;
        return current_threshold_;
    }

    // 判斷是否已經「想通了」(達到穩態)
    bool is_stable() const
    {
        return stable_count_ >= cfg_.min_stable_steps;
    }

    // 獲取當前閾值
    float get_threshold() const
    {
        return current_threshold_;
    }
    void set_threshold(float v)
    {
        current_threshold_ = std::clamp(v, cfg_.min_threshold, cfg_.max_threshold);
    }

    // 獲取當前噪聲增益
    float get_noise_gain() const
    {
        return current_noise_gain_;
    }

    // 獲取當前學習率
    float get_learning_rate() const
    {
        return current_lr_;
    }

    // 獲取上次稀疏度
    float get_last_sparsity() const
    {
        return last_sparsity_;
    }

    // 更新稀疏度 (用於 USM 優化版本)
    void update_sparsity(float sparsity)
    {
        last_sparsity_ = sparsity;
        float error = sparsity - cfg_.target_sparsity;

        // 更新穩定計數
        if (std::abs(error) < cfg_.stability_tolerance)
        {
            stable_count_++;
        }
        else
        {
            stable_count_ = 0;
        }

        // 更新噪聲增益
        float abs_error = std::abs(error);
        if (abs_error > cfg_.target_sparsity)
        {
            current_noise_gain_ = cfg_.noise_gain_high;
        }
        else
        {
            float t = abs_error / cfg_.target_sparsity;
            current_noise_gain_ = cfg_.noise_gain_low + t * (cfg_.noise_gain_high - cfg_.noise_gain_low);
        }
    }

    // 獲取穩定計數
    int get_stable_count() const
    {
        return stable_count_;
    }

    // 重置狀態 (用於新序列)
    void reset()
    {
        current_threshold_ = cfg_.initial_threshold;
        current_lr_ = cfg_.initial_lr;
        prev_error_ = 0.0f;
        sparsity_ema_ = 0.0f;
        error_ema_ = 0.0f;
        error_integral_ = 0.0f;
        threshold_velocity_ = 0.0f;
        last_change_ratio_ = 1.0f;
        stable_count_ = 0;
        is_first_call_ = true;
        current_noise_gain_ = cfg_.noise_gain_high;
        last_sparsity_ = 0.0f;
    }

    // 獲取配置
    const GlialConfig &config() const
    {
        return cfg_;
    }

  private:
    GlialConfig cfg_;
    float current_threshold_;
    float current_lr_;
    float prev_error_;
    float sparsity_ema_ = 0.0f;
    float error_ema_ = 0.0f;
    float error_integral_ = 0.0f;
    float threshold_velocity_ = 0.0f;
    float last_change_ratio_;
    int stable_count_;
    bool is_first_call_;
    float current_noise_gain_;
    float last_sparsity_ = 0.0f;
};

} // namespace components
} // namespace neurobit
