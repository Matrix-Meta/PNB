#pragma once
#include "neurobit/core/types.hpp"
#include <cmath>
#include <sycl/sycl.hpp>
#include <vector>

namespace neurobit
{
namespace components
{
/**
 * Hippocampus - 重新設計的記憶系統
 *
 * 核心改進:
 * 1. 熟悉度評分 (Familiarity Score) - 計算輸入與記憶的相似度
 * 2. 閾值預設 (Threshold Priming) - 根據熟悉度預設合適閾值
 * 3. 選擇性注入 (Selective Injection) - 只增強匹配的神經元
 * 4. 能量正規化 (Energy Normalization) - 保持總能量穩定
 *
 * 生物學對應:
 * - 快速權重 (W_fast): 海馬體短期記憶
 * - 慢速權重 (W_slow): 新皮層長期記憶
 * - 熟悉度: 前額葉評估信號
 * - 閾值預設: 丘腦調節
 */

class Hippocampus
{
  public:
    struct Config
    {
        size_t batch_size = 1;
        size_t input_dim;
        size_t hidden_dim;

        // 學習參數
        float learning_rate = 0.1f;
        float consolidation_rate = 0.1f;
        float decay_rate = 0.02f;
        float weight_clip = 1.0f;

        // 熟悉度參數
        float familiarity_threshold = 0.3f; // 熟悉度閾值
        float priming_strength = 0.5f;      // 閾值預設強度

        // 注入參數
        float injection_scale = 0.1f;    // 記憶注入縮放
        bool normalize_injection = true; // 是否正規化注入
    };

    struct RecallResult
    {
        float familiarity_score;   // [0, 1] 熟悉度
        float suggested_threshold; // 建議的初始閾值
        bool is_familiar;          // 是否被識別為熟悉
    };

    Hippocampus(const Config &cfg) : cfg_(cfg), last_familiarity_(0.0f), learned_threshold_(1.0f)
    {
    }

    ~Hippocampus()
    {
        if (!usm_initialized_)
            return;
        s::free(d_dot_product_, usm_context_);
        s::free(d_input_norm_, usm_context_);
        s::free(d_memory_norm_, usm_context_);
        s::free(d_familiarity_, usm_context_);
        usm_initialized_ = false;
        d_dot_product_ = nullptr;
        d_input_norm_ = nullptr;
        d_memory_norm_ = nullptr;
        d_familiarity_ = nullptr;
    }

    const float *enqueue_familiarity(s::queue &q, s::buffer<float, 1> &buf_X, s::buffer<float, 1> &buf_W_fast,
                                     s::buffer<float, 1> &buf_W_slow, std::vector<s::event> *profile_events = nullptr)
    {
        ensure_usm(q);

        const size_t M = cfg_.input_dim;
        const size_t N = cfg_.hidden_dim;
        float *d_dot_product = d_dot_product_;
        float *d_input_norm = d_input_norm_;
        float *d_memory_norm = d_memory_norm_;
        float *d_familiarity = d_familiarity_;

        auto ev0 = q.submit([&](s::handler &h) {
            h.single_task([=]() {
                *d_dot_product = 0.0f;
                *d_input_norm = 0.0f;
                *d_memory_norm = 0.0f;
            });
        });
        if (profile_events)
            profile_events->push_back(ev0);

        auto ev1 = q.submit([&](s::handler &h) {
            s::accessor acc_X{buf_X, h, s::read_only};
            s::accessor acc_Wf{buf_W_fast, h, s::read_only};
            s::accessor acc_Ws{buf_W_slow, h, s::read_only};

            h.parallel_for(s::range<1>{M}, [=](s::id<1> idx) {
                size_t i = idx[0];
                float x = acc_X[i];

                s::atomic_ref<float, s::memory_order::relaxed, s::memory_scope::device,
                              s::access::address_space::global_space>
                    atomic_input_norm(*d_input_norm);
                atomic_input_norm += x * x;

                float mem_projection = 0.0f;
                float mem_norm_sq = 0.0f;
                for (size_t j = 0; j < N; ++j)
                {
                    float w = acc_Wf[i * N + j] + acc_Ws[i * N + j];
                    mem_projection += w * x;
                    mem_norm_sq += w * w;
                }

                s::atomic_ref<float, s::memory_order::relaxed, s::memory_scope::device,
                              s::access::address_space::global_space>
                    atomic_mem_norm(*d_memory_norm);
                atomic_mem_norm += mem_norm_sq;

                s::atomic_ref<float, s::memory_order::relaxed, s::memory_scope::device,
                              s::access::address_space::global_space>
                    atomic_dot(*d_dot_product);
                atomic_dot += mem_projection;
            });
        });
        if (profile_events)
            profile_events->push_back(ev1);

        auto ev2 = q.submit([&](s::handler &h) {
            h.single_task([=]() {
                float input_norm = s::sqrt(*d_input_norm + 1e-8f);
                float memory_norm = s::sqrt(*d_memory_norm + 1e-8f);
                float dot_product = *d_dot_product;

                float familiarity = dot_product / (input_norm * memory_norm + 1e-8f);
                *d_familiarity = s::clamp(familiarity, 0.0f, 1.0f);
            });
        });
        if (profile_events)
            profile_events->push_back(ev2);

        return d_familiarity_;
    }

    /**
     * 計算熟悉度評分
     * 使用餘弦相似度衡量輸入與記憶的匹配程度
     */
    RecallResult compute_familiarity(s::queue &q, s::buffer<float, 1> &buf_X, // 輸入 [input_dim]
                                     s::buffer<float, 1> &buf_W_fast,         // 快速權重 [input_dim * hidden_dim]
                                     s::buffer<float, 1> &buf_W_slow,         // 慢速權重 [input_dim * hidden_dim]
                                     std::vector<s::event> *profile_events = nullptr)
    {
        const float *familiarity_ptr = enqueue_familiarity(q, buf_X, buf_W_fast, buf_W_slow, profile_events);
        q.wait();
        float familiarity = std::clamp(*familiarity_ptr, 0.0f, 1.0f);

        // 根據熟悉度計算建議閾值
        // 熟悉 -> 使用學習到的閾值 (更快收斂)
        // 新奇 -> 使用較高初始閾值 (需要更多探索)
        float suggested_th = learned_threshold_;
        if (familiarity < cfg_.familiarity_threshold)
        {
            // 新奇輸入: 使用保守的高閾值
            suggested_th = learned_threshold_ * (1.0f + (1.0f - familiarity) * cfg_.priming_strength);
        }
        else
        {
            // 熟悉輸入: 直接使用學習到的閾值
            suggested_th = learned_threshold_ * (1.0f - familiarity * cfg_.priming_strength * 0.2f);
        }

        last_familiarity_ = familiarity;

        return RecallResult{.familiarity_score = familiarity,
                            .suggested_threshold = suggested_th,
                            .is_familiar = (familiarity >= cfg_.familiarity_threshold)};
    }

    /**
     * 選擇性記憶注入
     * 策略: 適度增強，而非完全正規化
     */
    void inject_memory(s::queue &q, s::buffer<float, 1> &buf_X, // 輸入 [input_dim]
                       s::buffer<float, 1> &buf_W_fast,         // 快速權重
                       s::buffer<float, 1> &buf_W_slow,         // 慢速權重
                       s::buffer<float, 1> &buf_H,              // 隱藏狀態 [hidden_dim] (in/out)
                       float familiarity)                       // 熟悉度 (用於調節注入強度)
    {
        const size_t M = cfg_.input_dim;
        const size_t N = cfg_.hidden_dim;
        const float scale = cfg_.injection_scale * familiarity; // 熟悉度越高注入越強
        const bool normalize = cfg_.normalize_injection;

        if (scale < 0.01f)
            return; // 熟悉度太低，不注入

        // 計算記憶響應
        q.submit([&](s::handler &h) {
            s::accessor acc_X{buf_X, h, s::read_only};
            s::accessor acc_Wf{buf_W_fast, h, s::read_only};
            s::accessor acc_Ws{buf_W_slow, h, s::read_only};
            s::accessor acc_H{buf_H, h, s::read_write};

            h.parallel_for(s::range<1>{N}, [=](s::id<1> idx) {
                size_t j = idx[0];
                float h_val = acc_H[j];

                // 計算記憶響應
                float mem_response = 0.0f;
                for (size_t i = 0; i < M; ++i)
                {
                    float w = acc_Wf[i * N + j] + acc_Ws[i * N + j];
                    mem_response += w * acc_X[i];
                }

                // 注入策略: 混合原始信號與記憶
                // 保留大部分原始信號，只做適度增強
                float new_h = h_val * (1.0f - scale * 0.5f) + scale * mem_response;

                // 軟正規化: 限制過大的值，但允許適度增強
                if (normalize)
                {
                    float abs_h = (new_h > 0) ? new_h : -new_h;
                    if (abs_h > 5.0f)
                    {
                        new_h = new_h * (5.0f / abs_h);
                    }
                }

                acc_H[j] = new_h;
            });
        });
    }

    /**
     * Hebbian 學習 (帶穩定性考量)
     * 學習時同時記錄穩定閾值
     */
    void learn(s::queue &q, s::buffer<float, 1> &buf_X, s::buffer<float, 1> &buf_Z, // SNN 輸出 (spike)
               s::buffer<float, 1> &buf_W_fast,
               float final_threshold) // 學習時的穩定閾值
    {
        const size_t M = cfg_.input_dim;
        const size_t N = cfg_.hidden_dim;
        const float eta = cfg_.learning_rate;
        const float clip = cfg_.weight_clip;

        q.submit([&](s::handler &h) {
            s::accessor acc_X{buf_X, h, s::read_only};
            s::accessor acc_Z{buf_Z, h, s::read_only};
            s::accessor acc_W{buf_W_fast, h, s::read_write};

            h.parallel_for(s::range<2>{M, N}, [=](s::id<2> idx) {
                size_t i = idx[0];
                size_t j = idx[1];

                // Hebbian: dW = eta * x * z
                float delta = eta * acc_X[i] * acc_Z[j];
                float new_w = acc_W[i * N + j] + delta;
                acc_W[i * N + j] = s::clamp(new_w, -clip, clip);
            });
        });

        // 記錄學習時的穩定閾值 (指數移動平均)
        learned_threshold_ = 0.9f * learned_threshold_ + 0.1f * final_threshold;
    }

    /**
     * 記憶固化
     */
    void consolidate(s::queue &q, s::buffer<float, 1> &buf_W_fast, s::buffer<float, 1> &buf_W_slow)
    {
        const size_t total = cfg_.input_dim * cfg_.hidden_dim;
        const float transfer = cfg_.consolidation_rate;
        const float decay = cfg_.decay_rate;
        const float clip = cfg_.weight_clip;

        q.submit([&](s::handler &h) {
            s::accessor acc_fast{buf_W_fast, h, s::read_write};
            s::accessor acc_slow{buf_W_slow, h, s::read_write};

            h.parallel_for(s::range<1>{total}, [=](s::id<1> idx) {
                size_t i = idx[0];
                float wf = acc_fast[i];

                // 轉移到慢記憶
                float new_slow = acc_slow[i] + wf * transfer;
                acc_slow[i] = s::clamp(new_slow, -clip, clip);

                // 快記憶衰減
                acc_fast[i] = wf * (1.0f - decay);
            });
        });
    }

    /**
     * 獲取學習到的閾值
     */
    float get_learned_threshold() const
    {
        return learned_threshold_;
    }

    /**
     * 設置學習到的閾值
     */
    void set_learned_threshold(float th)
    {
        learned_threshold_ = th;
    }

    /**
     * 獲取上次熟悉度
     */
    float get_last_familiarity() const
    {
        return last_familiarity_;
    }

    /**
     * 重置
     */
    void reset()
    {
        last_familiarity_ = 0.0f;
        // 保留 learned_threshold_
    }

    const Config &config() const
    {
        return cfg_;
    }

  private:
    Config cfg_;
    float last_familiarity_;
    float learned_threshold_;

    void ensure_usm(s::queue &q)
    {
        if (usm_initialized_)
            return;
        usm_context_ = q.get_context();
        d_dot_product_ = s::malloc_shared<float>(1, q);
        d_input_norm_ = s::malloc_shared<float>(1, q);
        d_memory_norm_ = s::malloc_shared<float>(1, q);
        d_familiarity_ = s::malloc_shared<float>(1, q);
        *d_dot_product_ = 0.0f;
        *d_input_norm_ = 0.0f;
        *d_memory_norm_ = 0.0f;
        *d_familiarity_ = 0.0f;
        usm_initialized_ = true;
    }

    bool usm_initialized_ = false;
    s::context usm_context_{};
    float *d_dot_product_ = nullptr;
    float *d_input_norm_ = nullptr;
    float *d_memory_norm_ = nullptr;
    float *d_familiarity_ = nullptr;
};

} // namespace components
} // namespace neurobit
