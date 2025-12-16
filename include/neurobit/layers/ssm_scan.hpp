#pragma once

#include "neurobit/core/types.hpp"
#include <cmath>
#include <sycl/sycl.hpp>
#include <vector>

namespace neurobit
{
namespace layers
{

class SSMScan
{
  public:
    struct Config
    {
        size_t batch_size;
        size_t seq_len;
        size_t state_dim;
        bool use_stability_constraint = true; // 使用 tanh 約束 A
    };

    SSMScan(const Config &config) : cfg_(config)
    {
    }

    // 完整 SSM: s[t+1] = A*s[t] + B*x[t], y[t] = C*s[t] + D*x[t]
    // 簡化版本: s[t+1] = A*s[t] + x[t], y[t] = s[t] (B=I, C=I, D=0)
    void forward(s::queue &q, s::buffer<float, 1> &buf_X, // Input [B*L*D] or [B*D]
                 s::buffer<float, 1> &buf_A,              // SSM coefficients [D]
                 s::buffer<float, 1> &buf_Y,              // Output [B*L*D] or [B*D]
                 s::buffer<float, 1> &buf_State,          // Persistent state [B*D]
                 std::vector<s::event> *profile_events = nullptr)
    {
        const size_t B = cfg_.batch_size;
        const size_t L = cfg_.seq_len;
        const size_t D = cfg_.state_dim;
        const bool use_tanh = cfg_.use_stability_constraint;

        auto ev = q.submit([&](s::handler &h) {
            s::accessor acc_X{buf_X, h, s::read_only};
            s::accessor acc_A{buf_A, h, s::read_only};
            s::accessor acc_Y{buf_Y, h, s::write_only, s::no_init};
            s::accessor acc_S{buf_State, h, s::read_write};

            s::range<2> global_range{B, D};

            h.parallel_for(global_range, [=](s::id<2> idx) {
                size_t b = idx[0];
                size_t d = idx[1];

                // 獲取 A 係數並應用穩定性約束
                float a_raw = acc_A[d];
                float a_val = use_tanh ? s::tanh(a_raw) : a_raw;

                // 額外安全檢查: 確保 |a| < 1
                a_val = s::clamp(a_val, -0.999f, 0.999f);

                // 從持久狀態讀取
                float current_state = acc_S[b * D + d];

                // 序列掃描
                for (size_t t = 0; t < L; ++t)
                {
                    float x_val = acc_X[b * L * D + t * D + d];
                    current_state = a_val * current_state + x_val;
                    acc_Y[b * L * D + t * D + d] = current_state;
                }

                // 保存狀態供下次使用
                acc_S[b * D + d] = current_state;
            });
        });
        if (profile_events)
            profile_events->push_back(ev);
    }

    // 簡化版本: 單步更新 (seq_len=1)
    void forward_single(s::queue &q, s::buffer<float, 1> &buf_X, // Input [B*D]
                        s::buffer<float, 1> &buf_A,              // SSM coefficients [D]
                        s::buffer<float, 1> &buf_Y,              // Output [B*D]
                        s::buffer<float, 1> &buf_State,          // Persistent state [B*D]
                        std::vector<s::event> *profile_events = nullptr)
    {
        const size_t B = cfg_.batch_size;
        const size_t D = cfg_.state_dim;
        const bool use_tanh = cfg_.use_stability_constraint;

        auto ev = q.submit([&](s::handler &h) {
            s::accessor acc_X{buf_X, h, s::read_only};
            s::accessor acc_A{buf_A, h, s::read_only};
            s::accessor acc_Y{buf_Y, h, s::write_only, s::no_init};
            s::accessor acc_S{buf_State, h, s::read_write};

            s::range<2> global_range{B, D};

            h.parallel_for(global_range, [=](s::id<2> idx) {
                size_t b = idx[0];
                size_t d = idx[1];
                size_t linear_idx = b * D + d;

                float a_raw = acc_A[d];
                float a_val = use_tanh ? s::tanh(a_raw) : a_raw;
                a_val = s::clamp(a_val, -0.999f, 0.999f);

                float x_val = acc_X[linear_idx];
                float new_state = a_val * acc_S[linear_idx] + x_val;

                acc_S[linear_idx] = new_state;
                acc_Y[linear_idx] = new_state;
            });
        });
        if (profile_events)
            profile_events->push_back(ev);
    }

    // 重置狀態
    void reset_state(s::queue &q, s::buffer<float, 1> &buf_State, std::vector<s::event> *profile_events = nullptr)
    {
        const size_t total = cfg_.batch_size * cfg_.state_dim;
        auto ev = q.submit([&](s::handler &h) {
            s::accessor acc_S{buf_State, h, s::write_only, s::no_init};
            h.parallel_for(s::range<1>{total}, [=](s::id<1> idx) { acc_S[idx] = 0.0f; });
        });
        if (profile_events)
            profile_events->push_back(ev);
    }

  private:
    Config cfg_;
};

} // namespace layers
} // namespace neurobit
