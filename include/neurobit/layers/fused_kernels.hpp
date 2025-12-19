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
#include "../core/bit_packing.hpp"
#include "../core/types.hpp"
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/sycl.hpp>

namespace neurobit
{
namespace layers
{
namespace s = sycl;
using bfloat16 = s::ext::oneapi::bfloat16;

/**
 * 融合前向傳播 Kernel
 *
 * 融合操作:
 * 1. BitLinear: Y = W × X
 * 2. SSM: s = A×s + B×Y
 * 3. Memory Injection: s += mem × familiarity
 *
 * 從 3 個 kernel → 1 個融合 kernel
 * 預期提升: 2-3x
 */
template <typename T = bfloat16> class FusedForward
{
  private:
    s::queue &queue_;
    size_t hidden_dim_;

  public:
    explicit FusedForward(s::queue &q, size_t hidden_dim) : queue_(q), hidden_dim_(hidden_dim)
    {
    }

    /**
     * 融合前向傳播
     *
     * 輸入:
     *   X: [batch, input_dim] - 輸入
     *   W_packed: [output_dim, input_dim/4] - Bit-packed 權重
     *   state: [output_dim] - SSM 狀態
     *   memory: [output_dim] - 記憶權重
     *   A, B: SSM 參數
     *   familiarity: 記憶熟悉度
     *
     * 輸出:
     *   Y: [batch, output_dim] - 輸出
     *   state: [output_dim] - 更新的狀態
     */
    void forward(s::buffer<T> &X, s::buffer<uint8_t> &W_packed, s::buffer<T> &state, s::buffer<T> &memory,
                 s::buffer<T> &Y, float A, float B, float familiarity, size_t batch_size, size_t input_dim,
                 size_t output_dim)
    {
        queue_.submit([&](s::handler &h) {
            auto x = X.template get_access<s::access::mode::read>(h);
            auto w = W_packed.template get_access<s::access::mode::read>(h);
            auto s = state.template get_access<s::access::mode::read_write>(h);
            auto m = memory.template get_access<s::access::mode::read>(h);
            auto y = Y.template get_access<s::access::mode::write>(h);

            // 使用 local memory 優化
            constexpr size_t TILE = 16;
            s::local_accessor<T, 1> shared_state{output_dim, h};
            s::local_accessor<T, 1> shared_mem{output_dim, h};

            h.parallel_for(s::nd_range<1>{output_dim, TILE}, [=](s::nd_item<1> it) {
                size_t i = it.get_global_id(0);
                size_t local_i = it.get_local_id(0);

                // 1. 載入狀態和記憶到 shared memory
                if (local_i == 0)
                {
                    for (size_t j = 0; j < output_dim; ++j)
                    {
                        shared_state[j] = s[j];
                        shared_mem[j] = m[j];
                    }
                }
                it.barrier(s::access::fence_space::local_space);

                // 對每個 batch
                for (size_t b = 0; b < batch_size; ++b)
                {
                    // 2. BitLinear: y = W × x (bit-packed)
                    T y_val = T(0);

                    for (size_t k = 0; k < input_dim / 4; ++k)
                    {
                        // 解包 4 個權重
                        auto w4 = core::BitPackedWeights::unpack4(w[i * (input_dim / 4) + k]);

                        // 矩陣乘法
                        y_val += T(w4[0]) * x[b * input_dim + k * 4 + 0];
                        y_val += T(w4[1]) * x[b * input_dim + k * 4 + 1];
                        y_val += T(w4[2]) * x[b * input_dim + k * 4 + 2];
                        y_val += T(w4[3]) * x[b * input_dim + k * 4 + 3];
                    }

                    // 3. SSM 更新: s = A×s + B×y (融合)
                    T state_val = shared_state[i];
                    state_val = T(A) * state_val + T(B) * y_val;

                    // 4. Memory Injection (融合)
                    if (familiarity > 0.01f)
                    {
                        T mem_contrib = shared_mem[i] * T(familiarity);
                        state_val = state_val * (T(1.0f) - T(familiarity) * T(0.5f)) + mem_contrib;
                    }

                    // 更新 shared state
                    shared_state[i] = state_val;

                    // 輸出
                    y[b * output_dim + i] = state_val;
                }

                // 5. 寫回全局記憶體
                it.barrier(s::access::fence_space::local_space);
                if (local_i == 0)
                {
                    for (size_t j = 0; j < output_dim; ++j)
                    {
                        s[j] = shared_state[j];
                    }
                }
            });
        });
    }

    /**
     * 優化版本: 使用向量化
     */
    void forward_vectorized(s::buffer<T> &X, s::buffer<uint8_t> &W_packed, s::buffer<T> &state, s::buffer<T> &memory,
                            s::buffer<T> &Y, float A, float B, float familiarity, size_t batch_size, size_t input_dim,
                            size_t output_dim)
    {
        constexpr size_t VEC_SIZE = 4;

        queue_.submit([&](s::handler &h) {
            auto x = X.template get_access<s::access::mode::read>(h);
            auto w = W_packed.template get_access<s::access::mode::read>(h);
            auto s = state.template get_access<s::access::mode::read_write>(h);
            auto m = memory.template get_access<s::access::mode::read>(h);
            auto y = Y.template get_access<s::access::mode::write>(h);

            h.parallel_for(s::nd_range<2>{{batch_size, output_dim / VEC_SIZE}, {1, 64}}, [=](s::nd_item<2> it) {
                size_t b = it.get_global_id(0);
                size_t i_vec = it.get_global_id(1);
                size_t i = i_vec * VEC_SIZE;

                // 向量化處理 VEC_SIZE 個輸出
                s::vec<float, VEC_SIZE> y_vec{0.0f};
                s::vec<float, VEC_SIZE> state_vec;
                s::vec<float, VEC_SIZE> mem_vec;

                // 載入狀態和記憶
                for (size_t v = 0; v < VEC_SIZE; ++v)
                {
                    state_vec[v] = static_cast<float>(s[i + v]);
                    mem_vec[v] = static_cast<float>(m[i + v]);
                }

                // BitLinear (向量化)
                for (size_t k = 0; k < input_dim / 4; ++k)
                {
                    T x_val[4];
                    for (size_t kk = 0; kk < 4; ++kk)
                    {
                        x_val[kk] = x[b * input_dim + k * 4 + kk];
                    }

                    for (size_t v = 0; v < VEC_SIZE; ++v)
                    {
                        auto w4 = core::BitPackedWeights::unpack4(w[(i + v) * (input_dim / 4) + k]);
                        y_vec[v] += static_cast<float>(w4[0] * static_cast<int8_t>(x_val[0]));
                        y_vec[v] += static_cast<float>(w4[1] * static_cast<int8_t>(x_val[1]));
                        y_vec[v] += static_cast<float>(w4[2] * static_cast<int8_t>(x_val[2]));
                        y_vec[v] += static_cast<float>(w4[3] * static_cast<int8_t>(x_val[3]));
                    }
                }

                // SSM + Memory (向量化)
                for (size_t v = 0; v < VEC_SIZE; ++v)
                {
                    // SSM
                    state_vec[v] = A * state_vec[v] + B * y_vec[v];

                    // Memory injection
                    if (familiarity > 0.01f)
                    {
                        state_vec[v] = state_vec[v] * (1.0f - familiarity * 0.5f) + mem_vec[v] * familiarity;
                    }

                    // 存儲
                    s[i + v] = T(state_vec[v]);
                    y[b * output_dim + i + v] = T(state_vec[v]);
                }
            });
        });
    }
};

/**
 * SNN 推理循環融合
 *
 * 將整個 SNN 推理循環融合到單一 kernel
 * 減少 kernel 啟動開銷和記憶體傳輸
 */
template <typename T = bfloat16> class FusedSNNInference
{
  private:
    s::queue &queue_;
    size_t num_steps_;

  public:
    FusedSNNInference(s::queue &q, size_t num_steps) : queue_(q), num_steps_(num_steps)
    {
    }

    /**
     * 融合 SNN 推理
     * 在 device 上完成整個時間步循環
     */
    void infer(s::buffer<T> &input, s::buffer<uint8_t> &W_packed, s::buffer<T> &state, s::buffer<T> &output,
               s::buffer<T> &thresholds, size_t hidden_dim, size_t input_dim)
    {
        size_t num_steps = num_steps_; // 捕獲成員變量

        queue_.submit([&](s::handler &h) {
            auto x = input.template get_access<s::access::mode::read>(h);
            auto w = W_packed.template get_access<s::access::mode::read>(h);
            auto s_buf = state.template get_access<s::access::mode::read_write>(h);
            auto out = output.template get_access<s::access::mode::write>(h);
            auto thresh = thresholds.template get_access<s::access::mode::read>(h);

            // Local memory for state
            s::local_accessor<T, 1> local_state{hidden_dim, h};
            s::local_accessor<T, 1> local_spike{hidden_dim, h};

            h.parallel_for(s::nd_range<1>{hidden_dim, 16}, [=](s::nd_item<1> it) {
                size_t i = it.get_global_id(0);
                size_t local_i = it.get_local_id(0);

                // 載入初始狀態
                if (local_i == 0)
                {
                    for (size_t j = 0; j < hidden_dim; ++j)
                    {
                        local_state[j] = s_buf[j];
                        local_spike[j] = T(0);
                    }
                }
                it.barrier(s::access::fence_space::local_space);

                // 時間步循環 (在 device 上)
                for (size_t t = 0; t < num_steps; ++t)
                {
                    // 1. BitLinear
                    T y = T(0);
                    for (size_t k = 0; k < input_dim / 4; ++k)
                    {
                        auto w4 = core::BitPackedWeights::unpack4(w[i * (input_dim / 4) + k]);
                        y += T(w4[0]) * x[t * input_dim + k * 4 + 0];
                        y += T(w4[1]) * x[t * input_dim + k * 4 + 1];
                        y += T(w4[2]) * x[t * input_dim + k * 4 + 2];
                        y += T(w4[3]) * x[t * input_dim + k * 4 + 3];
                    }

                    // 2. 更新狀態
                    T s_val = local_state[i];
                    s_val = s_val * T(0.9f) + y;

                    // 3. Spike 檢測
                    T spike = T(0);
                    if (s_val > thresh[i])
                    {
                        spike = T(1);
                        s_val = s_val - thresh[i]; // Reset
                    }

                    local_state[i] = s_val;
                    local_spike[i] = spike;

                    it.barrier(s::access::fence_space::local_space);

                    // 4. 輸出
                    out[t * hidden_dim + i] = spike;
                }

                // 寫回狀態
                if (local_i == 0)
                {
                    for (size_t j = 0; j < hidden_dim; ++j)
                    {
                        s_buf[j] = local_state[j];
                    }
                }
            });
        });
    }
};

} // namespace layers
} // namespace neurobit
