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
#include "../core/precision.hpp"
#include "../core/types.hpp"
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/sycl.hpp>

namespace neurobit
{
namespace layers
{
namespace s = sycl;
using bfloat16 = s::ext::oneapi::bfloat16;
namespace matrix = s::ext::oneapi::experimental::matrix;

template <typename ComputeT = bfloat16> class BitLinearXMX
{
  private:
    s::queue &queue_;
    size_t input_dim_;  // K
    size_t output_dim_; // N
    bool use_xmx_;
    bool use_slm_tiling_;

    s::buffer<uint8_t> W_packed_;

    static constexpr size_t TM = 8;
    static constexpr size_t TN = 16;
    static constexpr size_t TK = 16;

  public:
    BitLinearXMX(s::queue &q, size_t input_dim, size_t output_dim, bool enable_slm_tiling = true)
        : queue_(q), input_dim_(input_dim), output_dim_(output_dim), use_slm_tiling_(enable_slm_tiling),
          W_packed_(s::range<1>(core::BitPackedWeights::packed_size(input_dim * output_dim)))
    {
        auto device = q.get_device();
        auto sg_sizes = device.get_info<s::info::device::sub_group_sizes>();
        use_xmx_ = std::find(sg_sizes.begin(), sg_sizes.end(), 16) != sg_sizes.end();

        // SLM check is now dynamic based on selected tile size in forward()
    }

    void set_weights(const std::vector<int8_t> &weights)
    {
        if (weights.size() != input_dim_ * output_dim_)
            throw std::invalid_argument("Size mismatch");
        auto packed = core::BitPackedWeights::pack_array(weights);
        queue_.submit([&](s::handler &h) {
            auto acc = W_packed_.template get_access<s::access::mode::write>(h);
            h.copy(packed.data(), acc);
        });
        queue_.wait();
    }

    template <typename InputT, typename OutputT>
    void forward(s::buffer<InputT> &X, s::buffer<OutputT> &Y, size_t batch_size)
    {
        if (use_xmx_)
        {
            if (use_slm_tiling_)
            {
                // Adaptive Tiling Selection
                if (batch_size >= 256 && output_dim_ >= 128 && input_dim_ >= 32)
                {
                    forward_xmx_slm_template<64, 128, 32>(X, Y, batch_size);
                }
                else if (batch_size >= 64 && output_dim_ >= 64 && input_dim_ >= 32)
                {
                    forward_xmx_slm_template<32, 64, 32>(X, Y, batch_size);
                }
                else
                {
                    forward_xmx_universal(X, Y, batch_size);
                }
            }
            else
            {
                forward_xmx_universal(X, Y, batch_size);
            }
        }
        else
        {
            forward_standard(X, Y, batch_size);
        }
    }

    // Compatibility wrappers
    template <typename InputT, typename OutputT>
    void forward_single(s::queue &, s::buffer<InputT> &X, s::buffer<int8_t, 1> &W_ext, s::buffer<OutputT> &Y)
    {
        update_packed_weights(W_ext);
        forward(X, Y, 1);
    }
    template <typename InputT, typename OutputT>
    void forward_single(s::queue &, s::buffer<InputT> &X, s::buffer<OutputT> &Y)
    {
        forward(X, Y, 1);
    }

    enum class Mode
    {
        Standard,
        XMX_Universal,
        XMX_SLM_Medium,
        XMX_SLM_Large
    };
    Mode get_mode(size_t batch_size) const
    {
        if (!use_xmx_)
            return Mode::Standard;
        if (!use_slm_tiling_)
            return Mode::XMX_Universal;

        if (batch_size >= 256 && output_dim_ >= 128 && input_dim_ >= 32)
            return Mode::XMX_SLM_Large;
        if (batch_size >= 64 && output_dim_ >= 64 && input_dim_ >= 32)
            return Mode::XMX_SLM_Medium;
        return Mode::XMX_Universal;
    }

  private:
    void update_packed_weights(s::buffer<int8_t, 1> &W_ext)
    {
        size_t K = input_dim_;
        size_t N = output_dim_;
        queue_.submit([&](s::handler &h) {
            auto w_in = W_ext.template get_access<s::access::mode::read>(h);
            auto w_out = W_packed_.template get_access<s::access::mode::write>(h);
            h.parallel_for(s::range<1>((K * N + 3) / 4), [=](s::id<1> idx) {
                size_t start = idx[0] * 4;
                uint8_t packed = 0;
                for (int i = 0; i < 4; ++i)
                {
                    if (start + i < K * N)
                    {
                        int8_t val = w_in[start + i];
                        uint8_t bits = static_cast<uint8_t>(val + 1);
                        packed |= (bits << (i * 2));
                    }
                }
                w_out[idx[0]] = packed;
            });
        });
    }

    // =========================================================
    // Kernel 1: Universal XMX (No Tiling, Direct Load)
    // =========================================================
    template <typename InputT, typename OutputT>
    void forward_xmx_universal(s::buffer<InputT> &X, s::buffer<OutputT> &Y, size_t M)
    {
        const size_t K = input_dim_;
        const size_t N = output_dim_;
        size_t num_m_blocks = (M + TM - 1) / TM;
        size_t num_n_blocks = (N + TN - 1) / TN;

        queue_.submit([&](s::handler &h) {
            auto acc_x = X.template get_access<s::access::mode::read>(h);
            auto acc_y = Y.template get_access<s::access::mode::write>(h);
            auto acc_w = W_packed_.template get_access<s::access::mode::read>(h);

            s::local_accessor<ComputeT, 2> slm_w{s::range<2>{TK, TN}, h};
            s::local_accessor<ComputeT, 2> slm_x{s::range<2>{TM, TK}, h};
            s::local_accessor<float, 2> slm_acc{s::range<2>{TM, TN}, h};

            h.parallel_for(
                s::nd_range<2>{s::range<2>{num_m_blocks, num_n_blocks * 16}, s::range<2>{1, 16}},
                [=](s::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                    auto sg = it.get_sub_group();
                    size_t m_base = it.get_group(0) * TM;
                    size_t n_base = it.get_group(1) * TN;
                    size_t lane = sg.get_local_id()[0];

                    matrix::joint_matrix<s::sub_group, float, matrix::use::accumulator, TM, TN> acc;
                    matrix::joint_matrix_fill(sg, acc, 0.0f);

                    for (size_t k_base = 0; k_base < K; k_base += TK)
                    {
                        // 1. Load X [TM, TK] -> SLM
                        for (int r = 0; r < TM; ++r)
                        {
                            size_t cur_m = m_base + r;
                            size_t cur_k = k_base + lane;
                            if (cur_m < M && cur_k < K)
                            {
                                slm_x[r][lane] = ComputeT(acc_x[cur_m * K + cur_k]);
                            }
                            else
                            {
                                slm_x[r][lane] = ComputeT(0.0f);
                            }
                        }

                        // 2. Load W [TK, TN] -> SLM
                        if (k_base + lane < K)
                        {
                            for (int c_blk = 0; c_blk < TN / 4; ++c_blk)
                            {
                                size_t col_start = c_blk * 4;
                                for (int bit = 0; bit < 4; ++bit)
                                {
                                    if (n_base + col_start + bit < N)
                                    {
                                        size_t current_w_idx = (k_base + lane) * N + (n_base + col_start + bit);
                                        uint8_t packed = acc_w[current_w_idx / 4];
                                        size_t shift = (current_w_idx % 4) * 2;
                                        uint8_t val_bits = (packed >> shift) & 0x03;
                                        slm_w[lane][col_start + bit] = ComputeT(static_cast<float>(val_bits) - 1.0f);
                                    }
                                    else
                                    {
                                        slm_w[lane][col_start + bit] = ComputeT(0.0f);
                                    }
                                }
                            }
                        }
                        else
                        {
                            for (int c = 0; c < TN; ++c)
                                slm_w[lane][c] = ComputeT(0.0f);
                        }

                        s::group_barrier(sg);

                        matrix::joint_matrix<s::sub_group, ComputeT, matrix::use::a, TM, TK, matrix::layout::row_major>
                            mat_x;
                        matrix::joint_matrix<s::sub_group, ComputeT, matrix::use::b, TK, TN, matrix::layout::row_major>
                            mat_w;

                        matrix::joint_matrix_load(sg, mat_x, slm_x.template get_multi_ptr<s::access::decorated::no>(),
                                                  TK);
                        matrix::joint_matrix_load(sg, mat_w, slm_w.template get_multi_ptr<s::access::decorated::no>(),
                                                  TN);
                        matrix::joint_matrix_mad(sg, acc, mat_x, mat_w, acc);

                        s::group_barrier(sg);
                    }

                    matrix::joint_matrix_store(sg, acc, slm_acc.template get_multi_ptr<s::access::decorated::no>(), TN,
                                               matrix::layout::row_major);
                    s::group_barrier(sg);

                    for (int r = 0; r < TM; ++r)
                    {
                        if (m_base + r < M && (n_base + lane < N))
                        {
                            acc_y[(m_base + r) * N + n_base + lane] = OutputT(slm_acc[r][lane]);
                        }
                    }
                });
        });
    }

    // =========================================================
    // Kernel 2: SLM Tiled Template (Vectorized Optimization)
    // =========================================================
    template <int TILE_M, int TILE_N, int TILE_K, typename InputT, typename OutputT>
    void forward_xmx_slm_template(s::buffer<InputT> &X, s::buffer<OutputT> &Y, size_t M)
    {
        const size_t K = input_dim_;
        const size_t N = output_dim_;

        queue_.submit([&](s::handler &h) {
            auto acc_x = X.template get_access<s::access::mode::read>(h);
            auto acc_y = Y.template get_access<s::access::mode::write>(h);
            auto acc_w = W_packed_.template get_access<s::access::mode::read>(h);

            s::local_accessor<ComputeT, 2> slm_x{s::range<2>{TILE_M, TILE_K}, h};
            s::local_accessor<ComputeT, 2> slm_w{s::range<2>{TILE_K, TILE_N}, h};
            s::local_accessor<float, 2> slm_acc{s::range<2>{TILE_M, TILE_N}, h};

            size_t m_tiles = (M + TILE_M - 1) / TILE_M;
            size_t n_tiles = (N + TILE_N - 1) / TILE_N;

            h.parallel_for(
                s::nd_range<2>{s::range<2>{m_tiles * (TILE_M / TM), n_tiles * (TILE_N / TN) * 16},
                               s::range<2>{TILE_M / TM, (TILE_N / TN) * 16}},
                [=](s::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                    auto sg = it.get_sub_group();
                    size_t m_grp = it.get_group(0);
                    size_t n_grp = it.get_group(1);

                    size_t m_base = m_grp * TILE_M;
                    size_t n_base = n_grp * TILE_N;

                    size_t loc_id = it.get_local_linear_id();
                    size_t sg_id = loc_id / 16;
                    size_t sg_m_off = (sg_id / (TILE_N / TN)) * TM;
                    size_t sg_n_off = (sg_id % (TILE_N / TN)) * TN;

                    matrix::joint_matrix<s::sub_group, float, matrix::use::accumulator, TM, TN> acc;
                    matrix::joint_matrix_fill(sg, acc, 0.0f);

                    // Constants for vectorization
                    constexpr int THREADS = (TILE_M / TM) * (TILE_N / TN) * 16; // 256 typically

                    // 1. Vectorized Load X
                    constexpr int ELEMS_X = TILE_M * TILE_K;
                    constexpr int FLOATS_PER_THREAD = (ELEMS_X + THREADS - 1) / THREADS;

                    // 2. Vectorized Load W (Bytes)
                    constexpr int WEIGHTS_PER_TILE = TILE_K * TILE_N;
                    constexpr int BYTES_PER_TILE = WEIGHTS_PER_TILE / 4;
                    constexpr int BYTES_PER_THREAD = (BYTES_PER_TILE + THREADS - 1) / THREADS;

                    for (size_t k_base = 0; k_base < K; k_base += TILE_K)
                    {
// 1. Load X (Vectorized)
#pragma unroll
                        for (int i = 0; i < FLOATS_PER_THREAD; ++i)
                        {
                            int idx = loc_id * FLOATS_PER_THREAD + i;
                            if (idx < ELEMS_X)
                            {
                                int r = idx / TILE_K;
                                int c = idx % TILE_K;
                                if (m_base + r < M && k_base + c < K)
                                {
                                    slm_x[r][c] = ComputeT(acc_x[(m_base + r) * K + (k_base + c)]);
                                }
                                else
                                {
                                    slm_x[r][c] = ComputeT(0.0f);
                                }
                            }
                        }

// 2. Load W (Vectorized Byte Access)
#pragma unroll
                        for (int i = 0; i < BYTES_PER_THREAD; ++i)
                        {
                            int byte_idx = loc_id * BYTES_PER_THREAD + i;
                            if (byte_idx < BYTES_PER_TILE)
                            {
                                // Tile is [TILE_K, TILE_N/4] bytes.
                                int tile_width_bytes = TILE_N / 4;
                                int r = byte_idx / tile_width_bytes;
                                int c_byte = byte_idx % tile_width_bytes;

                                int gx = k_base + r;
                                int gy_byte = (n_base / 4) + c_byte;

                                // Global packed index assuming N is multiple of 4
                                size_t global_byte_idx = gx * (N / 4) + gy_byte;

                                uint8_t packed = 0;
                                if (gx < K && gy_byte * 4 < N)
                                {
                                    packed = acc_w[global_byte_idx];
                                }

// Unpack 4 weights
#pragma unroll
                                for (int b = 0; b < 4; ++b)
                                {
                                    uint8_t val = (packed >> (b * 2)) & 0x03;
                                    slm_w[r][c_byte * 4 + b] = ComputeT(static_cast<float>(val) - 1.0f);
                                }
                            }
                        }

                        s::group_barrier(sg);

// 3. Compute
#pragma unroll
                        for (int k = 0; k < TILE_K; k += TK)
                        {
                            matrix::joint_matrix<s::sub_group, ComputeT, matrix::use::a, TM, TK,
                                                 matrix::layout::row_major>
                                mat_x;
                            matrix::joint_matrix<s::sub_group, ComputeT, matrix::use::b, TK, TN,
                                                 matrix::layout::row_major>
                                mat_w;

                            matrix::joint_matrix_load(sg, mat_x,
                                                      slm_x.template get_multi_ptr<s::access::decorated::no>() +
                                                          sg_m_off * TILE_K + k,
                                                      TILE_K);
                            matrix::joint_matrix_load(sg, mat_w,
                                                      slm_w.template get_multi_ptr<s::access::decorated::no>() +
                                                          k * TILE_N + sg_n_off,
                                                      TILE_N);

                            matrix::joint_matrix_mad(sg, acc, mat_x, mat_w, acc);
                        }
                        s::group_barrier(sg);
                    }

                    // 4. Store
                    auto slm_ptr = slm_acc.template get_multi_ptr<s::access::decorated::no>();
                    matrix::joint_matrix_store(sg, acc, slm_ptr + sg_m_off * TILE_N + sg_n_off, TILE_N,
                                               matrix::layout::row_major);
                    s::group_barrier(sg);

                    const size_t items_y = TILE_M * TILE_N;
                    const size_t store_per_thread = (items_y + THREADS - 1) / THREADS;
#pragma unroll
                    for (int i = 0; i < store_per_thread; ++i)
                    {
                        size_t lin = loc_id * store_per_thread + i;
                        if (lin < items_y)
                        {
                            size_t r = lin / TILE_N;
                            size_t c = lin % TILE_N;
                            if (m_base + r < M && n_base + c < N)
                            {
                                acc_y[(m_base + r) * N + n_base + c] = OutputT(slm_acc[r][c]);
                            }
                        }
                    }
                });
        });
    }

    template <typename InputT, typename OutputT>
    void forward_standard(s::buffer<InputT> &X, s::buffer<OutputT> &Y, size_t M)
    {
        size_t K = input_dim_;
        size_t N = output_dim_;
        queue_.submit([&](s::handler &h) {
            auto x = X.template get_access<s::access::mode::read>(h);
            auto y = Y.template get_access<s::access::mode::write>(h);
            auto w = W_packed_.template get_access<s::access::mode::read>(h);
            h.parallel_for(s::range<2>(M, N), [=](s::id<2> idx) {
                size_t m = idx[0];
                size_t n = idx[1];
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k)
                {
                    size_t w_idx = k * N + n;
                    uint8_t packed = w[w_idx / 4];
                    uint8_t bits = (packed >> ((w_idx % 4) * 2)) & 0x03;
                    float w_val = static_cast<float>(bits) - 1.0f;
                    sum += static_cast<float>(x[m * K + k]) * w_val;
                }
                y[m * N + n] = static_cast<OutputT>(sum);
            });
        });
    }
};
} // namespace layers
} // namespace neurobit
