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
#include <cmath>
#include <sycl/sycl.hpp>

namespace neurobit
{
namespace core
{
namespace s = sycl;

/**
 * 自適應批次調度器
 *
 * 目標:
 * 1. 飽和所有 Xe-Cores
 * 2. 最大化 L2 快取命中率
 * 3. 平衡記憶體與計算
 * 4. XMX 友好 (16 的倍數)
 */
class AdaptiveBatchScheduler
{
  private:
    size_t xe_cores_;
    size_t max_memory_;
    size_t local_memory_;
    size_t max_work_group_size_;

    static constexpr size_t L2_CACHE_SIZE = 18 * 1024 * 1024; // B580: 18 MB
    static constexpr size_t XMX_TILE_SIZE = 16;

  public:
    /**
     * 構造函數 - 從設備獲取信息
     */
    explicit AdaptiveBatchScheduler(const s::device &device)
    {
        xe_cores_ = device.get_info<s::info::device::max_compute_units>();
        max_memory_ = device.get_info<s::info::device::global_mem_size>();
        local_memory_ = device.get_info<s::info::device::local_mem_size>();
        max_work_group_size_ = device.get_info<s::info::device::max_work_group_size>();
    }

    /**
     * 計算最優批次大小
     *
     * @param input_dim 輸入維度
     * @param hidden_dim 隱藏維度
     * @param precision_bytes 精度字節數 (2 for BF16, 4 for FP32)
     * @return 最優批次大小
     */
    size_t compute_optimal_batch(size_t input_dim, size_t hidden_dim,
                                 size_t precision_bytes = 2 // BF16
    ) const
    {
        // 目標 1: 飽和所有 Xe-Cores
        // 每個 Xe-Core 至少處理 16 個元素
        size_t min_batch_for_saturation = xe_cores_ * 16;

        // 目標 2: 記憶體限制
        // 考慮輸入、輸出、權重、狀態
        size_t memory_per_sample = (input_dim + hidden_dim * 2) * precision_bytes; // I/O + state

        size_t weight_memory = hidden_dim * input_dim / 4; // Bit-packed

        size_t available_memory = static_cast<size_t>(max_memory_ * 0.7); // 70% 安全邊界
        size_t max_batch_by_memory = (available_memory - weight_memory) / memory_per_sample;

        // 目標 3: L2 快取友好
        // 希望激活數據能放入 L2
        size_t activation_per_sample = hidden_dim * precision_bytes;
        size_t max_batch_for_l2 = L2_CACHE_SIZE / activation_per_sample;

        // 目標 4: Work-group 平衡
        size_t elements_per_batch = hidden_dim;
        size_t work_groups_needed = (elements_per_batch + max_work_group_size_ - 1) / max_work_group_size_;
        size_t optimal_for_work_groups = std::max(xe_cores_ / work_groups_needed, size_t(1)) * 16;

        // 選擇最優值
        size_t optimal = std::min({size_t(256), // 上限
                                   max_batch_by_memory,
                                   std::max({min_batch_for_saturation,
                                             max_batch_for_l2 / 2, // L2 保守估計
                                             optimal_for_work_groups})});

        // 向上取整到 XMX_TILE_SIZE 的倍數 (利於 XMX)
        optimal = ((optimal + XMX_TILE_SIZE - 1) / XMX_TILE_SIZE) * XMX_TILE_SIZE;

        // 最小批次
        return std::max(optimal, XMX_TILE_SIZE);
    }

    /**
     * 計算最優 work-group 大小
     */
    size_t compute_work_group_size(size_t global_size) const
    {
        if (global_size >= xe_cores_ * 64)
        {
            return std::min(size_t(512), max_work_group_size_);
        }
        else if (global_size >= xe_cores_ * 16)
        {
            return std::min(size_t(256), max_work_group_size_);
        }
        else
        {
            return std::min(size_t(128), max_work_group_size_);
        }
    }

    /**
     * 計算最優 tile 大小 (用於矩陣乘法)
     */
    size_t compute_tile_size(size_t matrix_dim) const
    {
        // 基於 local memory 大小
        // Local memory per Xe-Core: 128 KB (B580)

        // 每個 tile 需要: 2 * tile_size^2 * precision_bytes
        size_t max_tile_from_memory = static_cast<size_t>(std::sqrt(local_memory_ / 4)); // 保守估計

        // XMX 友好的 tile 大小
        std::vector<size_t> xmx_tiles = {8, 16, 32, 64};

        for (auto it = xmx_tiles.rbegin(); it != xmx_tiles.rend(); ++it)
        {
            if (*it <= max_tile_from_memory && *it <= matrix_dim)
            {
                return *it;
            }
        }

        return 16; // 默認
    }

    /**
     * 檢查是否應該使用 XMX
     */
    bool should_use_xmx(size_t M, size_t K, size_t N, size_t tile_m = 8, size_t tile_k = 16, size_t tile_n = 16) const
    {
        // XMX 對齊要求
        bool is_aligned = (M % tile_m == 0) && (K % tile_k == 0) && (N % tile_n == 0);

        // 矩陣足夠大
        size_t total_ops = M * K * N;
        bool is_large_enough = total_ops >= (1024 * 1024); // 至少 1M ops

        return is_aligned && is_large_enough;
    }

    /**
     * 打印調度信息
     */
    void print_schedule_info(size_t input_dim, size_t hidden_dim, size_t precision_bytes = 2) const
    {
        auto batch = compute_optimal_batch(input_dim, hidden_dim, precision_bytes);
        auto wg_size = compute_work_group_size(hidden_dim);
        auto tile = compute_tile_size(hidden_dim);

        std::cout << "╔════════════════════════════════════════════════════════╗
";
        std::cout << "║         自適應批次調度信息                            ║
";
        std::cout << "╚════════════════════════════════════════════════════════╝

";

        std::cout << "設備資源:
";
        std::cout << "  Xe-Cores:        " << xe_cores_ << "
";
        std::cout << "  全局記憶體:      " << (max_memory_ / (1024 * 1024 * 1024)) << " GB
";
        std::cout << "  L2 快取:         " << (L2_CACHE_SIZE / (1024 * 1024)) << " MB
";
        std::cout << "  Local 記憶體:    " << (local_memory_ / 1024) << " KB
";
        std::cout << "  Max work-group:  " << max_work_group_size_ << "

";

        std::cout << "模型配置:
";
        std::cout << "  輸入維度:        " << input_dim << "
";
        std::cout << "  隱藏維度:        " << hidden_dim << "
";
        std::cout << "  精度:            " << (precision_bytes == 2 ? "BF16" : "FP32") << "

";

        std::cout << "調度決策:
";
        std::cout << "  最優批次:        " << batch << " ✓
";
        std::cout << "  Work-group 大小: " << wg_size << "
";
        std::cout << "  Tile 大小:       " << tile << " × " << tile << "
";
        std::cout << "  XMX 對齊:        " << (batch % 16 == 0 ? "✓ YES" : "✗ NO") << "

";

        // 性能估計
        size_t total_threads = batch * hidden_dim;
        size_t work_groups = (total_threads + wg_size - 1) / wg_size;
        float saturation = static_cast<float>(work_groups) / xe_cores_;

        std::cout << "性能估計:
";
        std::cout << "  總線程數:        " << total_threads << "
";
        std::cout << "  Work-groups:     " << work_groups << "
";
        std::cout << "  Xe-Core 飽和度:  " << (saturation * 100) << "% ";
        if (saturation >= 1.0)
            std::cout << "✓ 飽和
";
        else
            std::cout << "⚠️  未飽和
";

        std::cout << "
";
    }

    /**
     * 獲取設備信息
     */
    size_t get_xe_cores() const
    {
        return xe_cores_;
    }
    size_t get_max_memory() const
    {
        return max_memory_;
    }
    size_t get_local_memory() const
    {
        return local_memory_;
    }
};

} // namespace core
} // namespace neurobit
