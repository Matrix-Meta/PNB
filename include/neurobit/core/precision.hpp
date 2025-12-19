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
#include <sycl/sycl.hpp>
#include <type_traits>

namespace neurobit
{
namespace core
{
/**
 * 混合精度支援：自動選擇 BF16 或 FP16
 *
 * B580 (Xe-HPG Battlemage) 支援 BF16 硬體加速
 * 其他設備回退到 FP16
 *
 * 使用方式:
 *   using ComputeType = PrecisionSelector::SelectType;
 *   // 自動選擇 BF16 (B580) 或 FP16 (其他)
 */

namespace s = sycl;

// BF16 類型 - 使用 Intel oneAPI 擴展
#include <sycl/ext/oneapi/bfloat16.hpp>
using bfloat16 = sycl::ext::oneapi::bfloat16;

/**
 * 精度選擇器 - 自動檢測設備能力
 */
class PrecisionSelector
{
  public:
    enum class Precision
    {
        FP32,
        FP16,
        BF16
    };

    /**
     * 檢測設備支援的最佳精度
     */
    static Precision detect_best_precision(const s::queue &q)
    {
        auto device = q.get_device();

        // 檢查是否支援 BF16
        if (supports_bfloat16(device))
        {
            return Precision::BF16;
        }

        // 檢查是否支援 FP16
        if (supports_half(device))
        {
            return Precision::FP16;
        }

        // 回退到 FP32
        return Precision::FP32;
    }

    /**
     * 檢測是否為 Intel GPU
     */
    static bool is_intel_gpu(const s::device &device)
    {
        auto vendor = device.get_info<s::info::device::vendor>();
        return vendor.find("Intel") != std::string::npos;
    }

    /**
     * 檢測是否為 B580 或 Battlemage 系列
     */
    static bool is_battlemage(const s::device &device)
    {
        if (!is_intel_gpu(device))
            return false;

        auto name = device.get_info<s::info::device::name>();
        // B580, B770 等都是 Battlemage
        return name.find("Arc(TM) B") != std::string::npos || name.find("Battlemage") != std::string::npos;
    }

    /**
     * 檢測是否支援 BF16
     * 使用運行時測試確保跨平台兼容性
     */
    static bool supports_bfloat16(const s::device &device)
    {
        // 方法 1: 快速路徑 - 已知支援的設備
        if (is_battlemage(device))
        {
            return true; // B580 等 Battlemage 系列
        }

        if (is_intel_gpu(device))
        {
            auto name = device.get_info<s::info::device::name>();
            // Arc A 系列 (Alchemist)
            if (name.find("Arc(TM) A") != std::string::npos)
            {
                return true;
            }
            // Flex 系列
            if (name.find("Flex") != std::string::npos)
            {
                return true;
            }
            // Data Center GPU Max
            if (name.find("Max") != std::string::npos)
            {
                return true;
            }
        }

        // 方法 2: 運行時測試 - 嘗試執行 BF16 kernel
        // 適用於未知或新設備
        return runtime_test_bfloat16(device);
    }

    /**
     * 運行時測試 BF16 支援
     * 嘗試執行簡單的 BF16 kernel
     */
    static bool runtime_test_bfloat16(const s::device &device)
    {
        try
        {
            s::queue q(device);

            // 創建測試 buffer
            std::vector<float> test_data = {1.0f, 2.0f, 3.0f};
            s::buffer<float> buf_in{test_data};
            s::buffer<float> buf_out{3};

            // 嘗試執行 BF16 轉換
            q.submit([&](s::handler &h) {
                auto in = buf_in.get_access<s::access::mode::read>(h);
                auto out = buf_out.get_access<s::access::mode::write>(h);

                h.parallel_for(s::range<1>(3), [=](s::id<1> i) {
                    // 嘗試使用 bfloat16
                    bfloat16 bf = bfloat16(in[i]);
                    out[i] = static_cast<float>(bf);
                });
            });

            q.wait();

            // 驗證結果
            auto result = buf_out.get_host_access();
            for (size_t i = 0; i < 3; ++i)
            {
                if (std::abs(result[i] - test_data[i]) > 0.1f)
                {
                    return false; // 結果錯誤
                }
            }

            return true; // BF16 可用
        }
        catch (...)
        {
            return false; // BF16 不可用
        }
    }

    /**
     * 檢測是否支援 FP16
     */
    static bool supports_half(const s::device &device)
    {
        return device.has(s::aspect::fp16);
    }

    /**
     * 獲取精度名稱
     */
    static const char *precision_name(Precision p)
    {
        switch (p)
        {
        case Precision::BF16:
            return "BF16";
        case Precision::FP16:
            return "FP16";
        case Precision::FP32:
            return "FP32";
        default:
            return "Unknown";
        }
    }

    /**
     * 打印設備精度支援信息
     */
    static void print_precision_support(const s::queue &q)
    {
        auto device = q.get_device();
        auto best = detect_best_precision(q);

        std::cout << "Device: " << device.get_info<s::info::device::name>() << "
";
        std::cout << "Precision Support:
";
        std::cout << "  FP32:  ✓ (always supported)
";
        std::cout << "  FP16:  " << (supports_half(device) ? "✓" : "✗") << "
";
        std::cout << "  BF16:  " << (supports_bfloat16(device) ? "✓" : "✗")
                  << (is_battlemage(device) ? " (Battlemage HW accelerated)" : "") << "
";
        std::cout << "Selected: " << precision_name(best) << "
";
    }
};

/**
 * 混合精度包裝器
 *
 * 使用方式:
 *   MixedPrecision<AutoSelect> mp(queue);
 *   using T = mp.compute_type;  // 自動選擇 BF16/FP16
 */
template <typename PrecisionPolicy = struct AutoSelect> class MixedPrecision
{
  public:
    using storage_type = float; // Host 端儲存永遠是 FP32

    // 計算類型根據設備自動選擇
    // BF16 在 Intel GPU 上總是可用（編譯時）
    using compute_type =
        std::conditional_t<std::is_same_v<PrecisionPolicy, struct ForceBF16>, bfloat16,
                           std::conditional_t<std::is_same_v<PrecisionPolicy, struct ForceFP16>, s::half,
                                              bfloat16 // AutoSelect 默認 BF16
                                              >>;

  private:
    s::queue &queue_;
    PrecisionSelector::Precision precision_;

  public:
    explicit MixedPrecision(s::queue &q) : queue_(q), precision_(PrecisionSelector::detect_best_precision(q))
    {
    }

    /**
     * FP32 -> Compute Type 轉換
     */
    template <typename T = compute_type> void convert_to_compute(s::buffer<storage_type, 1> &src, s::buffer<T, 1> &dst)
    {
        size_t N = src.size();
        queue_.submit([&](s::handler &h) {
            auto in = src.template get_access<s::access::mode::read>(h);
            auto out = dst.template get_access<s::access::mode::write>(h);
            h.parallel_for(s::range<1>(N), [=](s::id<1> i) { out[i] = static_cast<T>(in[i]); });
        });
    }

    /**
     * Compute Type -> FP32 轉換
     */
    template <typename T = compute_type>
    void convert_from_compute(s::buffer<T, 1> &src, s::buffer<storage_type, 1> &dst)
    {
        size_t N = src.size();
        queue_.submit([&](s::handler &h) {
            auto in = src.template get_access<s::access::mode::read>(h);
            auto out = dst.template get_access<s::access::mode::write>(h);
            h.parallel_for(s::range<1>(N), [=](s::id<1> i) { out[i] = static_cast<storage_type>(in[i]); });
        });
    }

    /**
     * 獲取當前精度
     */
    PrecisionSelector::Precision get_precision() const
    {
        return precision_;
    }

    /**
     * 獲取精度名稱
     */
    const char *get_precision_name() const
    {
        return PrecisionSelector::precision_name(precision_);
    }

    /**
     * 是否使用降低精度 (非 FP32)
     */
    bool is_reduced_precision() const
    {
        return precision_ != PrecisionSelector::Precision::FP32;
    }
};

// 策略標籤
struct AutoSelect
{
}; // 自動選擇 (BF16 > FP16 > FP32)
struct ForceBF16
{
}; // 強制 BF16
struct ForceFP16
{
}; // 強制 FP16
struct ForceFP32
{
}; // 強制 FP32

/**
 * 簡化的類型選擇器（編譯時）
 */
template <typename Device> struct ComputeTypeSelector
{
    using type = bfloat16; // Intel GPU 默認 BF16
};

// 便捷別名 - 運行時選擇，編譯時默認 BF16
using DefaultComputeType = bfloat16;

/**
 * 運行時精度選擇器
 * 根據設備能力自動選擇最佳精度類型
 */
class RuntimePrecisionSelector
{
  private:
    PrecisionSelector::Precision precision_;

  public:
    explicit RuntimePrecisionSelector(const s::queue &q) : precision_(PrecisionSelector::detect_best_precision(q))
    {
    }

    /**
     * 執行帶自動精度的 kernel
     * 根據運行時檢測結果選擇合適的精度
     */
    template <typename Kernel> void execute(s::queue &q, Kernel &&kernel)
    {
        switch (precision_)
        {
        case PrecisionSelector::Precision::BF16:
            kernel.template operator()<bfloat16>(q);
            break;
        case PrecisionSelector::Precision::FP16:
            kernel.template operator()<s::half>(q);
            break;
        case PrecisionSelector::Precision::FP32:
        default:
            kernel.template operator()<float>(q);
            break;
        }
    }

    PrecisionSelector::Precision get_precision() const
    {
        return precision_;
    }

    const char *get_precision_name() const
    {
        return PrecisionSelector::precision_name(precision_);
    }
};

} // namespace core
} // namespace neurobit
