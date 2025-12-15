#pragma once
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/sycl.hpp>

namespace neurobit
{
namespace core
{
namespace s = sycl;
using bfloat16 = s::ext::oneapi::bfloat16;

/**
 * XMX (Xe Matrix Extensions) 支援檢測與包裝
 *
 * Intel XMX 提供硬體加速的矩陣運算 (類似 NVIDIA Tensor Core)
 * 支援: BF16, FP16, INT8
 *
 * B580 (Battlemage): 完全支援 ✓
 * Arc A 系列 (Alchemist): 完全支援 ✓
 */
class XMXSupport
{
  public:
    /**
     * 檢測設備是否支援 XMX
     */
    static bool is_supported(const s::device &device)
    {
        try
        {
            // 檢查 1: Sub-group 大小
            auto sg_sizes = device.get_info<s::info::device::sub_group_sizes>();
            bool has_sg16 = std::find(sg_sizes.begin(), sg_sizes.end(), 16) != sg_sizes.end();

            if (!has_sg16)
                return false;

            // 檢查 2: 設備型號
            auto name = device.get_info<s::info::device::name>();

            // Battlemage (B 系列)
            if (name.find("Arc(TM) B") != std::string::npos)
            {
                return true;
            }

            // Alchemist (A 系列)
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

            // 其他 Intel GPU - 運行時測試
            auto vendor = device.get_info<s::info::device::vendor>();
            if (vendor.find("Intel") != std::string::npos)
            {
                return runtime_test_xmx(device);
            }

            return false;
        }
        catch (...)
        {
            return false;
        }
    }

    /**
     * 獲取推薦的 tile 大小
     * XMX 最佳尺寸: 16x16 或 8x32
     */
    static constexpr size_t get_tile_m()
    {
        return 16;
    }
    static constexpr size_t get_tile_n()
    {
        return 16;
    }
    static constexpr size_t get_tile_k()
    {
        return 16;
    }

    /**
     * 打印 XMX 支援信息
     */
    static void print_support_info(const s::device &device)
    {
        std::cout << "XMX Support Information:\n";
        std::cout << "  Device: " << device.get_info<s::info::device::name>() << "\n";

        auto sg_sizes = device.get_info<s::info::device::sub_group_sizes>();
        std::cout << "  Sub-group sizes: ";
        for (auto sz : sg_sizes)
            std::cout << sz << " ";
        std::cout << "\n";

        bool supported = is_supported(device);
        std::cout << "  XMX Supported: " << (supported ? "✓ YES" : "✗ NO") << "\n";

        if (supported)
        {
            std::cout << "  Optimal tile: " << get_tile_m() << "×" << get_tile_k() << " × " << get_tile_k() << "×"
                      << get_tile_n() << "\n";
            std::cout << "  Accelerated types: BF16, FP16, INT8\n";
        }
    }

  private:
    /**
     * 運行時測試 XMX 功能
     * 嘗試執行簡單的矩陣乘法
     */
    static bool runtime_test_xmx(const s::device &device)
    {
        // 由於 joint_matrix 需要編譯時支援，
        // 這裡簡化為檢查 sub-group 大小
        try
        {
            auto sg_sizes = device.get_info<s::info::device::sub_group_sizes>();
            return std::find(sg_sizes.begin(), sg_sizes.end(), 16) != sg_sizes.end();
        }
        catch (...)
        {
            return false;
        }
    }
};

/**
 * 自適應矩陣乘法
 * 自動選擇 XMX 或標準實現
 */
template <typename T> class AdaptiveMatMul
{
  private:
    bool use_xmx_;
    s::queue &queue_;

  public:
    explicit AdaptiveMatMul(s::queue &q) : queue_(q), use_xmx_(XMXSupport::is_supported(q.get_device()))
    {
    }

    /**
     * 矩陣乘法: C = A × B
     * A: M × K
     * B: K × N
     * C: M × N
     */
    void multiply(s::buffer<T> &A, s::buffer<T> &B, s::buffer<T> &C, size_t M, size_t K, size_t N)
    {
        if (use_xmx_)
        {
            multiply_xmx(A, B, C, M, K, N);
        }
        else
        {
            multiply_standard(A, B, C, M, K, N);
        }
    }

    bool is_using_xmx() const
    {
        return use_xmx_;
    }

  private:
    /**
     * XMX 加速實現
     * 使用 joint_matrix
     */
    void multiply_xmx(s::buffer<T> &A, s::buffer<T> &B, s::buffer<T> &C, size_t M, size_t K, size_t N)
    {
        // 注意: joint_matrix 需要 SYCL 2020 擴展
        // 這裡先使用優化的標準實現
        // TODO: 實現真正的 joint_matrix 版本
        multiply_standard(A, B, C, M, K, N);
    }

    /**
     * 標準實現 (回退)
     * 使用向量化和共享記憶體
     */
    void multiply_standard(s::buffer<T> &A, s::buffer<T> &B, s::buffer<T> &C, size_t M, size_t K, size_t N)
    {
        constexpr size_t TILE_SIZE = 16;

        queue_.submit([&](s::handler &h) {
            auto a = A.template get_access<s::access::mode::read>(h);
            auto b = B.template get_access<s::access::mode::read>(h);
            auto c = C.template get_access<s::access::mode::write>(h);

            // 使用 local memory
            s::local_accessor<T, 2> tile_a{{TILE_SIZE, TILE_SIZE}, h};
            s::local_accessor<T, 2> tile_b{{TILE_SIZE, TILE_SIZE}, h};

            size_t global_m = ((M + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
            size_t global_n = ((N + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;

            h.parallel_for(s::nd_range<2>{{global_m, global_n}, {TILE_SIZE, TILE_SIZE}}, [=](s::nd_item<2> it) {
                size_t row = it.get_global_id(0);
                size_t col = it.get_global_id(1);
                size_t local_row = it.get_local_id(0);
                size_t local_col = it.get_local_id(1);

                T sum = 0;

                // Tiled 矩陣乘法
                for (size_t t = 0; t < K; t += TILE_SIZE)
                {
                    // 載入 tile A
                    if (row < M && (t + local_col) < K)
                    {
                        tile_a[local_row][local_col] = a[row * K + t + local_col];
                    }
                    else
                    {
                        tile_a[local_row][local_col] = T(0);
                    }

                    // 載入 tile B
                    if ((t + local_row) < K && col < N)
                    {
                        tile_b[local_row][local_col] = b[(t + local_row) * N + col];
                    }
                    else
                    {
                        tile_b[local_row][local_col] = T(0);
                    }

                    it.barrier(s::access::fence_space::local_space);

// 計算
#pragma unroll
                    for (size_t k = 0; k < TILE_SIZE; ++k)
                    {
                        sum += tile_a[local_row][k] * tile_b[k][local_col];
                    }

                    it.barrier(s::access::fence_space::local_space);
                }

                if (row < M && col < N)
                {
                    c[row * N + col] = sum;
                }
            });
        });
    }
};

} // namespace core
} // namespace neurobit
