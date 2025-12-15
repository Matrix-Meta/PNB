#pragma once
#include <sycl/sycl.hpp>

namespace neurobit
{
namespace derivatives
{

// =================================================================
// 1. Pade-Optimized Algebraic Sigmoid Derivative (The "Extreme" One)
// =================================================================
// Target: f'(x) = (1 + x^2)^(-1.5)
// Approx: Pade[0/2] = 1 / (1 + 1.5 * x^2)
//
// 優勢：
// 1. 極速：只用 FMA (Fused Multiply-Add) 和 Div。
// 2. 物理性：比 Sigmoid 更接近能量分佈。
// 3. 穩定：沒有絕對值運算的奇點 (Singularity at 0)，導數在 0 處光滑。
template <typename T> inline T pade_algebraic_sigmoid(T x, float scale = 1.0f)
{
    // 強制使用 float 進行中間計算，避免 BF16/FP16 的反覆類型轉換開銷
    float x_f = static_cast<float>(x);
    float x_scaled = x_f * scale;

    // Pade [0/2] Approximation for (1+x^2)^-1.5
    // Coeff 1.5 is derived from Taylor expansion matching
    return static_cast<T>(1.0f / (1.0f + 1.5f * x_scaled * x_scaled));
}

// =================================================================
// 2. Higher-Order Pade (Better Tail) [Optional]
// =================================================================
// 如果發現上面的尾部衰減太快 (x^-2)，可以用這個 [2/2] 版本
// Target Tail: x^-3
// Approx: (1 + 0.5 x^2) / (1 + 2 x^2 + 0.5 x^4) -> x^-2... 還是很難完美擬合 x^-3
// 但通常上面的 [0/2] 對於訓練來說已經足夠好了 (Gradient Flow 夠強)

// =================================================================
// 3. Standard Fast Sigmoid (For Comparison)
// =================================================================
// f'(x) = 1 / (1 + |x|)^2
template <typename T> inline T fast_sigmoid(T x, float scale = 1.0f)
{
    float x_f = static_cast<float>(x);
    float abs_x = sycl::fabs(x_f * scale);
    float denom = 1.0f + abs_x;
    return static_cast<T>(1.0f / (denom * denom));
}

} // namespace derivatives
} // namespace neurobit