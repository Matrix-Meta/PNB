#pragma once

#ifndef NDEBUG
#define NDEBUG
#endif

// 確保這裡引用的是系統路徑 (配合 -I ./mdspan/include)
#include <experimental/mdspan>
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cmath>
#include <span>

namespace neurobit {

// 定義命名空間別名
namespace s = sycl;
namespace stdex = std::experimental;

// 定義核心型別
template <typename T>
using Vector1D = stdex::mdspan<
    T,
    stdex::extents<size_t, std::dynamic_extent>
>;

template <typename T>
using MatrixView = stdex::mdspan<
    T,
    stdex::extents<size_t, std::dynamic_extent, std::dynamic_extent>
>;

template <typename T>
using Tensor3D = stdex::mdspan<
    T,
    stdex::extents<size_t, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>
>;

using DefaultFloat = float;

// 常用數學常數
constexpr float PI = 3.14159265358979323846f;
constexpr float EPSILON = 1e-6f;

// 安全的 tanh 約束函數 (用於 SSM 穩定性)
inline float stable_coeff(float x) {
    return std::tanh(x);
}

} // namespace neurobit
