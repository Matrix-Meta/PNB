#pragma once
#include <cstdint>
#include <sycl/sycl.hpp>
#include <vector>

namespace neurobit
{
namespace core
{
/**
 * Bit-Packing 權重壓縮
 *
 * 將三元權重 {-1, 0, +1} 壓縮為 2 bits
 * 4 個權重打包進 1 個 uint8_t
 *
 * 編碼:
 *   -1 -> 0b00
 *    0 -> 0b01
 *   +1 -> 0b10
 *
 * 打包格式 (little-endian):
 *   byte = w3w3_w2w2_w1w1_w0w0
 */
class BitPackedWeights
{
  public:
    // 編碼單個權重: {-1, 0, +1} -> {0, 1, 2}
    static constexpr uint8_t encode(int8_t weight)
    {
        return static_cast<uint8_t>(weight + 1);
    }

    // 解碼單個權重: {0, 1, 2} -> {-1, 0, +1}
    static constexpr int8_t decode(uint8_t encoded)
    {
        return static_cast<int8_t>(encoded) - 1;
    }

    // 打包 4 個權重到 1 byte
    static constexpr uint8_t pack4(int8_t w0, int8_t w1, int8_t w2, int8_t w3)
    {
        return (encode(w3) << 6) | (encode(w2) << 4) | (encode(w1) << 2) | (encode(w0));
    }

    // 從 packed byte 提取第 idx 個權重 (idx = 0-3)
    static constexpr int8_t extract(uint8_t packed, int idx)
    {
        return decode((packed >> (idx * 2)) & 0b11);
    }

    // 解包 4 個權重 (SYCL device 端使用)
    struct Unpacked4
    {
        int8_t w[4];

        constexpr int8_t operator[](int idx) const
        {
            return w[idx];
        }
    };

    static constexpr Unpacked4 unpack4(uint8_t packed)
    {
        return {decode(packed & 0b11), decode((packed >> 2) & 0b11), decode((packed >> 4) & 0b11),
                decode((packed >> 6) & 0b11)};
    }

    // Host 端批次打包
    static std::vector<uint8_t> pack_array(const std::vector<int8_t> &weights)
    {
        size_t packed_size = (weights.size() + 3) / 4;
        std::vector<uint8_t> packed(packed_size, 0);

        for (size_t i = 0; i < weights.size(); ++i)
        {
            size_t byte_idx = i / 4;
            size_t bit_idx = (i % 4) * 2;
            packed[byte_idx] |= (encode(weights[i]) << bit_idx);
        }

        return packed;
    }

    // Host 端批次解包 (用於驗證)
    static std::vector<int8_t> unpack_array(const std::vector<uint8_t> &packed, size_t original_size)
    {
        std::vector<int8_t> weights(original_size);

        for (size_t i = 0; i < original_size; ++i)
        {
            size_t byte_idx = i / 4;
            size_t bit_idx = (i % 4) * 2;
            weights[i] = decode((packed[byte_idx] >> bit_idx) & 0b11);
        }

        return weights;
    }

    // 獲取壓縮後大小
    static constexpr size_t packed_size(size_t original_size)
    {
        return (original_size + 3) / 4;
    }

    // 壓縮率
    static constexpr float compression_ratio()
    {
        return 4.0f; // 8 bits -> 2 bits
    }
};

} // namespace core
} // namespace neurobit
