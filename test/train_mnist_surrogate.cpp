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

#include "neurobit/core/types.hpp"
#include "neurobit/layers/bit_brain.hpp"
#include "pnb/min_toml.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

namespace s = sycl;

// =================================================================
// Data Loading (Reuse)
// =================================================================

struct MNISTData
{
    std::vector<std::vector<float>> images;
    std::vector<int> labels;
};

static void write_image_to_batch(float *dst, const std::vector<float> &src, bool normalize, float mean, float inv_std,
                                 bool augment_shift, int max_shift, std::mt19937 &rng)
{
    if (!dst)
        return;
    if (src.size() != 28u * 28u)
        return;

    int dx = 0;
    int dy = 0;
    if (augment_shift && max_shift > 0)
    {
        std::uniform_int_distribution<int> dist(-max_shift, max_shift);
        dx = dist(rng);
        dy = dist(rng);
    }

    for (int r = 0; r < 28; ++r)
    {
        for (int c = 0; c < 28; ++c)
        {
            int rr = r - dy;
            int cc = c - dx;
            float v = 0.0f;
            if (rr >= 0 && rr < 28 && cc >= 0 && cc < 28)
                v = src[static_cast<size_t>(rr) * 28u + static_cast<size_t>(cc)];
            if (normalize)
                v = (v - mean) * inv_std;
            dst[static_cast<size_t>(r) * 28u + static_cast<size_t>(c)] = v;
        }
    }
}

uint32_t swap_endian(uint32_t val)
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

void load_mnist(const std::string &image_path, const std::string &label_path, MNISTData &data)
{
    std::ifstream img_file(image_path, std::ios::binary);
    if (!img_file)
        throw std::runtime_error("無法開啟影像檔案：" + image_path);

    uint32_t magic = 0, num_imgs = 0, rows = 0, cols = 0;
    img_file.read((char *)&magic, 4);
    img_file.read((char *)&num_imgs, 4);
    img_file.read((char *)&rows, 4);
    img_file.read((char *)&cols, 4);

    magic = swap_endian(magic);
    num_imgs = swap_endian(num_imgs);
    rows = swap_endian(rows);
    cols = swap_endian(cols);

    std::cout << "載入" << num_imgs << "張影像[" << rows << "x" << cols << "]，來源：" << image_path << "...
";

    size_t image_size = rows * cols;
    data.images.resize(num_imgs, std::vector<float>(image_size));

    std::vector<unsigned char> raw_pixels(num_imgs * image_size);
    img_file.read((char *)raw_pixels.data(), raw_pixels.size());

    for (size_t i = 0; i < num_imgs; ++i)
    {
        for (size_t p = 0; p < image_size; ++p)
        {
            data.images[i][p] = raw_pixels[i * image_size + p] / 255.0f;
        }
    }

    std::ifstream lbl_file(label_path, std::ios::binary);
    if (!lbl_file)
        throw std::runtime_error("無法開啟標籤檔案：" + label_path);

    uint32_t magic_l = 0, num_lbls = 0;
    lbl_file.read((char *)&magic_l, 4);
    lbl_file.read((char *)&num_lbls, 4);
    num_lbls = swap_endian(num_lbls);

    if (num_imgs != num_lbls)
        throw std::runtime_error("影像/標籤數量不一致！");

    data.labels.resize(num_lbls);
    std::vector<unsigned char> raw_labels(num_lbls);
    lbl_file.read((char *)raw_labels.data(), num_lbls);

    for (size_t i = 0; i < num_lbls; ++i)
    {
        data.labels[i] = static_cast<int>(raw_labels[i]);
    }
}

// =================================================================
// Main
// =================================================================

struct BioLrScheduler
{
    float min_lr = 1e-5f;
    float max_lr = 0.02f;
    float plateau_decay_base = 0.96f;
    float grow_base = 1.03f;
    int patience_epochs = 8;

    float ema_alpha = 0.15f;
    float var_alpha = 0.05f;
    float k_sigma = 0.25f;
    float min_delta = 2e-4f;

    bool initialized = false;
    float loss_ema = 0.0f;
    float loss_var_ema = 0.0f;
    float prev_loss_ema = 0.0f;
    float last_threshold = 0.0f;
    int plateau_count = 0;
    float last_plasticity = 0.0f;
    float last_factor = 1.0f;

    void reset()
    {
        initialized = false;
        loss_ema = 0.0f;
        loss_var_ema = 0.0f;
        prev_loss_ema = 0.0f;
        last_threshold = min_delta;
        plateau_count = 0;
        last_plasticity = 0.0f;
        last_factor = 1.0f;
    }

    bool update(float epoch_loss, float uncertainty, float familiarity, float active_rate, float target_active,
                float &lr)
    {
        if (!initialized)
        {
            initialized = true;
            loss_ema = epoch_loss;
            prev_loss_ema = epoch_loss;
            loss_var_ema = 0.0f;
            last_threshold = min_delta;
            plateau_count = 0;
            last_plasticity = 0.0f;
            last_factor = 1.0f;
            return false;
        }

        prev_loss_ema = loss_ema;
        loss_ema = (1.0f - ema_alpha) * loss_ema + ema_alpha * epoch_loss;

        float diff = epoch_loss - loss_ema;
        loss_var_ema = (1.0f - var_alpha) * loss_var_ema + var_alpha * (diff * diff);

        float sigma = std::sqrt(std::max(loss_var_ema, 0.0f));
        last_threshold = std::max(min_delta, k_sigma * sigma);

        float improvement = prev_loss_ema - loss_ema;
        float novelty = 1.0f - std::clamp(familiarity, 0.0f, 1.0f);
        float unc = std::clamp(uncertainty, 0.0f, 1.0f);
        float homeo_err = std::fabs(std::clamp(active_rate, 0.0f, 1.0f) - std::clamp(target_active, 0.0f, 1.0f));
        float homeo = 1.0f - std::clamp(homeo_err / std::max(target_active, 1e-3f), 0.0f, 1.0f);

        float plasticity = std::clamp(0.85f * unc + 0.15f * novelty, 0.0f, 1.0f);
        plasticity *= (0.5f + 0.5f * homeo);
        last_plasticity = plasticity;

        float factor = 1.0f;
        if (improvement < -last_threshold)
        {
            factor = 0.85f;
            plateau_count = 0;
        }
        else if (improvement < last_threshold)
        {
            plateau_count++;
            if (plateau_count >= patience_epochs)
            {
                if (plasticity > 0.12f)
                {
                    factor = 1.0f + 0.01f * plasticity;
                }
                else
                {
                    float strength = 1.0f + (1.0f - plasticity);
                    factor = std::pow(std::clamp(plateau_decay_base, 0.5f, 0.999f), strength);
                }
                plateau_count = 0;
            }
        }
        else if (improvement > (2.0f * last_threshold))
        {
            factor = 1.0f + (grow_base - 1.0f) * plasticity;
            plateau_count = 0;
        }
        else
        {
            plateau_count = 0;
        }

        last_factor = factor;
        lr = std::clamp(lr * factor, min_lr, max_lr);
        return false;
    }
};

struct EventTiming
{
    uint64_t sum_ns = 0;
    uint64_t span_ns = 0;
    size_t count = 0;
};

static EventTiming timing_from_events(std::vector<s::event> &events)
{
    EventTiming out;
    uint64_t min_start = std::numeric_limits<uint64_t>::max();
    uint64_t max_end = 0;

    for (auto &ev : events)
    {
        ev.wait();
        try
        {
            uint64_t st = ev.get_profiling_info<s::info::event_profiling::command_start>();
            uint64_t en = ev.get_profiling_info<s::info::event_profiling::command_end>();
            if (en > st)
            {
                out.sum_ns += (en - st);
                if (st < min_start)
                    min_start = st;
                if (en > max_end)
                    max_end = en;
                out.count++;
            }
        }
        catch (...)
        {
        }
    }

    if (out.count > 0 && max_end > min_start)
        out.span_ns = max_end - min_start;
    return out;
}

template <class T> static bool write_pod(std::ostream &os, const T &v)
{
    os.write(reinterpret_cast<const char *>(&v), sizeof(T));
    return static_cast<bool>(os);
}

static bool write_bytes(std::ostream &os, const void *ptr, size_t bytes)
{
    if (bytes == 0)
        return true;
    os.write(reinterpret_cast<const char *>(ptr), static_cast<std::streamsize>(bytes));
    return static_cast<bool>(os);
}

static bool write_vec_f32(std::ostream &os, const std::vector<float> &v)
{
    uint64_t n = static_cast<uint64_t>(v.size());
    if (!write_pod(os, n))
        return false;
    return write_bytes(os, v.data(), static_cast<size_t>(n) * sizeof(float));
}

template <class T> static bool read_pod(std::istream &is, T &v)
{
    is.read(reinterpret_cast<char *>(&v), sizeof(T));
    return static_cast<bool>(is);
}

static bool read_bytes(std::istream &is, void *ptr, size_t bytes)
{
    if (bytes == 0)
        return true;
    is.read(reinterpret_cast<char *>(ptr), static_cast<std::streamsize>(bytes));
    return static_cast<bool>(is);
}

static bool read_vec_f32(std::istream &is, std::vector<float> &v)
{
    uint64_t n = 0;
    if (!read_pod(is, n))
        return false;
    if (n > (std::numeric_limits<size_t>::max() / sizeof(float)))
        return false;
    v.resize(static_cast<size_t>(n));
    return read_bytes(is, v.data(), static_cast<size_t>(n) * sizeof(float));
}

struct CheckpointMeta
{
    uint32_t epoch = 0;
    uint32_t seed = 0;
    float lr = 0.0f;

    float train_acc = 0.0f;
    float train_loss = 0.0f;
    float train_unc = 0.0f;

    bool has_val = false;
    float val_acc = 0.0f;
    float val_loss = 0.0f;
    float val_unc = 0.0f;
};

static bool load_checkpoint_file(const std::string &path, neurobit::layers::BitBrainLayer::CheckpointState &st,
                                 CheckpointMeta &meta, std::string *error = nullptr)
{
    auto fail = [&](const std::string &msg) {
        if (error)
            *error = msg;
        return false;
    };

    std::ifstream is(path, std::ios::binary);
    if (!is)
        return fail("無法開啟檢查點檔案");

    char magic[8] = {};
    if (!read_bytes(is, magic, sizeof(magic)))
        return fail("讀取檢查點magic失敗");
    const char expect[8] = {'P', 'N', 'B', 'C', 'K', 'P', 'T', '1'};
    if (!std::equal(std::begin(magic), std::end(magic), std::begin(expect)))
        return fail("檢查點magic不匹配");

    uint32_t version = 0;
    if (!read_pod(is, version))
        return fail("讀取檢查點版本失敗");
    if (version != 1)
        return fail("不支援的檢查點版本");

    uint64_t batch_size = 0, seq_len = 0, input_dim = 0, hidden_dim = 0, output_dim = 0, md_group_size = 0,
             md_group_count = 0;
    uint32_t epoch = 0, seed = 0, flags = 0;
    float lr = 0.0f, train_acc = 0.0f, train_loss = 0.0f, train_unc = 0.0f, val_acc = 0.0f, val_loss = 0.0f,
          val_unc = 0.0f;

    if (!read_pod(is, batch_size) || !read_pod(is, seq_len) || !read_pod(is, input_dim) || !read_pod(is, hidden_dim) ||
        !read_pod(is, output_dim) || !read_pod(is, md_group_size) || !read_pod(is, md_group_count) ||
        !read_pod(is, epoch) || !read_pod(is, seed) || !read_pod(is, lr) || !read_pod(is, train_acc) ||
        !read_pod(is, train_loss) || !read_pod(is, train_unc) || !read_pod(is, val_acc) || !read_pod(is, val_loss) ||
        !read_pod(is, val_unc) || !read_pod(is, flags))
    {
        return fail("讀取檢查點Header失敗");
    }

    st = neurobit::layers::BitBrainLayer::CheckpointState{};
    st.version = version;
    st.batch_size = static_cast<size_t>(batch_size);
    st.seq_len = static_cast<size_t>(seq_len);
    st.input_dim = static_cast<size_t>(input_dim);
    st.hidden_dim = static_cast<size_t>(hidden_dim);
    st.output_dim = static_cast<size_t>(output_dim);
    st.md_group_size = static_cast<size_t>(md_group_size);
    st.md_group_count = static_cast<size_t>(md_group_count);

    meta.epoch = epoch;
    meta.seed = seed;
    meta.lr = lr;
    meta.train_acc = train_acc;
    meta.train_loss = train_loss;
    meta.train_unc = train_unc;
    meta.val_acc = val_acc;
    meta.val_loss = val_loss;
    meta.val_unc = val_unc;
    meta.has_val = ((flags & 1u) != 0u);

    auto &gs = st.glial_state;
    if (!read_pod(is, gs.current_threshold) || !read_pod(is, gs.current_lr) || !read_pod(is, gs.prev_error) ||
        !read_pod(is, gs.error_integral) || !read_pod(is, gs.threshold_velocity) || !read_pod(is, gs.sparsity_ema) ||
        !read_pod(is, gs.error_ema) || !read_pod(is, gs.last_change_ratio) || !read_pod(is, gs.stable_count) ||
        !read_pod(is, gs.is_first_call) || !read_pod(is, gs.current_noise_gain) || !read_pod(is, gs.last_sparsity) ||
        !read_pod(is, st.target_sparsity) || !read_pod(is, st.glial_threshold_snapshot) ||
        !read_pod(is, st.hippocampus_learned_threshold))
    {
        return fail("讀取檢查點Glial/Hippocampus狀態失敗");
    }

    for (float &w : st.neuromodulator_w)
    {
        if (!read_pod(is, w))
            return fail("讀取檢查點Neuromodulator權重失敗");
    }

    auto &nm = st.neuromodulator_last;
    if (!read_pod(is, nm.target_sparsity) || !read_pod(is, nm.vigilance) || !read_pod(is, nm.noise_gain) ||
        !read_pod(is, nm.md_silu_tau) || !read_pod(is, nm.md_silu_alpha) || !read_pod(is, nm.md_silu_novelty_power) ||
        !read_pod(is, nm.md_silu_novelty_sharpness))
    {
        return fail("讀取檢查點Neuromodulator狀態失敗");
    }

    if (!read_pod(is, st.md_vigilance) || !read_pod(is, st.md_tau) || !read_pod(is, st.md_alpha) ||
        !read_pod(is, st.md_novelty_power) || !read_pod(is, st.md_novelty_sharpness))
    {
        return fail("讀取檢查點M-DSiLU參數失敗");
    }

    uint8_t use_bias_flags = 0;
    if (!read_pod(is, use_bias_flags))
        return fail("讀取檢查點bias旗標失敗");
    st.proj_in_use_bias = ((use_bias_flags & (1u << 0)) != 0u);
    st.proj_mid1_use_bias = ((use_bias_flags & (1u << 1)) != 0u);
    st.proj_mid2_use_bias = ((use_bias_flags & (1u << 2)) != 0u);
    st.proj_out_use_bias = ((use_bias_flags & (1u << 3)) != 0u);

    if (!read_vec_f32(is, st.md_group_offsets) || !read_vec_f32(is, st.proj_in_weight) ||
        !read_vec_f32(is, st.proj_in_bias) || !read_vec_f32(is, st.proj_mid1_weight) ||
        !read_vec_f32(is, st.proj_mid1_bias) || !read_vec_f32(is, st.proj_mid2_weight) ||
        !read_vec_f32(is, st.proj_mid2_bias) || !read_vec_f32(is, st.proj_out_weight) ||
        !read_vec_f32(is, st.proj_out_bias) || !read_vec_f32(is, st.w_fast) || !read_vec_f32(is, st.w_slow) ||
        !read_vec_f32(is, st.ssm_A) || !read_vec_f32(is, st.ssm_state))
    {
        return fail("讀取檢查點向量失敗");
    }

    return true;
}

static bool save_checkpoint_file(const std::string &path, const neurobit::layers::BitBrainLayer::CheckpointState &st,
                                 uint32_t epoch, uint32_t seed, float lr, float train_acc, float train_loss,
                                 float train_unc, bool has_val, float val_acc, float val_loss, float val_unc)
{
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os)
        return false;

    const char magic[8] = {'P', 'N', 'B', 'C', 'K', 'P', 'T', '1'};
    if (!write_bytes(os, magic, sizeof(magic)))
        return false;
    uint32_t version = 1;
    if (!write_pod(os, version))
        return false;

    uint64_t batch_size = static_cast<uint64_t>(st.batch_size);
    uint64_t seq_len = static_cast<uint64_t>(st.seq_len);
    uint64_t input_dim = static_cast<uint64_t>(st.input_dim);
    uint64_t hidden_dim = static_cast<uint64_t>(st.hidden_dim);
    uint64_t output_dim = static_cast<uint64_t>(st.output_dim);
    uint64_t md_group_size = static_cast<uint64_t>(st.md_group_size);
    uint64_t md_group_count = static_cast<uint64_t>(st.md_group_count);

    uint32_t flags = 0u;
    if (has_val)
        flags |= 1u;

    if (!write_pod(os, batch_size) || !write_pod(os, seq_len) || !write_pod(os, input_dim) ||
        !write_pod(os, hidden_dim) || !write_pod(os, output_dim) || !write_pod(os, md_group_size) ||
        !write_pod(os, md_group_count) || !write_pod(os, epoch) || !write_pod(os, seed) || !write_pod(os, lr) ||
        !write_pod(os, train_acc) || !write_pod(os, train_loss) || !write_pod(os, train_unc) ||
        !write_pod(os, val_acc) || !write_pod(os, val_loss) || !write_pod(os, val_unc) || !write_pod(os, flags))
    {
        return false;
    }

    const auto &gs = st.glial_state;
    if (!write_pod(os, gs.current_threshold) || !write_pod(os, gs.current_lr) || !write_pod(os, gs.prev_error) ||
        !write_pod(os, gs.error_integral) || !write_pod(os, gs.threshold_velocity) || !write_pod(os, gs.sparsity_ema) ||
        !write_pod(os, gs.error_ema) || !write_pod(os, gs.last_change_ratio) || !write_pod(os, gs.stable_count) ||
        !write_pod(os, gs.is_first_call) || !write_pod(os, gs.current_noise_gain) || !write_pod(os, gs.last_sparsity) ||
        !write_pod(os, st.target_sparsity) || !write_pod(os, st.glial_threshold_snapshot) ||
        !write_pod(os, st.hippocampus_learned_threshold))
    {
        return false;
    }

    for (float w : st.neuromodulator_w)
    {
        if (!write_pod(os, w))
            return false;
    }

    const auto &nm = st.neuromodulator_last;
    if (!write_pod(os, nm.target_sparsity) || !write_pod(os, nm.vigilance) || !write_pod(os, nm.noise_gain) ||
        !write_pod(os, nm.md_silu_tau) || !write_pod(os, nm.md_silu_alpha) ||
        !write_pod(os, nm.md_silu_novelty_power) || !write_pod(os, nm.md_silu_novelty_sharpness))
    {
        return false;
    }

    if (!write_pod(os, st.md_vigilance) || !write_pod(os, st.md_tau) || !write_pod(os, st.md_alpha) ||
        !write_pod(os, st.md_novelty_power) || !write_pod(os, st.md_novelty_sharpness))
    {
        return false;
    }

    uint8_t use_bias_flags = 0u;
    if (st.proj_in_use_bias)
        use_bias_flags |= 1u << 0;
    if (st.proj_mid1_use_bias)
        use_bias_flags |= 1u << 1;
    if (st.proj_mid2_use_bias)
        use_bias_flags |= 1u << 2;
    if (st.proj_out_use_bias)
        use_bias_flags |= 1u << 3;
    if (!write_pod(os, use_bias_flags))
        return false;

    if (!write_vec_f32(os, st.md_group_offsets) || !write_vec_f32(os, st.proj_in_weight) ||
        !write_vec_f32(os, st.proj_in_bias) || !write_vec_f32(os, st.proj_mid1_weight) ||
        !write_vec_f32(os, st.proj_mid1_bias) || !write_vec_f32(os, st.proj_mid2_weight) ||
        !write_vec_f32(os, st.proj_mid2_bias) || !write_vec_f32(os, st.proj_out_weight) ||
        !write_vec_f32(os, st.proj_out_bias) || !write_vec_f32(os, st.w_fast) || !write_vec_f32(os, st.w_slow) ||
        !write_vec_f32(os, st.ssm_A) || !write_vec_f32(os, st.ssm_state))
    {
        return false;
    }

    os.flush();
    return static_cast<bool>(os);
}

// =================================================================
// CLI / Config / Resume helpers
// =================================================================

struct TrainConfig
{
    size_t In = 784;
    size_t Hidden = 1024;
    size_t Out = 10;
    size_t batch = 64;

    float target_active = 0.15f;

    int epochs = 20;
    float lr = 0.01f;
    float weight_decay = 0.0001f;
    uint32_t seed = 1337;
    bool profile = false;

    bool normalize = false;
    float mnist_mean = 0.1307f;
    float mnist_std = 0.3081f;
    bool augment = false;
    int augment_shift = 2;
    float label_smoothing = 0.0f;

    bool postpeak_enable = true;
    float postpeak_train_acc = 99.95f;
    float postpeak_drop = 0.10f;
    float postpeak_min_lr = 1e-4f;
    bool postpeak_disable_augment = true;
    bool postpeak_disable_label_smoothing = true;

    bool val_kick_enable = true;
    int val_kick_patience = 4;
    float val_kick_drop = 0.50f;
    bool val_kick_restore_best = true;

    bool ckpt_enable = true;
    std::string ckpt_dir = "checkpoints";
    bool ckpt_save_best = true;
    bool ckpt_save_last = true;
    int ckpt_every = 0;

    bool export_enable = true;
    std::string export_dir = "exports";
    bool export_onnx = true;

    bool ema_enable = false;
    float ema_decay = 0.999f;
    int ema_warmup_epochs = 0;

    bool swa_enable = false;
    int swa_start_epoch = 0;
    int swa_freq_epochs = 1;
};

struct TrainConfigSet
{
    bool lr = false;
    bool seed = false;
};

struct CliArgs
{
    std::string config_path;
    std::string resume_path;
    bool help = false;
};

struct ResumeData
{
    bool enabled = false;
    neurobit::layers::BitBrainLayer::CheckpointState st;
    CheckpointMeta meta;
    int start_epoch = 0;
};

static uint32_t make_random_seed_u32()
{
    std::random_device rd;
    uint32_t a = static_cast<uint32_t>(rd());
    uint32_t b = static_cast<uint32_t>(rd());
    uint32_t c = static_cast<uint32_t>(rd());
    return (a << 16) ^ (b << 1) ^ c;
}

static void print_usage()
{
    std::cout << "用法:
";
    std::cout << "./train_mnist_surrogate [--config <path>] [--resume <ckpt>]
";
    std::cout << "預設會讀取工作目錄下的config.toml(若存在)，否則使用內建預設值。
";
}

static bool parse_cli_args(int argc, char **argv, CliArgs &out, std::string &err)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            out.help = true;
            continue;
        }
        if ((arg == "--config" || arg == "--cfg") && i + 1 < argc)
        {
            out.config_path = argv[++i];
            continue;
        }
        if (arg.rfind("--config=", 0) == 0)
        {
            out.config_path = arg.substr(std::string("--config=").size());
            continue;
        }
        if ((arg == "--resume" || arg == "--load-ckpt") && i + 1 < argc)
        {
            out.resume_path = argv[++i];
            continue;
        }
        if (arg.rfind("--resume=", 0) == 0)
        {
            out.resume_path = arg.substr(std::string("--resume=").size());
            continue;
        }

        err = "未知參數:" + arg;
        return false;
    }
    return true;
}

static bool file_exists(const std::string &path)
{
    std::error_code ec;
    return std::filesystem::exists(path, ec);
}

static bool load_train_config_from_toml(const std::string &path, TrainConfig &cfg, TrainConfigSet &set)
{
    try
    {
        auto t = pnb::toml_min::parse_file(path);
        for (const auto &w : t.warnings)
            std::cout << "config警告:" << w << "
";

        cfg.In = t.get_size("model.in", cfg.In);
        cfg.Hidden = std::max<size_t>(16, t.get_size("model.hidden", cfg.Hidden));
        cfg.Out = std::max<size_t>(1, t.get_size("model.out", cfg.Out));
        cfg.batch = std::max<size_t>(1, t.get_size("model.batch", cfg.batch));

        if (t.has("activity.target"))
        {
            float v = t.get_float("activity.target", cfg.target_active);
            if (v > 1.0f)
                v *= 0.01f;
            cfg.target_active = std::clamp(v, 0.0f, 1.0f);
        }

        cfg.epochs = std::max(1, t.get_int("train.epochs", cfg.epochs));
        if (t.has("train.lr"))
        {
            cfg.lr = t.get_float("train.lr", cfg.lr);
            set.lr = true;
        }
        if (t.has("train.seed"))
        {
            auto raw_seed = t.get_raw("train.seed");
            if (raw_seed)
            {
                std::string token = pnb::toml_min::normalize_number_token(pnb::toml_min::trim(*raw_seed));
                try
                {
                    long long s = std::stoll(token, nullptr, 10);
                    if (s == -1)
                    {
                        cfg.seed = make_random_seed_u32();
                        std::cout << "config:train.seed=-1|改用隨機seed=" << cfg.seed << "
";
                    }
                    else if (s >= 0 && s <= 0xFFFFFFFFll)
                    {
                        cfg.seed = static_cast<uint32_t>(s);
                    }
                }
                catch (...)
                {
                }
            }
            set.seed = true;
        }
        cfg.weight_decay = t.get_float("train.weight_decay", cfg.weight_decay);
        cfg.profile = t.get_bool("train.profile", cfg.profile);

        cfg.normalize = t.get_bool("data.normalize", cfg.normalize);
        cfg.mnist_mean = t.get_float("data.mnist_mean", cfg.mnist_mean);
        cfg.mnist_std = t.get_float("data.mnist_std", cfg.mnist_std);
        cfg.augment = t.get_bool("data.augment", cfg.augment);
        cfg.augment_shift = std::max(0, t.get_int("data.augment_shift", cfg.augment_shift));
        cfg.label_smoothing = t.get_float("data.label_smoothing", cfg.label_smoothing);

        cfg.postpeak_enable = t.get_bool("schedule.postpeak.enable", cfg.postpeak_enable);
        cfg.postpeak_train_acc = t.get_float("schedule.postpeak.train_acc", cfg.postpeak_train_acc);
        cfg.postpeak_drop = t.get_float("schedule.postpeak.drop", cfg.postpeak_drop);
        cfg.postpeak_min_lr = t.get_float("schedule.postpeak.min_lr", cfg.postpeak_min_lr);
        cfg.postpeak_disable_augment = t.get_bool("schedule.postpeak.disable_augment", cfg.postpeak_disable_augment);
        cfg.postpeak_disable_label_smoothing =
            t.get_bool("schedule.postpeak.disable_label_smoothing", cfg.postpeak_disable_label_smoothing);

        cfg.val_kick_enable = t.get_bool("schedule.val_kick.enable", cfg.val_kick_enable);
        cfg.val_kick_patience = std::max(1, t.get_int("schedule.val_kick.patience", cfg.val_kick_patience));
        cfg.val_kick_drop = t.get_float("schedule.val_kick.drop", cfg.val_kick_drop);
        cfg.val_kick_restore_best = t.get_bool("schedule.val_kick.restore_best", cfg.val_kick_restore_best);

        cfg.ckpt_enable = t.get_bool("checkpoint.enable", cfg.ckpt_enable);
        cfg.ckpt_dir = t.get_string("checkpoint.dir", cfg.ckpt_dir);
        cfg.ckpt_save_best = t.get_bool("checkpoint.save_best", cfg.ckpt_save_best);
        cfg.ckpt_save_last = t.get_bool("checkpoint.save_last", cfg.ckpt_save_last);
        cfg.ckpt_every = std::max(0, t.get_int("checkpoint.every", cfg.ckpt_every));

        cfg.export_enable = t.get_bool("export.enable", cfg.export_enable);
        cfg.export_dir = t.get_string("export.dir", cfg.export_dir);
        cfg.export_onnx = t.get_bool("export.onnx", cfg.export_onnx);

        cfg.ema_enable = t.get_bool("ema.enable", cfg.ema_enable);
        cfg.ema_decay = t.get_float("ema.decay", cfg.ema_decay);
        cfg.ema_warmup_epochs = std::max(0, t.get_int("ema.warmup_epochs", cfg.ema_warmup_epochs));

        cfg.swa_enable = t.get_bool("swa.enable", cfg.swa_enable);
        cfg.swa_start_epoch = std::max(0, t.get_int("swa.start_epoch", cfg.swa_start_epoch));
        cfg.swa_freq_epochs = std::max(1, t.get_int("swa.freq_epochs", cfg.swa_freq_epochs));
        return true;
    }
    catch (const std::exception &e)
    {
        std::cout << "載入config失敗:" << path << "|" << e.what() << "
";
        return false;
    }
}

static bool load_train_config(const CliArgs &cli, TrainConfig &cfg, TrainConfigSet &set)
{
    if (!cli.config_path.empty())
    {
        if (!load_train_config_from_toml(cli.config_path, cfg, set))
            return false;
        std::cout << "載入config:" << cli.config_path << "
";
        return true;
    }

    if (!file_exists("config.toml"))
        return true;

    if (!load_train_config_from_toml("config.toml", cfg, set))
        return false;
    std::cout << "載入config:config.toml
";
    return true;
}

static bool load_mnist_required(const std::string &img, const std::string &lbl, MNISTData &out)
{
    try
    {
        load_mnist(img, lbl, out);
        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "載入MNIST失敗：" << e.what() << "
";
        return false;
    }
}

static bool load_mnist_optional_val(const std::string &img, const std::string &lbl, MNISTData &out)
{
    try
    {
        load_mnist(img, lbl, out);
        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "載入驗證集失敗(將略過)：" << e.what() << "
";
        return false;
    }
}

static bool load_resume(const std::string &path, ResumeData &out, std::string &err)
{
    if (path.empty())
        return true;
    if (!load_checkpoint_file(path, out.st, out.meta, &err))
        return false;
    out.enabled = true;
    out.start_epoch = static_cast<int>(out.meta.epoch) + 1;
    return true;
}

static void apply_resume_to_config(TrainConfig &cfg, const TrainConfigSet &set, const ResumeData &resume)
{
    if (!resume.enabled)
        return;

    size_t cfg_in0 = cfg.In;
    size_t cfg_hidden0 = cfg.Hidden;
    size_t cfg_out0 = cfg.Out;
    size_t cfg_batch0 = cfg.batch;

    cfg.In = resume.st.input_dim;
    cfg.Hidden = resume.st.hidden_dim;
    cfg.Out = resume.st.output_dim;
    cfg.batch = resume.st.batch_size;

    if (!set.seed)
        cfg.seed = resume.meta.seed;
    if (!set.lr)
        cfg.lr = resume.meta.lr;

    if (cfg_in0 != cfg.In || cfg_hidden0 != cfg.Hidden || cfg_out0 != cfg.Out || cfg_batch0 != cfg.batch)
    {
        std::cout << "警告:resume後模型維度以檢查點為準|in=" << cfg.In << "|hidden=" << cfg.Hidden
                  << "|out=" << cfg.Out << "|batch=" << cfg.batch << "
";
    }
}

int main(int argc, char **argv)
{
    CliArgs cli;
    std::string cli_err;
    if (!parse_cli_args(argc, argv, cli, cli_err))
    {
        std::cout << cli_err << "
";
        print_usage();
        return 1;
    }
    if (cli.help)
    {
        print_usage();
        return 0;
    }

    TrainConfig cfg;
    TrainConfigSet cfg_set;
    if (!load_train_config(cli, cfg, cfg_set))
        return 1;

    ResumeData resume;
    std::string resume_err;
    if (!load_resume(cli.resume_path, resume, resume_err))
    {
        std::cout << "載入檢查點失敗:" << cli.resume_path << "|" << resume_err << "
";
        return 1;
    }
    apply_resume_to_config(cfg, cfg_set, resume);
    if (resume.enabled)
    {
        std::cout << "載入檢查點成功:" << cli.resume_path << "|epoch=" << resume.meta.epoch << "|seed=" << cfg.seed
                  << "|lr=" << cfg.lr << "
";
        if (cfg.epochs <= resume.start_epoch)
        {
            std::cout << "epochs=" << cfg.epochs << "小於等於resume後起始輪數=" << resume.start_epoch << "，無需續訓
";
            return 0;
        }
    }

    if (cfg.Out != 10)
    {
        std::cout << "警告：MNIST標籤為0-9；Out=" << cfg.Out << "可能不符合此資料集。
";
    }

    std::cout << "初始化PNBBitBrain（" << cfg.In << "->" << cfg.Hidden << "->" << cfg.Out
              << "），資料集：MNIST（BatchSize:" << cfg.batch << "）...
";
    std::cout << "架構：BitNetb1.58+SSM+M-DSiLU+Glial+Hippocampus
";
    std::cout << "隱層維度(Hidden):" << cfg.Hidden << "
";

    MNISTData data;
    MNISTData val_data;
    if (!load_mnist_required("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", data))
        return 1;
    bool has_val = load_mnist_optional_val("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte", val_data);

    try
    {
        size_t In = cfg.In;
        size_t Hidden = cfg.Hidden;
        size_t Out = cfg.Out;
        size_t BATCH_SIZE = cfg.batch;
        float target_active = cfg.target_active;
        int epochs = cfg.epochs;
        float LR = cfg.lr;
        float weight_decay = cfg.weight_decay;
        bool enable_profile = cfg.profile;
        uint32_t seed = cfg.seed;
        bool enable_normalize = cfg.normalize;
        float mnist_mean = cfg.mnist_mean;
        float mnist_std = cfg.mnist_std;
        bool enable_augment = cfg.augment;
        int augment_shift = cfg.augment_shift;
        float label_smoothing = cfg.label_smoothing;
        bool enable_postpeak = cfg.postpeak_enable;
        float postpeak_train_acc = cfg.postpeak_train_acc;
        float postpeak_drop = cfg.postpeak_drop;
        float postpeak_min_lr = cfg.postpeak_min_lr;
        bool postpeak_disable_augment = cfg.postpeak_disable_augment;
        bool postpeak_disable_label_smoothing = cfg.postpeak_disable_label_smoothing;
        bool enable_val_kick = cfg.val_kick_enable;
        int val_kick_patience = cfg.val_kick_patience;
        float val_kick_drop = cfg.val_kick_drop;
        bool val_kick_restore_best = cfg.val_kick_restore_best;
        bool enable_ckpt = cfg.ckpt_enable;
        std::string ckpt_dir = cfg.ckpt_dir;
        bool ckpt_save_best = cfg.ckpt_save_best;
        bool ckpt_save_last = cfg.ckpt_save_last;
        int ckpt_every = cfg.ckpt_every;
        bool enable_export = cfg.export_enable;
        bool enable_export_onnx = cfg.export_onnx;
        std::string export_dir = cfg.export_dir;
        bool enable_ema = cfg.ema_enable;
        float ema_decay = cfg.ema_decay;
        int ema_warmup_epochs = cfg.ema_warmup_epochs;
        bool enable_swa = cfg.swa_enable;
        int swa_start_epoch = cfg.swa_start_epoch;
        int swa_freq_epochs = cfg.swa_freq_epochs;

        s::queue q{s::default_selector_v,
                   s::property_list{s::property::queue::enable_profiling{}, s::property::queue::in_order{}}};
        std::cout << "運行裝置：" << q.get_device().get_info<s::info::device::name>() << "
";
        label_smoothing = std::clamp(label_smoothing, 0.0f, 0.2f);
        mnist_std = std::fmax(mnist_std, 1e-6f);
        float inv_mnist_std = 1.0f / mnist_std;
        weight_decay = std::clamp(weight_decay, 0.0f, 0.1f);
        postpeak_drop = std::clamp(postpeak_drop, 0.0f, 1.0f);
        postpeak_min_lr = std::clamp(postpeak_min_lr, 1e-7f, 1.0f);
        val_kick_drop = std::clamp(val_kick_drop, 0.0f, 1.0f);
        target_active = std::clamp(target_active, 0.01f, 0.99f);
        ema_decay = std::clamp(ema_decay, 0.0f, 0.9999999f);
        if (!std::isfinite(ema_decay))
            ema_decay = 0.999f;

        // BitBrain Config
        neurobit::layers::BitBrainLayer::Config brain_cfg;
        brain_cfg.batch_size = BATCH_SIZE;
        brain_cfg.input_dim = In;
        brain_cfg.hidden_dim = Hidden;
        brain_cfg.output_dim = Out;
        // Glial settings
        brain_cfg.glial_config.initial_threshold = 0.0f;
        brain_cfg.glial_config.target_sparsity = target_active;
        brain_cfg.glial_config.min_threshold = 0.0f;
        // Keep Glial homeostasis slow and stable (avoid per-batch LR runaway).
        brain_cfg.glial_config.adaptive_lr = false;
        brain_cfg.glial_config.initial_lr = 0.001f;
        brain_cfg.glial_config.max_threshold = 3.0f; // Increased limit for threshold
        brain_cfg.vigilance = 0.4f;                  // M-DSiLU vigilance baseline (neuromodulator may adjust it)

        brain_cfg.enable_neuromodulator = true;
        float active_band = 0.01f;
        brain_cfg.neuromodulator_config.active_min = std::clamp(target_active - active_band, 0.0f, 1.0f);
        brain_cfg.neuromodulator_config.active_max = std::clamp(target_active + active_band, 0.0f, 1.0f);
        if (brain_cfg.neuromodulator_config.active_max < brain_cfg.neuromodulator_config.active_min + 1e-3f)
        {
            brain_cfg.neuromodulator_config.active_min = std::clamp(target_active - 1e-3f, 0.0f, 1.0f);
            brain_cfg.neuromodulator_config.active_max = std::clamp(target_active + 1e-3f, 0.0f, 1.0f);
        }
        brain_cfg.neuromodulator_config.base_target = target_active;
        brain_cfg.neuromodulator_config.novelty_gain = 0.05f;
        brain_cfg.neuromodulator_config.uncertainty_gain = 0.03f;
        brain_cfg.neuromodulator_config.base_vigilance = 0.4f;
        brain_cfg.neuromodulator_config.novelty_vigilance_gain = 0.05f;
        brain_cfg.neuromodulator_config.enable_rule = true;
        brain_cfg.neuromodulator_config.enable_learned = true;
        brain_cfg.neuromodulator_config.learned_lr = 0.01f;
        brain_cfg.neuromodulator_config.learned_kp = 0.2f;
        brain_cfg.neuromodulator_config.freeze_inference = true;

        // The Brain
        neurobit::layers::BitBrainLayer brain(q, brain_cfg);
        brain.set_feedback(0.0f, 0.0f, 0.0f, true);
        std::cout << "目標活躍率設定=" << (target_active * 100.0f) << "%|容忍=±1%|範圍="
                  << (brain_cfg.neuromodulator_config.active_min * 100.0f) << "%~"
                  << (brain_cfg.neuromodulator_config.active_max * 100.0f) << "%
";

        struct WeightPack
        {
            bool proj_in_use_bias = false;
            bool proj_mid1_use_bias = false;
            bool proj_mid2_use_bias = false;
            bool proj_out_use_bias = false;

            std::vector<float> md_group_offsets;
            std::vector<float> proj_in_weight;
            std::vector<float> proj_in_bias;
            std::vector<float> proj_mid1_weight;
            std::vector<float> proj_mid1_bias;
            std::vector<float> proj_mid2_weight;
            std::vector<float> proj_mid2_bias;
            std::vector<float> proj_out_weight;
            std::vector<float> proj_out_bias;
            std::vector<float> w_fast;
            std::vector<float> w_slow;
            std::vector<float> ssm_A;
        };

        auto pack_from_ckpt = [](const neurobit::layers::BitBrainLayer::CheckpointState &st) {
            WeightPack p;
            p.proj_in_use_bias = st.proj_in_use_bias;
            p.proj_mid1_use_bias = st.proj_mid1_use_bias;
            p.proj_mid2_use_bias = st.proj_mid2_use_bias;
            p.proj_out_use_bias = st.proj_out_use_bias;
            p.md_group_offsets = st.md_group_offsets;
            p.proj_in_weight = st.proj_in_weight;
            p.proj_in_bias = st.proj_in_bias;
            p.proj_mid1_weight = st.proj_mid1_weight;
            p.proj_mid1_bias = st.proj_mid1_bias;
            p.proj_mid2_weight = st.proj_mid2_weight;
            p.proj_mid2_bias = st.proj_mid2_bias;
            p.proj_out_weight = st.proj_out_weight;
            p.proj_out_bias = st.proj_out_bias;
            p.w_fast = st.w_fast;
            p.w_slow = st.w_slow;
            p.ssm_A = st.ssm_A;
            return p;
        };

        auto apply_pack_to_ckpt = [](neurobit::layers::BitBrainLayer::CheckpointState &st, const WeightPack &p) {
            st.proj_in_use_bias = p.proj_in_use_bias;
            st.proj_mid1_use_bias = p.proj_mid1_use_bias;
            st.proj_mid2_use_bias = p.proj_mid2_use_bias;
            st.proj_out_use_bias = p.proj_out_use_bias;
            st.md_group_offsets = p.md_group_offsets;
            st.proj_in_weight = p.proj_in_weight;
            st.proj_in_bias = p.proj_in_bias;
            st.proj_mid1_weight = p.proj_mid1_weight;
            st.proj_mid1_bias = p.proj_mid1_bias;
            st.proj_mid2_weight = p.proj_mid2_weight;
            st.proj_mid2_bias = p.proj_mid2_bias;
            st.proj_out_weight = p.proj_out_weight;
            st.proj_out_bias = p.proj_out_bias;
            st.w_fast = p.w_fast;
            st.w_slow = p.w_slow;
            st.ssm_A = p.ssm_A;
        };

        auto ema_update_vec = [](std::vector<float> &ema, const std::vector<float> &cur, float decay) {
            if (ema.size() != cur.size())
                ema = cur;
            float one_minus = 1.0f - decay;
            for (size_t i = 0; i < cur.size(); ++i)
                ema[i] = decay * ema[i] + one_minus * cur[i];
        };

        auto swa_update_vec = [](std::vector<float> &swa, const std::vector<float> &cur, uint64_t count) {
            if (swa.size() != cur.size())
            {
                swa = cur;
                return;
            }
            float inv = 1.0f / static_cast<float>(count + 1);
            for (size_t i = 0; i < cur.size(); ++i)
                swa[i] += (cur[i] - swa[i]) * inv;
        };

        std::optional<WeightPack> ema_pack;
        std::optional<WeightPack> swa_pack;
        uint64_t swa_count = 0;

        // Memory Buffers (Hippocampus Weights)
        // Usually these would be persistent. Initializing to small random values.
        std::vector<float> h_W_fast(In * Hidden);
        std::vector<float> h_W_slow(In * Hidden);
        std::mt19937 gen(42);
        std::normal_distribution<float> d(0.0f, 0.01f);
        for (auto &w : h_W_fast)
            w = d(gen);
        for (auto &w : h_W_slow)
            w = d(gen);

        s::buffer<float, 1> buf_W_fast{h_W_fast.data(), s::range<1>(h_W_fast.size())};
        s::buffer<float, 1> buf_W_slow{h_W_slow.data(), s::range<1>(h_W_slow.size())};

        // Data Buffers
        s::buffer<float, 1> buf_X{s::range<1>(BATCH_SIZE * In)};
        s::buffer<float, 1> buf_Out{s::range<1>(BATCH_SIZE * Out)};
        s::buffer<float, 1> buf_Grad_Out{s::range<1>(BATCH_SIZE * Out)};
        s::buffer<float, 1> buf_Grad_In{s::range<1>(BATCH_SIZE * In)}; // Backprop to input
        s::buffer<int, 1> buf_Label{s::range<1>(BATCH_SIZE)};

        float *epoch_loss_sum = s::malloc_shared<float>(1, q);
        float *epoch_entropy_sum = s::malloc_shared<float>(1, q);
        uint32_t *epoch_correct_sum = s::malloc_shared<uint32_t>(1, q);
        float *val_loss_sum = s::malloc_shared<float>(1, q);
        float *val_entropy_sum = s::malloc_shared<float>(1, q);
        uint32_t *val_correct_sum = s::malloc_shared<uint32_t>(1, q);

        int total_correct = 0;
        int samples_processed = 0;
        BioLrScheduler lr_sched;
        lr_sched.min_lr = std::max(lr_sched.min_lr, postpeak_min_lr);

        std::vector<size_t> train_indices(data.images.size());
        std::iota(train_indices.begin(), train_indices.end(), 0);
        std::mt19937 rng(seed);
        float best_val_acc = -1.0f;
        float best_val_loss = std::numeric_limits<float>::infinity();
        int best_epoch = -1;
        float best_epoch_lr = LR;
        std::unique_ptr<neurobit::layers::BitBrainLayer::CheckpointState> best_state;
        bool postpeak_triggered = false;
        int val_no_improve_epochs = 0;

        if (resume.enabled)
        {
            std::string import_err;
            if (!brain.import_checkpoint_host(resume.st, buf_W_fast, buf_W_slow, &import_err))
            {
                std::cout << "套用檢查點失敗:" << import_err << "
";
                return 1;
            }
            best_val_acc = resume.meta.has_val ? resume.meta.val_acc : resume.meta.train_acc;
            best_val_loss = resume.meta.has_val ? resume.meta.val_loss : resume.meta.train_loss;
            best_epoch = static_cast<int>(resume.meta.epoch);
            best_epoch_lr = resume.meta.lr;
            best_state = std::make_unique<neurobit::layers::BitBrainLayer::CheckpointState>(resume.st);
        }

        int start_epoch = resume.enabled ? resume.start_epoch : 0;
        for (int epoch = start_epoch; epoch < epochs; ++epoch)
        {
            auto epoch_start_time = std::chrono::high_resolution_clock::now();
            total_correct = 0;
            samples_processed = 0;
            *epoch_loss_sum = 0.0f;
            *epoch_entropy_sum = 0.0f;
            *epoch_correct_sum = 0u;

            double prep_ms_sum = 0.0;
            double fwd_wall_ms_sum = 0.0;
            double bwd_wall_ms_sum = 0.0;
            double step_wall_ms_sum = 0.0;
            double loss_wall_ms_sum = 0.0;
            double fwd_dev_span_ms_sum = 0.0;
            double bwd_dev_span_ms_sum = 0.0;
            double step_dev_span_ms_sum = 0.0;
            double loss_dev_span_ms_sum = 0.0;
            double fwd_dev_sum_ms_sum = 0.0;
            double bwd_dev_sum_ms_sum = 0.0;
            double step_dev_sum_ms_sum = 0.0;
            double loss_dev_sum_ms_sum = 0.0;
            size_t prof_batches = 0;

            std::vector<s::event> fwd_events;
            std::vector<s::event> bwd_events;
            std::vector<s::event> step_events;
            std::vector<s::event> loss_events;
            fwd_events.reserve(64);
            bwd_events.reserve(64);
            step_events.reserve(32);
            loss_events.reserve(8);

            std::shuffle(train_indices.begin(), train_indices.end(), rng);

            for (size_t i = 0; i < train_indices.size(); i += BATCH_SIZE)
            {
                size_t current_batch = std::min(BATCH_SIZE, train_indices.size() - i);
                brain.set_effective_batch_size(current_batch);

                // 1. Prepare Batch
                auto prep_t0 = std::chrono::high_resolution_clock::now();
                {
                    s::host_accessor acc_X{buf_X, s::write_only};
                    s::host_accessor acc_L{buf_Label, s::write_only};
                    for (size_t b = 0; b < current_batch; ++b)
                    {
                        size_t src = train_indices[i + b];
                        float *dst = acc_X.get_pointer() + b * In;
                        const auto &img = data.images[src];
                        if (!enable_normalize && !enable_augment)
                        {
                            std::copy(img.begin(), img.end(), dst);
                        }
                        else
                        {
                            write_image_to_batch(dst, img, enable_normalize, mnist_mean, inv_mnist_std, enable_augment,
                                                 augment_shift, rng);
                        }
                        acc_L[b] = data.labels[src];
                    }
                }
                auto prep_t1 = std::chrono::high_resolution_clock::now();
                prep_ms_sum += std::chrono::duration<double, std::milli>(prep_t1 - prep_t0).count();

                // 2. Forward
                auto fwd_t0 = std::chrono::high_resolution_clock::now();
                if (enable_profile)
                {
                    fwd_events.clear();
                    brain.forward(buf_X, buf_Out, buf_W_fast, buf_W_slow, &fwd_events);
                }
                else
                {
                    brain.forward(buf_X, buf_Out, buf_W_fast, buf_W_slow, nullptr);
                }
                auto fwd_t1 = std::chrono::high_resolution_clock::now();
                fwd_wall_ms_sum += std::chrono::duration<double, std::milli>(fwd_t1 - fwd_t0).count();
                if (enable_profile)
                {
                    auto fwd_timing = timing_from_events(fwd_events);
                    fwd_dev_span_ms_sum += static_cast<double>(fwd_timing.span_ns) / 1e6;
                    fwd_dev_sum_ms_sum += static_cast<double>(fwd_timing.sum_ns) / 1e6;
                }

                // 3. Loss + Gradient (device-side; accumulates epoch metrics)
                auto loss_t0 = std::chrono::high_resolution_clock::now();
                if (enable_profile)
                    loss_events.clear();
                auto ev_loss = q.submit([&](s::handler &h) {
                    s::accessor acc_Out(buf_Out, h, s::read_only);
                    s::accessor acc_L(buf_Label, h, s::read_only);
                    s::accessor acc_Grad(buf_Grad_Out, h, s::write_only, s::no_init);
                    h.parallel_for(s::range<1>(current_batch), [=](s::id<1> idx_b) {
                        size_t b = idx_b[0];
                        int label = acc_L[b];
                        int out_i = static_cast<int>(Out);
                        int lbl = label;
                        if (lbl < 0)
                            lbl = 0;
                        if (out_i > 0 && lbl >= out_i)
                            lbl = out_i - 1;

                        float max_val = -1e9f;
                        for (int k = 0; k < out_i; ++k)
                        {
                            float v = acc_Out[b * Out + k];
                            if (v > max_val)
                                max_val = v;
                        }

                        float sum_exp = 0.0f;
                        for (int k = 0; k < out_i; ++k)
                        {
                            sum_exp += s::exp(acc_Out[b * Out + k] - max_val);
                        }
                        float inv_sum = 1.0f / s::fmax(sum_exp, 1e-12f);

                        int pred = 0;
                        float max_prob = -1.0f;
                        for (int k = 0; k < out_i; ++k)
                        {
                            float p = s::exp(acc_Out[b * Out + k] - max_val) * inv_sum;
                            if (p > max_prob)
                            {
                                max_prob = p;
                                pred = k;
                            }
                        }

                        if (pred == lbl)
                        {
                            s::atomic_ref<uint32_t, s::memory_order::relaxed, s::memory_scope::device,
                                          s::access::address_space::global_space>
                                atomic_correct(*epoch_correct_sum);
                            atomic_correct.fetch_add(1u);
                        }

                        float eps = label_smoothing;
                        eps = s::clamp(eps, 0.0f, 0.2f);
                        float other = (out_i > 1) ? (eps / static_cast<float>(out_i - 1)) : 0.0f;
                        float loss = 0.0f;
                        float entropy = 0.0f;
                        for (int k = 0; k < out_i; ++k)
                        {
                            float p = s::exp(acc_Out[b * Out + k] - max_val) * inv_sum;
                            p = s::fmax(p, 1e-12f);
                            entropy += -p * s::log(p);
                            float target = (k == lbl) ? (1.0f - eps) : other;
                            loss += -target * s::log(p);
                        }

                        s::atomic_ref<float, s::memory_order::relaxed, s::memory_scope::device,
                                      s::access::address_space::global_space>
                            atomic_loss(*epoch_loss_sum);
                        atomic_loss.fetch_add(loss);

                        s::atomic_ref<float, s::memory_order::relaxed, s::memory_scope::device,
                                      s::access::address_space::global_space>
                            atomic_ent(*epoch_entropy_sum);
                        atomic_ent.fetch_add(entropy);

                        for (int k = 0; k < out_i; ++k)
                        {
                            float p = s::exp(acc_Out[b * Out + k] - max_val) * inv_sum;
                            float eps = label_smoothing;
                            eps = s::clamp(eps, 0.0f, 0.2f);
                            float other = (out_i > 1) ? (eps / static_cast<float>(out_i - 1)) : 0.0f;
                            float target = (k == lbl) ? (1.0f - eps) : other;
                            acc_Grad[b * Out + k] = (p - target) / static_cast<float>(current_batch);
                        }
                    });
                });
                if (enable_profile)
                    loss_events.push_back(ev_loss);
                auto loss_t1 = std::chrono::high_resolution_clock::now();
                loss_wall_ms_sum += std::chrono::duration<double, std::milli>(loss_t1 - loss_t0).count();
                if (enable_profile)
                {
                    auto loss_timing = timing_from_events(loss_events);
                    loss_dev_span_ms_sum += static_cast<double>(loss_timing.span_ns) / 1e6;
                    loss_dev_sum_ms_sum += static_cast<double>(loss_timing.sum_ns) / 1e6;
                }

                samples_processed += current_batch;

                // 4. Backward
                auto bwd_t0 = std::chrono::high_resolution_clock::now();
                if (enable_profile)
                {
                    bwd_events.clear();
                    brain.backward(buf_Grad_Out, buf_Grad_In, &bwd_events);
                }
                else
                {
                    brain.backward(buf_Grad_Out, buf_Grad_In, nullptr);
                }
                auto bwd_t1 = std::chrono::high_resolution_clock::now();
                bwd_wall_ms_sum += std::chrono::duration<double, std::milli>(bwd_t1 - bwd_t0).count();
                if (enable_profile)
                {
                    auto bwd_timing = timing_from_events(bwd_events);
                    bwd_dev_span_ms_sum += static_cast<double>(bwd_timing.span_ns) / 1e6;
                    bwd_dev_sum_ms_sum += static_cast<double>(bwd_timing.sum_ns) / 1e6;
                }

                // 5. Update
                auto step_t0 = std::chrono::high_resolution_clock::now();
                if (enable_profile)
                {
                    step_events.clear();
                    brain.step(LR, weight_decay, &step_events);
                }
                else
                {
                    brain.step(LR, weight_decay, nullptr);
                }
                auto step_t1 = std::chrono::high_resolution_clock::now();
                step_wall_ms_sum += std::chrono::duration<double, std::milli>(step_t1 - step_t0).count();
                if (enable_profile)
                {
                    auto step_timing = timing_from_events(step_events);
                    step_dev_span_ms_sum += static_cast<double>(step_timing.span_ns) / 1e6;
                    step_dev_sum_ms_sum += static_cast<double>(step_timing.sum_ns) / 1e6;
                }

                prof_batches++;
            }
            q.wait(); // Ensure all GPU operations are done before timing

            auto epoch_end_time = std::chrono::high_resolution_clock::now();
            auto epoch_duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end_time - epoch_start_time).count();

            // Stats (device-accumulated)
            total_correct = static_cast<int>(*epoch_correct_sum);
            float acc_frac = (samples_processed > 0)
                                 ? (static_cast<float>(total_correct) / static_cast<float>(samples_processed))
                                 : 0.0f;
            float accuracy = acc_frac * 100.0f;
            float avg_loss = (samples_processed > 0) ? (*epoch_loss_sum / static_cast<float>(samples_processed)) : 0.0f;
            float avg_entropy =
                (samples_processed > 0) ? (*epoch_entropy_sum / static_cast<float>(samples_processed)) : 0.0f;
            float avg_unc = avg_entropy / std::log(static_cast<float>(Out));
            float val_acc = 0.0f;
            float val_loss = 0.0f;
            float val_unc = 0.0f;
            int val_samples = 0;

            auto run_validation = [&](float &out_acc, float &out_loss, float &out_unc, int &out_samples) {
                out_acc = 0.0f;
                out_loss = 0.0f;
                out_unc = 0.0f;
                out_samples = 0;

                *val_loss_sum = 0.0f;
                *val_entropy_sum = 0.0f;
                *val_correct_sum = 0u;

                for (size_t i = 0; i < val_data.images.size(); i += BATCH_SIZE)
                {
                    size_t current_batch = std::min(BATCH_SIZE, val_data.images.size() - i);
                    brain.set_effective_batch_size(current_batch);

                    {
                        s::host_accessor acc_X{buf_X, s::write_only};
                        s::host_accessor acc_L{buf_Label, s::write_only};
                        for (size_t b = 0; b < current_batch; ++b)
                        {
                            float *dst = acc_X.get_pointer() + b * In;
                            const auto &img = val_data.images[i + b];
                            if (!enable_normalize)
                            {
                                std::copy(img.begin(), img.end(), dst);
                            }
                            else
                            {
                                write_image_to_batch(dst, img, enable_normalize, mnist_mean, inv_mnist_std, false, 0, rng);
                            }
                            acc_L[b] = val_data.labels[i + b];
                        }
                    }

                    brain.forward_inference(buf_X, buf_Out, buf_W_fast, buf_W_slow, nullptr);

                    q.submit([&](s::handler &h) {
                        s::accessor acc_Out(buf_Out, h, s::read_only);
                        s::accessor acc_L(buf_Label, h, s::read_only);
                        h.parallel_for(s::range<1>(current_batch), [=](s::id<1> idx_b) {
                            size_t b = idx_b[0];
                            int label = acc_L[b];
                            int out_i = static_cast<int>(Out);
                            int lbl = label;
                            if (lbl < 0)
                                lbl = 0;
                            if (out_i > 0 && lbl >= out_i)
                                lbl = out_i - 1;

                            float max_val = -1e9f;
                            for (int k = 0; k < out_i; ++k)
                            {
                                float v = acc_Out[b * Out + k];
                                if (v > max_val)
                                    max_val = v;
                            }

                            float sum_exp = 0.0f;
                            for (int k = 0; k < out_i; ++k)
                            {
                                sum_exp += s::exp(acc_Out[b * Out + k] - max_val);
                            }
                            float inv_sum = 1.0f / s::fmax(sum_exp, 1e-12f);

                            int pred = 0;
                            float max_prob = -1.0f;
                            for (int k = 0; k < out_i; ++k)
                            {
                                float p = s::exp(acc_Out[b * Out + k] - max_val) * inv_sum;
                                if (p > max_prob)
                                {
                                    max_prob = p;
                                    pred = k;
                                }
                            }

                            if (pred == lbl)
                            {
                                s::atomic_ref<uint32_t, s::memory_order::relaxed, s::memory_scope::device,
                                              s::access::address_space::global_space>
                                    atomic_correct(*val_correct_sum);
                                atomic_correct.fetch_add(1u);
                            }

                            float p_label = s::exp(acc_Out[b * Out + lbl] - max_val) * inv_sum;
                            p_label = s::fmax(p_label, 1e-12f);
                            float loss = -s::log(p_label);

                            float entropy = 0.0f;
                            for (int k = 0; k < out_i; ++k)
                            {
                                float p = s::exp(acc_Out[b * Out + k] - max_val) * inv_sum;
                                p = s::fmax(p, 1e-12f);
                                entropy += -p * s::log(p);
                            }

                            s::atomic_ref<float, s::memory_order::relaxed, s::memory_scope::device,
                                          s::access::address_space::global_space>
                                atomic_loss(*val_loss_sum);
                            atomic_loss.fetch_add(loss);

                            s::atomic_ref<float, s::memory_order::relaxed, s::memory_scope::device,
                                          s::access::address_space::global_space>
                                atomic_ent(*val_entropy_sum);
                            atomic_ent.fetch_add(entropy);
                        });
                    });

                    out_samples += static_cast<int>(current_batch);
                }

                q.wait();

                out_acc = (out_samples > 0)
                              ? (static_cast<float>(*val_correct_sum) / static_cast<float>(out_samples)) * 100.0f
                              : 0.0f;
                out_loss = (out_samples > 0) ? (*val_loss_sum / static_cast<float>(out_samples)) : 0.0f;
                float val_entropy = (out_samples > 0) ? (*val_entropy_sum / static_cast<float>(out_samples)) : 0.0f;
                out_unc = val_entropy / std::log(static_cast<float>(Out));
            };

            if (has_val)
                run_validation(val_acc, val_loss, val_unc, val_samples);

            brain.set_feedback(avg_loss, avg_unc, acc_frac, true);
            brain.sync_host_feedback();
            float sched_loss = has_val ? val_loss : avg_loss;
            float sched_unc = has_val ? val_unc : avg_unc;
            lr_sched.update(sched_loss, sched_unc, brain.get_last_familiarity_host(), brain.get_last_active_rate_host(),
                            brain.get_target_sparsity_host(), LR);
            if (LR < lr_sched.min_lr)
                LR = lr_sched.min_lr;
            auto wabs = brain.get_weight_abs_stats_host();

            if (enable_postpeak && !postpeak_triggered && accuracy >= postpeak_train_acc)
            {
                postpeak_triggered = true;
                lr_sched.min_lr = std::max(lr_sched.min_lr, postpeak_min_lr);
                float new_lr = std::fmax(LR * postpeak_drop, postpeak_min_lr);
                if (new_lr < LR)
                {
                    LR = new_lr;
                    lr_sched.reset();
                }
                if (LR < lr_sched.min_lr)
                    LR = lr_sched.min_lr;

                bool changed = false;
                if (postpeak_disable_augment && enable_augment)
                {
                    enable_augment = false;
                    changed = true;
                }
                if (postpeak_disable_label_smoothing && label_smoothing != 0.0f)
                {
                    label_smoothing = 0.0f;
                    changed = true;
                }

                std::cout << "學習率下架:觸發=訓練峰值|門檻=" << postpeak_train_acc << "%|LR=" << LR;
                if (changed)
                {
                    std::cout << "|關閉增強=" << (enable_augment ? 0 : 1) << "|label_smoothing=" << label_smoothing;
                }
                std::cout << "
";
            }

            std::cout << "第" << epoch << "輪：準確率=" << accuracy << "%"
                      << "|活躍率=" << (brain.get_last_active_rate_host() * 100.0f) << "%"
                      << "|目標=" << (brain.get_target_sparsity_host() * 100.0f) << "%"
                      << "|損失=" << avg_loss << "|不確定性=" << (avg_unc * 100.0f) << "%"
                      << "|LR門檻=" << lr_sched.last_threshold << "|LR倍率=" << lr_sched.last_factor
                      << "|可塑性=" << (lr_sched.last_plasticity * 100.0f) << "%"
                      << "|平均權重(|W|)=" << wabs.mean_all << "|各層(|W|)in/m1/m2/out=" << wabs.proj_in << "/"
                      << wabs.proj_mid1 << "/" << wabs.proj_mid2 << "/" << wabs.proj_out
                      << "|Glial閾值=" << brain.get_glial_threshold_host() << "|學習率=" << LR
                      << "|耗時=" << epoch_duration << "ms
";

            if (has_val)
            {
                std::cout << "驗證：準確率=" << val_acc << "%|損失=" << val_loss << "|不確定性=" << (val_unc * 100.0f)
                          << "%|樣本=" << val_samples << "
";
            }

            neurobit::layers::BitBrainLayer::CheckpointState ckpt_current;
            bool have_ckpt_current = false;
            if ((enable_ema || enable_swa) && has_val)
            {
                brain.export_checkpoint_host(ckpt_current, buf_W_fast, buf_W_slow);
                have_ckpt_current = true;
            }

            if (enable_ema && has_val && have_ckpt_current)
            {
                if (!ema_pack)
                    ema_pack = pack_from_ckpt(ckpt_current);
                WeightPack cur = pack_from_ckpt(ckpt_current);

                if (epoch >= ema_warmup_epochs)
                {
                    ema_update_vec(ema_pack->md_group_offsets, cur.md_group_offsets, ema_decay);
                    ema_update_vec(ema_pack->proj_in_weight, cur.proj_in_weight, ema_decay);
                    ema_update_vec(ema_pack->proj_in_bias, cur.proj_in_bias, ema_decay);
                    ema_update_vec(ema_pack->proj_mid1_weight, cur.proj_mid1_weight, ema_decay);
                    ema_update_vec(ema_pack->proj_mid1_bias, cur.proj_mid1_bias, ema_decay);
                    ema_update_vec(ema_pack->proj_mid2_weight, cur.proj_mid2_weight, ema_decay);
                    ema_update_vec(ema_pack->proj_mid2_bias, cur.proj_mid2_bias, ema_decay);
                    ema_update_vec(ema_pack->proj_out_weight, cur.proj_out_weight, ema_decay);
                    ema_update_vec(ema_pack->proj_out_bias, cur.proj_out_bias, ema_decay);
                    ema_update_vec(ema_pack->w_fast, cur.w_fast, ema_decay);
                    ema_update_vec(ema_pack->w_slow, cur.w_slow, ema_decay);
                    ema_update_vec(ema_pack->ssm_A, cur.ssm_A, ema_decay);
                    ema_pack->proj_in_use_bias = cur.proj_in_use_bias;
                    ema_pack->proj_mid1_use_bias = cur.proj_mid1_use_bias;
                    ema_pack->proj_mid2_use_bias = cur.proj_mid2_use_bias;
                    ema_pack->proj_out_use_bias = cur.proj_out_use_bias;
                }

                neurobit::layers::BitBrainLayer::CheckpointState ema_eval = ckpt_current;
                apply_pack_to_ckpt(ema_eval, *ema_pack);
                std::string err;
                if (brain.import_checkpoint_host(ema_eval, buf_W_fast, buf_W_slow, &err))
                {
                    float ema_val_acc = 0.0f, ema_val_loss = 0.0f, ema_val_unc = 0.0f;
                    int ema_val_samples = 0;
                    run_validation(ema_val_acc, ema_val_loss, ema_val_unc, ema_val_samples);
                    std::cout << "驗證EMA：準確率=" << ema_val_acc << "%|損失=" << ema_val_loss << "|不確定性="
                              << (ema_val_unc * 100.0f) << "%|樣本=" << ema_val_samples << "
";
                }
                else
                {
                    std::cout << "驗證EMA：套用權重失敗|" << err << "
";
                }
                std::string restore_err;
                if (!brain.import_checkpoint_host(ckpt_current, buf_W_fast, buf_W_slow, &restore_err))
                    std::cout << "驗證EMA：還原權重失敗|" << restore_err << "
";
            }

            if (enable_swa && has_val && have_ckpt_current)
            {
                bool do_update = (epoch >= swa_start_epoch) && ((epoch - swa_start_epoch) % swa_freq_epochs == 0);
                if (!swa_pack)
                    swa_pack = pack_from_ckpt(ckpt_current);
                if (do_update)
                {
                    WeightPack cur = pack_from_ckpt(ckpt_current);
                    swa_update_vec(swa_pack->md_group_offsets, cur.md_group_offsets, swa_count);
                    swa_update_vec(swa_pack->proj_in_weight, cur.proj_in_weight, swa_count);
                    swa_update_vec(swa_pack->proj_in_bias, cur.proj_in_bias, swa_count);
                    swa_update_vec(swa_pack->proj_mid1_weight, cur.proj_mid1_weight, swa_count);
                    swa_update_vec(swa_pack->proj_mid1_bias, cur.proj_mid1_bias, swa_count);
                    swa_update_vec(swa_pack->proj_mid2_weight, cur.proj_mid2_weight, swa_count);
                    swa_update_vec(swa_pack->proj_mid2_bias, cur.proj_mid2_bias, swa_count);
                    swa_update_vec(swa_pack->proj_out_weight, cur.proj_out_weight, swa_count);
                    swa_update_vec(swa_pack->proj_out_bias, cur.proj_out_bias, swa_count);
                    swa_update_vec(swa_pack->w_fast, cur.w_fast, swa_count);
                    swa_update_vec(swa_pack->w_slow, cur.w_slow, swa_count);
                    swa_update_vec(swa_pack->ssm_A, cur.ssm_A, swa_count);
                    swa_pack->proj_in_use_bias = cur.proj_in_use_bias;
                    swa_pack->proj_mid1_use_bias = cur.proj_mid1_use_bias;
                    swa_pack->proj_mid2_use_bias = cur.proj_mid2_use_bias;
                    swa_pack->proj_out_use_bias = cur.proj_out_use_bias;
                    swa_count++;
                }

                if (swa_count > 0)
                {
                    neurobit::layers::BitBrainLayer::CheckpointState swa_eval = ckpt_current;
                    apply_pack_to_ckpt(swa_eval, *swa_pack);
                    std::string err;
                    if (brain.import_checkpoint_host(swa_eval, buf_W_fast, buf_W_slow, &err))
                    {
                        float swa_val_acc = 0.0f, swa_val_loss = 0.0f, swa_val_unc = 0.0f;
                        int swa_val_samples = 0;
                        run_validation(swa_val_acc, swa_val_loss, swa_val_unc, swa_val_samples);
                        std::cout << "驗證SWA：準確率=" << swa_val_acc << "%|損失=" << swa_val_loss << "|不確定性="
                                  << (swa_val_unc * 100.0f) << "%|樣本=" << swa_val_samples << "|count=" << swa_count
                                  << "
";
                    }
                    else
                    {
                        std::cout << "驗證SWA：套用權重失敗|" << err << "
";
                    }
                    std::string restore_err;
                    if (!brain.import_checkpoint_host(ckpt_current, buf_W_fast, buf_W_slow, &restore_err))
                        std::cout << "驗證SWA：還原權重失敗|" << restore_err << "
";
                }
            }

            bool is_best = false;
            if (has_val)
            {
                float acc_eps = 1e-6f;
                if (val_acc > best_val_acc + acc_eps)
                    is_best = true;
                else if (std::fabs(val_acc - best_val_acc) <= acc_eps && val_loss < best_val_loss)
                    is_best = true;
            }
            else
            {
                if (accuracy > best_val_acc)
                    is_best = true;
            }

            if (enable_profile && prof_batches > 0)
            {
                double inv = 1.0 / static_cast<double>(prof_batches);
                std::cout << "計時(每batch平均):準備=" << (prep_ms_sum * inv) << "ms"
                          << "|前向(牆鐘)=" << (fwd_wall_ms_sum * inv) << "ms"
                          << "|前向(裝置span)=" << (fwd_dev_span_ms_sum * inv) << "ms"
                          << "|前向(裝置sum)=" << (fwd_dev_sum_ms_sum * inv) << "ms"
                          << "|損失(牆鐘)=" << (loss_wall_ms_sum * inv) << "ms"
                          << "|損失(裝置span)=" << (loss_dev_span_ms_sum * inv) << "ms"
                          << "|損失(裝置sum)=" << (loss_dev_sum_ms_sum * inv) << "ms"
                          << "|反向(牆鐘)=" << (bwd_wall_ms_sum * inv) << "ms"
                          << "|反向(裝置span)=" << (bwd_dev_span_ms_sum * inv) << "ms"
                          << "|反向(裝置sum)=" << (bwd_dev_sum_ms_sum * inv) << "ms"
                          << "|更新(牆鐘)=" << (step_wall_ms_sum * inv) << "ms"
                          << "|更新(裝置span)=" << (step_dev_span_ms_sum * inv) << "ms"
                          << "|更新(裝置sum)=" << (step_dev_sum_ms_sum * inv) << "ms
";
            }

            if (enable_ckpt && (ckpt_save_last || ckpt_save_best || ckpt_every > 0))
            {
                try
                {
                    std::filesystem::create_directories(ckpt_dir);
                }
                catch (...)
                {
                }

                neurobit::layers::BitBrainLayer::CheckpointState ckpt;
                brain.export_checkpoint_host(ckpt, buf_W_fast, buf_W_slow);

                if (ckpt_save_last)
                {
                    std::string path = ckpt_dir + "/mnist_surrogate_last.ckpt";
                    if (save_checkpoint_file(path, ckpt, static_cast<uint32_t>(epoch), seed, LR, accuracy, avg_loss,
                                             avg_unc, has_val, val_acc, val_loss, val_unc))
                    {
                        std::cout << "保存檢查點last:" << path << "
";
                    }
                    else
                    {
                        std::cout << "保存檢查點失敗last:" << path << "
";
                    }
                }

                if (ckpt_save_best && is_best)
                {
                    best_val_acc = has_val ? val_acc : accuracy;
                    best_val_loss = has_val ? val_loss : avg_loss;
                    best_epoch = epoch;
                    best_epoch_lr = LR;
                    best_state = std::make_unique<neurobit::layers::BitBrainLayer::CheckpointState>(ckpt);
                    std::string path = ckpt_dir + "/mnist_surrogate_best.ckpt";
                    if (save_checkpoint_file(path, ckpt, static_cast<uint32_t>(epoch), seed, LR, accuracy, avg_loss,
                                             avg_unc, has_val, val_acc, val_loss, val_unc))
                    {
                        std::cout << "保存檢查點best:" << path << "
";
                    }
                    else
                    {
                        std::cout << "保存檢查點失敗best:" << path << "
";
                    }
                }

                if (ckpt_every > 0 && (epoch % ckpt_every) == 0)
                {
                    std::string path = ckpt_dir + "/mnist_surrogate_epoch" + std::to_string(epoch) + ".ckpt";
                    if (save_checkpoint_file(path, ckpt, static_cast<uint32_t>(epoch), seed, LR, accuracy, avg_loss,
                                             avg_unc, has_val, val_acc, val_loss, val_unc))
                    {
                        std::cout << "保存檢查點epoch:" << path << "
";
                    }
                    else
                    {
                        std::cout << "保存檢查點失敗epoch:" << path << "
";
                    }
                }
            }

            if (has_val && enable_val_kick && postpeak_triggered)
            {
                if (is_best)
                {
                    val_no_improve_epochs = 0;
                }
                else
                {
                    val_no_improve_epochs++;
                    if (val_no_improve_epochs >= val_kick_patience)
                    {
                        float new_lr = std::fmax(LR * val_kick_drop, postpeak_min_lr);
                        bool kicked = (new_lr < LR);
                        LR = new_lr;
                        lr_sched.reset();
                        lr_sched.min_lr = std::max(lr_sched.min_lr, postpeak_min_lr);
                        if (LR < lr_sched.min_lr)
                            LR = lr_sched.min_lr;
                        val_no_improve_epochs = 0;

                        bool restored = false;
                        if (val_kick_restore_best && best_state)
                        {
                            std::string import_err;
                            if (brain.import_checkpoint_host(*best_state, buf_W_fast, buf_W_slow, &import_err))
                            {
                                restored = true;
                            }
                            else
                            {
                                std::cout << "驗證推力:回退best失敗|" << import_err << "
";
                            }
                        }

                        if (kicked || restored)
                        {
                            std::cout << "驗證推力:patience=" << val_kick_patience << "|LR=" << LR
                                      << "|回退best=" << (restored ? 1 : 0) << "
";
                        }
                    }
                }
            }
        }

        if (enable_export)
        {
            try
            {
                std::filesystem::create_directories(export_dir);
            }
            catch (...)
            {
            }

            auto export_routine = [&](std::string filename_base, const neurobit::layers::BitBrainLayer::CheckpointState &state,
                                      uint32_t ep, float lr_val, float t_acc, float t_loss, float t_unc,
                                      bool h_val, float v_acc, float v_loss, float v_unc) {
                std::string bin_path = export_dir + "/" + filename_base + ".bin";
                if (save_checkpoint_file(bin_path, state, ep, seed, lr_val, t_acc, t_loss, t_unc, h_val, v_acc, v_loss, v_unc))
                {
                    std::cout << "保存模型bin:" << bin_path << "
";
                }
                else
                {
                    std::cout << "保存模型bin失敗:" << bin_path << "
";
                    return;
                }

                if (enable_export_onnx)
                {
                    std::string onnx_path = export_dir + "/" + filename_base + ".onnx";
                    std::string cmd = "python3 tools/pnb_ckpt_to_onnx.py --in \"" + bin_path + "\" --out \"" + onnx_path + "\"";
                    int rc = std::system(cmd.c_str());
                    if (rc == 0)
                    {
                        std::cout << "保存模型onnx:" << onnx_path << "
";
                    }
                    else
                    {
                        std::cout << "保存模型onnx失敗(需要python3+onnx):" << onnx_path
                                  << "|可先執行:python3 -m pip install onnx
";
                    }
                }
            };

            // 1. Export Last (Current State)
            {
                neurobit::layers::BitBrainLayer::CheckpointState last_state;
                brain.export_checkpoint_host(last_state, buf_W_fast, buf_W_slow);
                // 使用當前(最後)的狀態參數
                export_routine("mnist_surrogate_last", last_state, 
                               static_cast<uint32_t>(std::max(0, epochs - 1)), LR, 
                               0.0f, 0.0f, 0.0f, 
                               has_val, best_val_acc, best_val_loss, 0.0f);
            }

            // 2. Export Best (If exists)
            if (best_state)
            {
                export_routine("mnist_surrogate_best", *best_state, 
                               static_cast<uint32_t>(std::max(best_epoch, 0)), best_epoch_lr, 
                               0.0f, 0.0f, 0.0f, 
                               true, best_val_acc, best_val_loss, 0.0f);
            }
        }

        s::free(epoch_loss_sum, q);
        s::free(epoch_entropy_sum, q);
        s::free(epoch_correct_sum, q);
        s::free(val_loss_sum, q);
        s::free(val_entropy_sum, q);
        s::free(val_correct_sum, q);
    }
    catch (const s::exception &e)
    {
        std::cout << "SYCL錯誤：" << e.what() << "
";
        return 1;
    }
    return 0;
}
