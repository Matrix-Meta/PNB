#include "neurobit/core/types.hpp"
#include "neurobit/layers/bit_brain.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
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

    std::cout << "載入" << num_imgs << "張影像[" << rows << "x" << cols << "]，來源：" << image_path << "...\n";

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

int main(int argc, char **argv)
{
    size_t In = 784;
    size_t Hidden = 1024;
    size_t Out = 10;
    float LR = 0.01f;
    size_t BATCH_SIZE = 64;
    int epochs = 20;
    bool enable_profile = false;
    uint32_t seed = 1337;
    bool enable_ckpt = true;
    std::string ckpt_dir = "checkpoints";
    bool ckpt_save_best = true;
    bool ckpt_save_last = true;
    int ckpt_every = 0;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if ((arg == "--epochs" || arg == "-e") && i + 1 < argc)
        {
            epochs = std::max(1, std::stoi(argv[++i]));
        }
        else if (arg.rfind("--epochs=", 0) == 0)
        {
            epochs = std::max(1, std::stoi(arg.substr(std::string("--epochs=").size())));
        }
        else if (arg == "--hidden" && i + 1 < argc)
        {
            Hidden = std::max<size_t>(16, static_cast<size_t>(std::stoul(argv[++i])));
        }
        else if (arg.rfind("--hidden=", 0) == 0)
        {
            Hidden = std::max<size_t>(16, static_cast<size_t>(std::stoul(arg.substr(std::string("--hidden=").size()))));
        }
        else if ((arg == "--batch" || arg == "--batch_size" || arg == "-b") && i + 1 < argc)
        {
            BATCH_SIZE = std::max<size_t>(1, static_cast<size_t>(std::stoul(argv[++i])));
        }
        else if (arg.rfind("--batch=", 0) == 0)
        {
            BATCH_SIZE =
                std::max<size_t>(1, static_cast<size_t>(std::stoul(arg.substr(std::string("--batch=").size()))));
        }
        else if ((arg == "--out" || arg == "--classes") && i + 1 < argc)
        {
            Out = std::max<size_t>(1, static_cast<size_t>(std::stoul(argv[++i])));
        }
        else if (arg.rfind("--out=", 0) == 0)
        {
            Out = std::max<size_t>(1, static_cast<size_t>(std::stoul(arg.substr(std::string("--out=").size()))));
        }
        else if (arg == "--profile")
        {
            enable_profile = true;
        }
        else if (arg == "--seed" && i + 1 < argc)
        {
            seed = static_cast<uint32_t>(std::stoul(argv[++i]));
        }
        else if (arg.rfind("--seed=", 0) == 0)
        {
            seed = static_cast<uint32_t>(std::stoul(arg.substr(std::string("--seed=").size())));
        }
        else if (arg == "--no-ckpt")
        {
            enable_ckpt = false;
        }
        else if (arg == "--ckpt")
        {
            enable_ckpt = true;
        }
        else if ((arg == "--ckpt-dir" || arg == "--ckpt_dir") && i + 1 < argc)
        {
            ckpt_dir = argv[++i];
        }
        else if (arg.rfind("--ckpt-dir=", 0) == 0)
        {
            ckpt_dir = arg.substr(std::string("--ckpt-dir=").size());
        }
        else if (arg == "--no-ckpt-best")
        {
            ckpt_save_best = false;
        }
        else if (arg == "--ckpt-best")
        {
            ckpt_save_best = true;
        }
        else if (arg == "--no-ckpt-last")
        {
            ckpt_save_last = false;
        }
        else if (arg == "--ckpt-last")
        {
            ckpt_save_last = true;
        }
        else if (arg == "--ckpt-every" && i + 1 < argc)
        {
            ckpt_every = std::max(0, std::stoi(argv[++i]));
        }
        else if (arg.rfind("--ckpt-every=", 0) == 0)
        {
            ckpt_every = std::max(0, std::stoi(arg.substr(std::string("--ckpt-every=").size())));
        }
    }

    if (Out != 10)
    {
        std::cout << "警告：MNIST標籤為0-9；Out=" << Out << "可能不符合此資料集。\n";
    }

    std::cout << "初始化PNBBitBrain（" << In << "->" << Hidden << "->" << Out
              << "），資料集：MNIST（BatchSize:" << BATCH_SIZE << "）...\n";
    std::cout << "架構：BitNetb1.58+SSM+M-DSiLU+Glial+Hippocampus\n";
    std::cout << "隱層維度(Hidden):" << Hidden << "\n";

    MNISTData data;
    MNISTData val_data;
    bool has_val = false;

    try
    {
        load_mnist("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", data);
    }
    catch (const std::exception &e)
    {
        std::cerr << "載入MNIST失敗：" << e.what() << "\n";
        return 1;
    }

    try
    {
        load_mnist("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte", val_data);
        has_val = true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "載入驗證集失敗(將略過)：" << e.what() << "\n";
        has_val = false;
    }

    try
    {
        s::queue q{s::default_selector_v,
                   s::property_list{s::property::queue::enable_profiling{}, s::property::queue::in_order{}}};
        std::cout << "運行裝置：" << q.get_device().get_info<s::info::device::name>() << "\n";

        // BitBrain Config
        neurobit::layers::BitBrainLayer::Config cfg;
        cfg.batch_size = BATCH_SIZE;
        cfg.input_dim = In;
        cfg.hidden_dim = Hidden;
        cfg.output_dim = Out;
        // Glial settings
        cfg.glial_config.initial_threshold = 0.0f;
        // Aim for brain-like sparse activity (Active ~5–20%).
        cfg.glial_config.target_sparsity = 0.15f;
        cfg.glial_config.min_threshold = 0.0f;
        // Keep Glial homeostasis slow and stable (avoid per-batch LR runaway).
        cfg.glial_config.adaptive_lr = false;
        cfg.glial_config.initial_lr = 0.001f;
        cfg.glial_config.max_threshold = 3.0f; // Increased limit for threshold
        cfg.vigilance = 0.4f;                  // M-DSiLU vigilance baseline (neuromodulator may adjust it)

        cfg.enable_neuromodulator = true;
        cfg.neuromodulator_config.active_min = 0.05f;
        cfg.neuromodulator_config.active_max = 0.20f;
        cfg.neuromodulator_config.base_target = 0.15f;
        cfg.neuromodulator_config.novelty_gain = 0.05f;
        cfg.neuromodulator_config.uncertainty_gain = 0.03f;
        cfg.neuromodulator_config.base_vigilance = 0.4f;
        cfg.neuromodulator_config.novelty_vigilance_gain = 0.05f;
        cfg.neuromodulator_config.enable_rule = true;
        cfg.neuromodulator_config.enable_learned = true;
        cfg.neuromodulator_config.learned_lr = 0.01f;
        cfg.neuromodulator_config.learned_kp = 0.2f;
        cfg.neuromodulator_config.freeze_inference = true;

        // The Brain
        neurobit::layers::BitBrainLayer brain(q, cfg);
        brain.set_feedback(0.0f, 0.0f, 0.0f, true);

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

        std::vector<size_t> train_indices(data.images.size());
        std::iota(train_indices.begin(), train_indices.end(), 0);
        std::mt19937 rng(seed);
        float best_val_acc = -1.0f;
        float best_val_loss = std::numeric_limits<float>::infinity();

        for (int epoch = 0; epoch < epochs; ++epoch)
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
                        std::copy(data.images[src].begin(), data.images[src].end(), acc_X.get_pointer() + b * In);
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
                            atomic_loss(*epoch_loss_sum);
                        atomic_loss.fetch_add(loss);

                        s::atomic_ref<float, s::memory_order::relaxed, s::memory_scope::device,
                                      s::access::address_space::global_space>
                            atomic_ent(*epoch_entropy_sum);
                        atomic_ent.fetch_add(entropy);

                        for (int k = 0; k < out_i; ++k)
                        {
                            float p = s::exp(acc_Out[b * Out + k] - max_val) * inv_sum;
                            float target = (k == lbl) ? 1.0f : 0.0f;
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
                    brain.step(LR, 0.0001f, &step_events);
                }
                else
                {
                    brain.step(LR, 0.0001f, nullptr);
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

            if (has_val)
            {
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
                            std::copy(val_data.images[i + b].begin(), val_data.images[i + b].end(),
                                      acc_X.get_pointer() + b * In);
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

                    val_samples += static_cast<int>(current_batch);
                }

                q.wait();

                val_acc = (val_samples > 0)
                              ? (static_cast<float>(*val_correct_sum) / static_cast<float>(val_samples)) * 100.0f
                              : 0.0f;
                val_loss = (val_samples > 0) ? (*val_loss_sum / static_cast<float>(val_samples)) : 0.0f;
                float val_entropy = (val_samples > 0) ? (*val_entropy_sum / static_cast<float>(val_samples)) : 0.0f;
                val_unc = val_entropy / std::log(static_cast<float>(Out));
            }

            brain.set_feedback(avg_loss, avg_unc, acc_frac, true);
            brain.sync_host_feedback();
            float sched_loss = has_val ? val_loss : avg_loss;
            float sched_unc = has_val ? val_unc : avg_unc;
            lr_sched.update(sched_loss, sched_unc, brain.get_last_familiarity_host(), brain.get_last_active_rate_host(),
                            brain.get_target_sparsity_host(), LR);
            auto wabs = brain.get_weight_abs_stats_host();

            std::cout << "第" << epoch << "輪：準確率=" << accuracy << "%"
                      << "|活躍率=" << (brain.get_last_active_rate_host() * 100.0f) << "%"
                      << "|目標=" << (brain.get_target_sparsity_host() * 100.0f) << "%"
                      << "|損失=" << avg_loss << "|不確定性=" << (avg_unc * 100.0f) << "%"
                      << "|LR門檻=" << lr_sched.last_threshold << "|LR倍率=" << lr_sched.last_factor
                      << "|可塑性=" << (lr_sched.last_plasticity * 100.0f) << "%"
                      << "|平均權重(|W|)=" << wabs.mean_all << "|各層(|W|)in/m1/m2/out=" << wabs.proj_in << "/"
                      << wabs.proj_mid1 << "/" << wabs.proj_mid2 << "/" << wabs.proj_out
                      << "|Glial閾值=" << brain.get_glial_threshold_host() << "|學習率=" << LR
                      << "|耗時=" << epoch_duration << "ms\n";

            if (has_val)
            {
                std::cout << "驗證：準確率=" << val_acc << "%|損失=" << val_loss << "|不確定性=" << (val_unc * 100.0f)
                          << "%|樣本=" << val_samples << "\n";
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
                          << "|更新(裝置sum)=" << (step_dev_sum_ms_sum * inv) << "ms\n";
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
                    std::string path = ckpt_dir + "/mnist_surrogate_last.pnbckpt";
                    if (save_checkpoint_file(path, ckpt, static_cast<uint32_t>(epoch), seed, LR, accuracy, avg_loss,
                                             avg_unc, has_val, val_acc, val_loss, val_unc))
                    {
                        std::cout << "保存檢查點last:" << path << "\n";
                    }
                    else
                    {
                        std::cout << "保存檢查點失敗last:" << path << "\n";
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

                if (ckpt_save_best && is_best)
                {
                    best_val_acc = has_val ? val_acc : accuracy;
                    best_val_loss = has_val ? val_loss : avg_loss;
                    std::string path = ckpt_dir + "/mnist_surrogate_best.pnbckpt";
                    if (save_checkpoint_file(path, ckpt, static_cast<uint32_t>(epoch), seed, LR, accuracy, avg_loss,
                                             avg_unc, has_val, val_acc, val_loss, val_unc))
                    {
                        std::cout << "保存檢查點best:" << path << "\n";
                    }
                    else
                    {
                        std::cout << "保存檢查點失敗best:" << path << "\n";
                    }
                }

                if (ckpt_every > 0 && (epoch % ckpt_every) == 0)
                {
                    std::string path = ckpt_dir + "/mnist_surrogate_epoch" + std::to_string(epoch) + ".pnbckpt";
                    if (save_checkpoint_file(path, ckpt, static_cast<uint32_t>(epoch), seed, LR, accuracy, avg_loss,
                                             avg_unc, has_val, val_acc, val_loss, val_unc))
                    {
                        std::cout << "保存檢查點epoch:" << path << "\n";
                    }
                    else
                    {
                        std::cout << "保存檢查點失敗epoch:" << path << "\n";
                    }
                }
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
        std::cout << "SYCL錯誤：" << e.what() << "\n";
        return 1;
    }
    return 0;
}
