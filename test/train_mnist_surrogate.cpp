#include "neurobit/core/types.hpp"
#include "neurobit/layers/bit_brain.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
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
        throw std::runtime_error("Cannot open image file: " + image_path);

    uint32_t magic = 0, num_imgs = 0, rows = 0, cols = 0;
    img_file.read((char *)&magic, 4);
    img_file.read((char *)&num_imgs, 4);
    img_file.read((char *)&rows, 4);
    img_file.read((char *)&cols, 4);

    magic = swap_endian(magic);
    num_imgs = swap_endian(num_imgs);
    rows = swap_endian(rows);
    cols = swap_endian(cols);

    std::cout << "Loading " << num_imgs << " images [" << rows << "x" << cols << "] from " << image_path << "...\n";

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
        throw std::runtime_error("Cannot open label file: " + label_path);

    uint32_t magic_l = 0, num_lbls = 0;
    lbl_file.read((char *)&magic_l, 4);
    lbl_file.read((char *)&num_lbls, 4);
    num_lbls = swap_endian(num_lbls);

    if (num_imgs != num_lbls)
        throw std::runtime_error("Image/Label count mismatch!");

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

int main(int argc, char **argv)
{
    size_t In = 784;
    size_t Hidden = 512;
    size_t Out = 10;
    float LR = 0.01f;
    size_t BATCH_SIZE = 64;
    int epochs = 20;

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
    }

    std::cout << "Initializing PNB BitBrain (784 -> 512 -> 10) on REAL MNIST (Batch Size: " << BATCH_SIZE << ")...\n";
    std::cout << "Architecture: BitNet b1.58 + SSM + M-DSiLU + Glial + Hippocampus\n";

    MNISTData data;

    try
    {
        load_mnist("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", data);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to load MNIST: " << e.what() << "\n";
        return 1;
    }

    try
    {
        s::queue q;
        std::cout << "Running on: " << q.get_device().get_info<s::info::device::name>() << "\n";

        // BitBrain Config
        neurobit::layers::BitBrainLayer::Config cfg;
        cfg.batch_size = BATCH_SIZE;
        cfg.input_dim = In;
        cfg.hidden_dim = Hidden;
        cfg.output_dim = Out;
        // Glial settings
        cfg.glial_config.initial_threshold = 0.0f;
        cfg.glial_config.target_sparsity = 0.5f;
        cfg.glial_config.min_threshold = -1.0f;
        cfg.glial_config.initial_lr = 0.01f;
        cfg.glial_config.max_lr = 0.1f;
        cfg.glial_config.lr_growth = 1.01f;
        cfg.glial_config.lr_shrink = 0.99f;
        cfg.glial_config.max_threshold = 3.0f; // Increased limit for threshold
        cfg.vigilance = 0.5f;                  // M-DSiLU vigilance
        cfg.glial_target_novelty_gain = 0.3f;
        cfg.glial_priming_rate = 0.1f;

        // The Brain
        neurobit::layers::BitBrainLayer brain(q, cfg);

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

        int total_correct = 0;
        int samples_processed = 0;

        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            auto epoch_start_time = std::chrono::high_resolution_clock::now();
            total_correct = 0;
            samples_processed = 0;
            double sparsity_sum = 0.0;
            size_t sparsity_batches = 0;

            for (size_t i = 0; i < data.images.size(); i += BATCH_SIZE)
            {
                size_t current_batch = std::min(BATCH_SIZE, data.images.size() - i);
                if (current_batch != BATCH_SIZE)
                    break;

                // 1. Prepare Batch
                {
                    s::host_accessor acc_X{buf_X, s::write_only};
                    for (size_t b = 0; b < current_batch; ++b)
                    {
                        std::copy(data.images[i + b].begin(), data.images[i + b].end(), acc_X.get_pointer() + b * In);
                    }
                }

                // 2. Forward
                brain.forward(buf_X, buf_Out, buf_W_fast, buf_W_slow);
                {
                    auto &glial = brain.get_glial();
                    sparsity_sum += glial.get_last_sparsity();
                    sparsity_batches++;
                }

                // 3. Loss & Gradient
                {
                    s::host_accessor acc_Out{buf_Out, s::read_only};
                    s::host_accessor acc_Grad{buf_Grad_Out, s::write_only};

                    for (size_t b = 0; b < current_batch; ++b)
                    {
                        int label = data.labels[i + b];
                        const float *out_ptr = acc_Out.get_pointer() + b * Out;
                        float *grad_ptr = acc_Grad.get_pointer() + b * Out;

                        // Softmax
                        float max_val = -1e9f;
                        for (int k = 0; k < Out; ++k)
                            if (out_ptr[k] > max_val)
                                max_val = out_ptr[k];

                        float sum_exp = 0.0f;
                        std::vector<float> probs(Out);
                        for (int k = 0; k < Out; ++k)
                        {
                            probs[k] = std::exp(out_ptr[k] - max_val);
                            sum_exp += probs[k];
                        }

                        int pred = -1;
                        float max_prob = -1.0f;
                        for (int k = 0; k < Out; ++k)
                        {
                            probs[k] /= sum_exp;
                            if (probs[k] > max_prob)
                            {
                                max_prob = probs[k];
                                pred = k;
                            }

                            float target = (k == label) ? 1.0f : 0.0f;
                            grad_ptr[k] = (probs[k] - target) / current_batch;
                        }
                        if (pred == label)
                            total_correct++;
                    }
                }
                samples_processed += current_batch;

                // 4. Backward
                brain.backward(buf_Grad_Out, buf_Grad_In);

                // 5. Update
                brain.step(LR, 0.0001f);
            }
            q.wait(); // Ensure all GPU operations are done before timing

            auto epoch_end_time = std::chrono::high_resolution_clock::now();
            auto epoch_duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end_time - epoch_start_time).count();

            // Stats
            float accuracy = (float)total_correct / samples_processed * 100.0f;
            auto &glial = brain.get_glial();
            float avg_active =
                sparsity_batches ? static_cast<float>(sparsity_sum / static_cast<double>(sparsity_batches)) : 0.0f;

            std::cout << "Epoch " << epoch << ": Accuracy = " << accuracy << "%"
                      << " | Active: " << (avg_active * 100.0f) << "%"
                      << " | Target: " << (glial.get_target_sparsity() * 100.0f) << "%"
                      << " | Glial Th: " << glial.get_threshold() << " | LR: " << LR << " | Time: " << epoch_duration
                      << "ms\n";

            if (glial.is_stable())
                LR *= 0.5f;
        }
    }
    catch (const s::exception &e)
    {
        std::cout << "SYCL Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
