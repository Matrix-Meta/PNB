/*
 * Copyright 2025 Project Neuro-Bit Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "neurobit/core/types.hpp"
#include "neurobit/layers/bit_brain.hpp"
#include <vector>
#include <fstream>
#include <sycl/sycl.hpp>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <random>

namespace s = sycl;

struct CheckpointMeta { uint32_t epoch = 0; float val_acc = 0.0f; };
template <class T> static bool read_pod(std::istream &is, T &v) { is.read(reinterpret_cast<char *>(&v), sizeof(T)); return static_cast<bool>(is); }
static bool read_bytes(std::istream &is, void *ptr, size_t bytes) { if (bytes == 0) return true; is.read(reinterpret_cast<char *>(ptr), bytes); return static_cast<bool>(is); }
static bool read_vec_f32(std::istream &is, std::vector<float> &v) {
    uint64_t n = 0; if (!read_pod(is, n)) return false;
    v.resize(n); return read_bytes(is, v.data(), n * sizeof(float));
}

static bool load_checkpoint(const std::string &path, neurobit::layers::BitBrainLayer::CheckpointState &st, CheckpointMeta &meta) {
    std::ifstream is(path, std::ios::binary); if (!is) return false;
    is.seekg(12); uint64_t d[7]; for(int i=0; i<7; ++i) read_pod(is, d[i]);
    st.batch_size=d[0]; st.seq_len=d[1]; st.input_dim=d[2]; st.hidden_dim=d[3]; st.output_dim=d[4];
    st.md_group_size=d[5]; st.md_group_count=d[6];
    uint32_t ep, sd, flags; float lr, ta, tl, tu, va, vl, vu;
    read_pod(is, ep); read_pod(is, sd); read_pod(is, lr); read_pod(is, ta); read_pod(is, tl); read_pod(is, tu); read_pod(is, va); read_pod(is, vl); read_pod(is, vu); read_pod(is, flags);
    meta.val_acc = va;
    is.read(reinterpret_cast<char*>(&st.glial_state), 12 * sizeof(float));
    read_pod(is, st.target_sparsity); read_pod(is, st.glial_threshold_snapshot); read_pod(is, st.hippocampus_learned_threshold);
    for (float &w : st.neuromodulator_w) read_pod(is, w);
    is.seekg(7 * 4, std::ios::cur);
    read_pod(is, st.md_vigilance); read_pod(is, st.md_tau); read_pod(is, st.md_alpha); read_pod(is, st.md_novelty_power); read_pod(is, st.md_novelty_sharpness);
    uint8_t ub; read_pod(is, ub);
    st.proj_in_use_bias=(ub&1); st.proj_mid1_use_bias=(ub&2); st.proj_mid2_use_bias=(ub&4); st.proj_out_use_bias=(ub&8);
    read_vec_f32(is, st.md_group_offsets); read_vec_f32(is, st.proj_in_weight); read_vec_f32(is, st.proj_in_bias);
    read_vec_f32(is, st.proj_mid1_weight); read_vec_f32(is, st.proj_mid1_bias); read_vec_f32(is, st.proj_mid2_weight); read_vec_f32(is, st.proj_mid2_bias);
    read_vec_f32(is, st.proj_out_weight); read_vec_f32(is, st.proj_out_bias);
    read_vec_f32(is, st.w_fast); read_vec_f32(is, st.w_slow);
    read_vec_f32(is, st.ssm_A); read_vec_f32(is, st.ssm_state);
    return true;
}

int main(int argc, char* argv[]) {
    using namespace neurobit;
    std::string model_path = (argc > 1) ? argv[1] : "exports/mnist_surrogate_best.bin";

    std::cout << "\n=== PNB-X Marathon Reasoning (10,000 Steps) Test ===\n" << std::endl;
    std::cout << "Loading Model: " << model_path << std::endl;

    try {
        s::queue q{s::gpu_selector_v};
        layers::BitBrainLayer::CheckpointState st; CheckpointMeta meta;
        if (!load_checkpoint(model_path, st, meta)) throw std::runtime_error("Load failed: " + model_path);

        std::cout << "Model loaded. Checkpoint Val Acc: " << meta.val_acc << "%" << std::endl;

        layers::BitBrainLayer::Config cfg;
        cfg.input_dim=st.input_dim; cfg.hidden_dim=st.hidden_dim; cfg.output_dim=st.output_dim;
        cfg.batch_size=st.batch_size; cfg.seq_len=st.seq_len; cfg.md_silu_threshold_group_size=st.md_group_size;
        cfg.vigilance=st.md_vigilance; cfg.md_silu_tau=st.md_tau; cfg.md_silu_alpha=st.md_alpha;
        cfg.md_silu_novelty_power=st.md_novelty_power; cfg.md_silu_novelty_sharpness=st.md_novelty_sharpness;
        cfg.enable_neuromodulator = true;

        layers::BitBrainLayer brain(q, cfg); brain.set_effective_batch_size(1);
        s::buffer<float, 1> buf_wf{s::range<1>(st.input_dim * st.hidden_dim)}, buf_ws{s::range<1>(st.input_dim * st.hidden_dim)};
        brain.import_checkpoint_host(st, buf_wf, buf_ws);
        
        // 確保載入的目標活躍率被應用
        brain.get_glial().set_target_sparsity(st.target_sparsity);

        std::ifstream fi("data/t10k-images-idx3-ubyte", std::ios::binary); fi.seekg(16);
        std::vector<unsigned char> pix(784); fi.read((char*)pix.data(), 784);
        std::vector<float> img_clean(784); for(int p=0; p<784; ++p) img_clean[p] = (float)pix[p];

        s::buffer<float, 1> b_in{s::range<1>(784)}, b_out{s::range<1>(10)};
        auto push_img = [&](const std::vector<float>& img) {
            s::host_accessor acc(b_in, s::write_only);
            for(int p=0; p<784; ++p) acc[p] = (img[p] / 255.0f - 0.1307f) / 0.3081f;
        };

        auto run_marathon = [&](float goal) {
            int steps = 0; float conf = 0;
            // Cooldown buffer (Blank image)
            s::buffer<float, 1> b_blank{s::range<1>(784)};
            { s::host_accessor acc(b_blank, s::write_only); for(int i=0; i<784; ++i) acc[i] = -0.1307f/0.3081f; } // Normalized 0

            while(conf < goal && steps < 10000) { 
                // Cooldown / Blink: Every 500 steps, rest for 10 steps
                if(steps > 0 && steps % 500 == 0) {
                    std::cout << "[Blinking...] Resting for 10 steps." << std::endl;
                    for(int k=0; k<10; ++k) {
                        brain.forward(b_blank, b_out, buf_wf, buf_ws);
                        brain.get_glial().set_threshold(brain.get_glial_threshold_host() * 0.95f); // Metabolize threshold
                        q.wait();
                    }
                }

                brain.forward(b_in, b_out, buf_wf, buf_ws); q.wait();
                std::vector<float> logits(10);
                { s::host_accessor acc(b_out, s::read_only); std::copy(acc.get_pointer(), acc.get_pointer()+10, logits.begin()); }
                conf = *std::max_element(logits.begin(), logits.end());
                steps++;
                // 每 1000 步輸出一次日誌
                if(steps % 1000 == 0) std::cout << "Thinking... Step " << steps << " | Current Conf: " << conf << std::endl;
            }
            return std::make_pair(steps, conf);
        };

        std::vector<float> noise_levels = {0.5f, 0.8f};
        float GOAL = 3.0f; // 調高門檻，要求極高信心度

        for(float level : noise_levels) {
            std::cout << "\n>>> Testing Noise Level: " << (int)(level*100) << "%" << std::endl;
            std::mt19937 rng(42); std::normal_distribution<float> dist(0, level * 255.0f);
            std::vector<float> img_noisy = img_clean;
            for(float &p : img_noisy) p = std::clamp(p + dist(rng), 0.0f, 255.0f);
            push_img(img_noisy);

            // 1. Standard
            brain.import_checkpoint_host(st, buf_wf, buf_ws);
            brain.get_glial().set_threshold(st.glial_state.current_threshold);
            auto r1 = run_marathon(GOAL);
            std::cout << "[Standard] Steps: " << r1.first << " | Final Conf: " << r1.second << " | Threshold: " << brain.get_glial_threshold_host() << std::endl;

            // 2. Learning
            size_t h = st.hidden_dim; s::buffer<float, 1> b_sp{s::range<1>(h)};
            q.submit([&](s::handler& c){ s::accessor sp{b_sp, c, s::write_only}; c.parallel_for(h, [=](s::id<1> idx){ sp[idx]=(idx<h/4)?15.0f:0.0f; }); }).wait();
            brain.get_hippocampus().learn(q, b_in, b_sp, buf_wf, 1.0f); q.wait();

            // 3. CLS
            auto r2 = run_marathon(GOAL);
            std::cout << "[CLS-Aug]  Steps: " << r2.first << " | Final Conf: " << r2.second << " | Threshold: " << brain.get_glial_threshold_host() << std::endl;
        }

    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << std::endl; }
    return 0;
}
