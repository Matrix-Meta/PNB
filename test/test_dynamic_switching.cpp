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

void load_mnist_sample(const std::string &path, int idx, std::vector<float> &img) {
    std::ifstream f(path, std::ios::binary); f.seekg(16 + idx * 784);
    std::vector<unsigned char> b(784); f.read((char*)b.data(), 784);
    img.resize(784); for(int i=0; i<784; ++i) img[i] = (float)b[i];
}

int main() {
    using namespace neurobit;
    std::cout << "
=== PNB-X Dynamic Scene Switching Marathon Test ===
" << std::endl;
    try {
        s::queue q{s::gpu_selector_v};
        layers::BitBrainLayer::CheckpointState st; CheckpointMeta meta;
        if(!load_checkpoint("exports/mnist_surrogate_best.bin", st, meta)) throw std::runtime_error("Load failed");

        layers::BitBrainLayer::Config cfg;
        cfg.input_dim=st.input_dim; cfg.hidden_dim=st.hidden_dim; cfg.output_dim=st.output_dim;
        cfg.md_silu_threshold_group_size=st.md_group_size;
        cfg.vigilance=st.md_vigilance; cfg.md_silu_tau=st.md_tau; cfg.md_silu_alpha=st.md_alpha;
        cfg.md_silu_novelty_power=st.md_novelty_power; cfg.md_silu_novelty_sharpness=st.md_novelty_sharpness;
        cfg.enable_neuromodulator = true;

        layers::BitBrainLayer brain(q, cfg); brain.set_effective_batch_size(1);
        s::buffer<float, 1> buf_wf{s::range<1>(st.input_dim * st.hidden_dim)}, buf_ws{s::range<1>(st.input_dim * st.hidden_dim)};
        brain.import_checkpoint_host(st, buf_wf, buf_ws);

        std::vector<float> iA_raw, iB_raw;
        load_mnist_sample("data/t10k-images-idx3-ubyte", 0, iA_raw); 
        load_mnist_sample("data/t10k-images-idx3-ubyte", 1, iB_raw);
        
        std::mt19937 rng(42); std::normal_distribution<float> noise(0, 76.5f); // 30% Noise
        auto add_n = [&](std::vector<float> v) { for(float &p:v) p=std::clamp(p+noise(rng), 0.0f, 255.0f); return v; };
        std::vector<float> iA_noisy = add_n(iA_raw), iB_noisy = add_n(iB_raw);

        s::buffer<float, 1> b_in{s::range<1>(784)}, b_out{s::range<1>(10)};
        auto run_reasoning = [&](const std::vector<float>& img, const std::string& msg, float goal) {
            { s::host_accessor acc(b_in, s::write_only); for(int p=0; p<784; ++p) acc[p]= (img[p]/255.0f - 0.1307f)/0.3081f; }
            int steps = 0; float conf = 0;
            // 重置膠質細胞到基準閾值
            brain.get_glial().set_threshold(st.glial_state.current_threshold);
            
            while(conf < goal && steps < 1000) {
                brain.forward(b_in, b_out, buf_wf, buf_ws); q.wait();
                std::vector<float> logits(10);
                { s::host_accessor acc(b_out, s::read_only); std::copy(acc.get_pointer(), acc.get_pointer()+10, logits.begin()); }
                conf = *std::max_element(logits.begin(), logits.end());
                steps++;
            }
            std::cout << ">>> [" << msg << "] Steps: " << steps << " | Conf: " << conf << std::endl;
            return steps;
        };

        float GOAL = 0.7f;

        // 1. Clean Baseline
        std::cout << "Phase 1: Clean Baseline (Image 0)..." << std::endl;
        run_reasoning(iA_raw, "Clean Baseline", GOAL);

        // 2. Encounter Scene A (Noisy)
        std::cout << "
Phase 2: Encountering Scene A (30% Noise)..." << std::endl;
        run_reasoning(iA_noisy, "Scene A (First time)", GOAL);
        
        std::cout << "Learning Scene A (One-shot)..." << std::endl;
        size_t hdim = st.hidden_dim; s::buffer<float, 1> b_sp{s::range<1>(hdim)};
        q.submit([&](s::handler& c){ s::accessor sp{b_sp, c, s::write_only}; c.parallel_for(hdim, [=](s::id<1> idx){ sp[idx]=(idx<hdim/4)?15.0f:0.0f; }); }).wait();
        brain.get_hippocampus().learn(q, b_in, b_sp, buf_wf, 1.0f); q.wait();
        run_reasoning(iA_noisy, "Scene A (Learned)", GOAL);

        // 3. Switch to Scene B (Noisy)
        std::cout << "
Phase 3: Switching to Scene B (Image 1, 30% Noise)..." << std::endl;
        run_reasoning(iB_noisy, "Scene B (First time)", GOAL);
        std::cout << "Learning Scene B (One-shot)..." << std::endl;
        brain.get_hippocampus().learn(q, b_in, b_sp, buf_wf, 1.0f); q.wait();
        run_reasoning(iB_noisy, "Scene B (Learned)", GOAL);

        // 4. Recall Scene A
        std::cout << "
Phase 4: Switching BACK to Scene A (Retention Check)..." << std::endl;
        int final_a = run_reasoning(iA_noisy, "Scene A (Recall)", GOAL);

        // 5. Consolidation (Sleep)
        std::cout << "
Phase 5: Consolidating Memory (Sleep Cycle)..." << std::endl;
        brain.get_hippocampus().consolidate(q, buf_wf, buf_ws);
        q.wait();

        std::cout << "Clearing Short-term Memory (W_fast = 0) to verify long-term transfer..." << std::endl;
        size_t total_w = st.input_dim * st.hidden_dim;
        q.submit([&](s::handler& c){ s::accessor wf{buf_wf, c, s::write_only}; c.parallel_for(total_w, [=](s::id<1> idx){ wf[idx]=0.0f; }); }).wait();

        std::cout << "
Phase 6: Final Long-term Recall Test (No W_fast):" << std::endl;
        float c_a = run_reasoning(iA_noisy, "Scene A (Long-term)", GOAL);
        float c_b = run_reasoning(iB_noisy, "Scene B (Long-term)", GOAL);

        std::cout << "
--- Final Consolidation Summary ---" << std::endl;
        if(c_a == 1 && c_b == 1) {
            std::cout << "✅ SUCCESS: All memories successfully transferred to Neocortex (W_slow)!" << std::endl;
        } else {
            std::cout << "⚠️ NOTICE: Long-term recall requires " << c_a << "/" << c_b << " steps." << std::endl;
        }

    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << std::endl; }
    return 0;
}
