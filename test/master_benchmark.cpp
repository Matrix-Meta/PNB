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

#include "neurobit/components/glial.hpp"
#include "neurobit/components/hippocampus.hpp"
#include "neurobit/core/types.hpp"
#include "neurobit/core/xmx.hpp"
#include "neurobit/layers/bit_brain.hpp"
#include "neurobit/layers/bit_linear.hpp"
#include "neurobit/layers/spike_neuron.hpp"
#include "neurobit/layers/ssm_scan.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <sycl/sycl.hpp>
#include <tuple>
#include <vector>

namespace s = sycl;

// ============================================================
// Test Framework Utilities
// ============================================================
struct TestResult
{
    std::string name;
    bool passed;
    std::string message;
};

static int total_tests = 0;
static int passed_tests = 0;

void report_test(const TestResult &result)
{
    total_tests++;
    if (result.passed)
    {
        passed_tests++;
        std::cout << "  [PASS] " << result.name << "
";
    }
    else
    {
        std::cout << "  [FAIL] " << result.name << " - " << result.message << "
";
    }
}

// ============================================================
// Test 1: Types and Constants
// ============================================================
void test_types()
{
    std::cout << "
=== Test 1: Types and Constants ===
";

    // Test PI constant
    {
        bool passed = std::abs(neurobit::PI - 3.14159265f) < 1e-5f;
        report_test({"PI constant", passed, "Value mismatch"});
    }

    // Test EPSILON constant
    {
        bool passed = neurobit::EPSILON == 1e-6f;
        report_test({"EPSILON constant", passed, "Value mismatch"});
    }

    // Test stable_coeff (tanh)
    {
        float result = neurobit::stable_coeff(0.0f);
        bool passed = std::abs(result) < 1e-6f;
        report_test({"stable_coeff(0) = 0", passed, "Expected 0"});
    }

    {
        float result = neurobit::stable_coeff(100.0f);
        bool passed = std::abs(result - 1.0f) < 1e-5f;
        report_test({"stable_coeff(large) ≈ 1", passed, "Expected ~1"});
    }

    // Test Vector1D type exists
    {
        float data[4] = {1, 2, 3, 4};
        neurobit::Vector1D<float> vec(data, 4);
        bool passed = vec[0] == 1.0f && vec[3] == 4.0f;
        report_test({"Vector1D type", passed, "Access failed"});
    }

    // Test MatrixView type
    {
        float data[6] = {1, 2, 3, 4, 5, 6};
        neurobit::MatrixView<float> mat(data, 2, 3);
        bool passed = mat[0, 0] == 1.0f && mat[1, 2] == 6.0f;
        report_test({"MatrixView type", passed, "Access failed"});
    }

    // Test Tensor3D type
    {
        float data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        neurobit::Tensor3D<float> tensor(data, 2, 2, 2);
        bool passed = tensor[0, 0, 0] == 1.0f && tensor[1, 1, 1] == 8.0f;
        report_test({"Tensor3D type", passed, "Access failed"});
    }
}

// ============================================================
// Test 2: BitLinearXMX Layer
// ============================================================
void test_bit_linear_xmx(s::queue &q)
{
    std::cout << "
=== Test 2: BitLinearXMX Layer (BF16) ===
";

    using namespace neurobit::layers;
    using bfloat16 = sycl::ext::oneapi::bfloat16;

    // Small scale functional test
    {
        const size_t batch = 2;
        const size_t in_features = 4;
        const size_t out_features = 3;

        BitLinearXMX<bfloat16> layer(q, in_features, out_features);

        // Weights: alternating +1, -1
        std::vector<int8_t> h_W(in_features * out_features);
        for (size_t i = 0; i < h_W.size(); i++)
        {
            h_W[i] = (i % 2 == 0) ? 1 : -1;
        }
        layer.set_weights(h_W);

        // Input: all 1.0
        std::vector<bfloat16> h_X(batch * in_features, bfloat16(1.0f));
        std::vector<bfloat16> h_Y(batch * out_features, bfloat16(0.0f));

        s::buffer<bfloat16, 1> buf_X{h_X.data(), s::range<1>(h_X.size())};
        s::buffer<bfloat16, 1> buf_Y{h_Y.data(), s::range<1>(h_Y.size())};

        layer.forward(buf_X, buf_Y, batch);
        q.wait();

        // Verify
        {
            s::host_accessor acc(buf_Y, s::read_only);
            bool passed = true;
            for (size_t b = 0; b < batch && passed; b++)
            {
                for (size_t n = 0; n < out_features && passed; n++)
                {
                    float expected = 0.0f;
                    for (size_t k = 0; k < in_features; k++)
                    {
                        expected += static_cast<float>(h_W[k * out_features + n]);
                    }
                    float actual = static_cast<float>(acc[b * out_features + n]);
                    // Strict check for small integer results
                    if (std::abs(actual - expected) > 1e-3f)
                    {
                        passed = false;
                        std::cout << "    Mismatch at [" << b << "," << n << "]: expected " << expected << ", got "
                                  << actual << "
";
                    }
                }
            }
            report_test({"BitLinearXMX forward", passed, "Output mismatch"});
        }
    }

    // Performance smoke test (small scale)
    {
        const size_t batch = 16;
        const size_t in_features = 512;
        const size_t out_features = 256;

        BitLinearXMX<bfloat16> layer(q, in_features, out_features);
        std::vector<int8_t> h_W(in_features * out_features, 1);
        layer.set_weights(h_W);
        std::vector<bfloat16> h_X(batch * in_features, bfloat16(0.1f));
        std::vector<bfloat16> h_Y(batch * out_features);
        s::buffer<bfloat16, 1> buf_X{h_X.data(), s::range<1>(h_X.size())};
        s::buffer<bfloat16, 1> buf_Y{h_Y.data(), s::range<1>(h_Y.size())};

        // Warmup
        layer.forward(buf_X, buf_Y, batch);
        q.wait();

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; i++)
        {
            layer.forward(buf_X, buf_Y, batch);
        }
        q.wait();
        auto end = std::chrono::high_resolution_clock::now();

        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        double avg_us = duration_us / 10.0;
        double gflops = (batch * in_features * out_features * 2.0) / (avg_us / 1e6) / 1e9;

        std::cout << "  Performance smoke test: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS
";
        report_test({"BitLinearXMX performance smoke test", true, ""});
    }
}

// ============================================================
// Test 2b: SLM Tiling Comparison
// ============================================================
void test_slm_tiling_comparison(s::queue &q)
{
    std::cout << "
=== Test 2b: SLM Tiling Optimization Comparison ===
";

    using namespace neurobit::layers;
    using bfloat16 = sycl::ext::oneapi::bfloat16;

    struct TestConfig
    {
        size_t batch;
        size_t M;
        size_t N;
        std::string name;
    };

    std::vector<TestConfig> configs = {
        {64, 1024, 1024, "Medium Batch (64)"},
    };

    for (const auto &config : configs)
    {
        std::cout << "Config: " << config.name << "
";

        std::vector<bfloat16> h_X(config.batch * config.M, bfloat16(0.5f));
        std::vector<int8_t> h_W(config.M * config.N, 1);
        std::vector<bfloat16> h_Y_basic(config.batch * config.N);
        std::vector<bfloat16> h_Y_slm(config.batch * config.N);

        s::buffer<bfloat16> buf_X{h_X.data(), s::range<1>(h_X.size())};
        s::buffer<bfloat16> buf_Y_basic{h_Y_basic.data(), s::range<1>(h_Y_basic.size())};
        s::buffer<bfloat16> buf_Y_slm{h_Y_slm.data(), s::range<1>(h_Y_slm.size())};

        double time_basic = 0.0;
        double time_slm = 0.0;
        const int test_iters = 20;

        // Basic XMX
        {
            BitLinearXMX<bfloat16> layer(q, config.M, config.N, false); // Explicitly disable SLM
            layer.set_weights(h_W);
            for (int i = 0; i < 3; ++i)
                layer.forward(buf_X, buf_Y_basic, config.batch);
            q.wait();

            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < test_iters; i++)
                layer.forward(buf_X, buf_Y_basic, config.batch);
            q.wait();
            auto end = std::chrono::high_resolution_clock::now();
            time_basic =
                std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (test_iters * 1000.0);
        }

        // XMX + SLM Tiling
        {
            // enable_slm_tiling = true
            neurobit::layers::BitLinearXMX<bfloat16> layer(q, config.M, config.N, true);
            layer.set_weights(h_W);
            for (int i = 0; i < 3; ++i)
                layer.forward(buf_X, buf_Y_slm, config.batch);
            q.wait();

            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < test_iters; i++)
                layer.forward(buf_X, buf_Y_slm, config.batch);
            q.wait();
            auto end = std::chrono::high_resolution_clock::now();
            time_slm =
                std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (test_iters * 1000.0);
        }

        double speedup = time_basic / time_slm;
        std::cout << "  Basic: " << time_basic << " ms | SLM: " << time_slm << " ms | Speedup: " << speedup << "x
";

        bool passed = speedup > 0.9; // Allow small regression, expected improvement
        report_test({config.name + " SLM Tiling", passed, "Speedup check"});
    }
}

// ============================================================
// Test 3: SSMScan Layer
// ============================================================
void test_ssm_scan(s::queue &q)
{
    std::cout << "
=== Test 3: SSMScan Layer ===
";

    using namespace neurobit::layers;

    const size_t batch = 2;
    const size_t seq_len = 4;
    const size_t state_dim = 3;

    SSMScan::Config cfg{batch, seq_len, state_dim, true};
    SSMScan layer(cfg);

    std::vector<float> h_X(batch * seq_len * state_dim, 1.0f);
    std::vector<float> h_A(state_dim, 0.9f);
    std::vector<float> h_Y(batch * seq_len * state_dim, 0.0f);
    std::vector<float> h_State(batch * state_dim, 0.0f);

    s::buffer<float, 1> buf_X{h_X.data(), s::range<1>(h_X.size())};
    s::buffer<float, 1> buf_A{h_A.data(), s::range<1>(h_A.size())};
    s::buffer<float, 1> buf_Y{h_Y.data(), s::range<1>(h_Y.size())};
    s::buffer<float, 1> buf_State{h_State.data(), s::range<1>(h_State.size())};

    layer.forward(q, buf_X, buf_A, buf_Y, buf_State);
    q.wait();

    {
        s::host_accessor acc(buf_Y, s::read_only);
        bool has_nonzero = false;
        for (size_t i = 0; i < h_Y.size(); i++)
        {
            if (std::abs(acc[i]) > 1e-6f)
            {
                has_nonzero = true;
                break;
            }
        }
        report_test({"SSMScan forward", has_nonzero, "All outputs are zero"});
    }
}

// ============================================================
// Test 4: SpikeNeuron Layer
// ============================================================
void test_spike_neuron(s::queue &q)
{
    std::cout << "
=== Test 4: SpikeNeuron Layer ===
";

    using namespace neurobit::layers;

    const size_t batch = 2;
    const size_t seq_len = 4;
    const size_t state_dim = 8;

    SpikeNeuron::Config cfg{batch, seq_len, state_dim, 1.0f, 0.5f, true};
    SpikeNeuron layer(q, cfg);

    // High input -> expect spikes
    std::vector<float> h_X(batch * state_dim, 2.0f);
    std::vector<float> h_V(batch * state_dim, 0.0f);
    std::vector<float> h_Z(batch * state_dim, 0.0f);
    std::vector<float> h_OU(batch * state_dim, 0.0f);
    std::vector<int> h_Activity(1, 0);

    s::buffer<float, 1> buf_X{h_X.data(), s::range<1>(h_X.size())};
    s::buffer<float, 1> buf_V{h_V.data(), s::range<1>(h_V.size())};
    s::buffer<float, 1> buf_Z{h_Z.data(), s::range<1>(h_Z.size())};
    s::buffer<float, 1> buf_OU{h_OU.data(), s::range<1>(h_OU.size())};
    s::buffer<int, 1> buf_Activity{h_Activity.data(), s::range<1>(1)};

    layer.forward_single(q, buf_X, buf_V, buf_Z, buf_OU, buf_Activity);
    q.wait();

    {
        s::host_accessor acc_activity(buf_Activity, s::read_only);
        int total_spikes = acc_activity[0];
        bool passed = total_spikes > 0;
        report_test(
            {"SpikeNeuron forward_single", passed, "No spikes generated (got " + std::to_string(total_spikes) + ")"});
    }
}

// ============================================================
// Test 5: GlialCell
// ============================================================
void test_glial_cell()
{
    std::cout << "
=== Test 5: GlialCell ===
";

    using namespace neurobit::components;

    GlialConfig cfg;
    cfg.initial_threshold = 1.0f;
    cfg.target_sparsity = 0.1f;
    cfg.min_threshold = 0.1f;
    cfg.max_threshold = 10.0f;

    GlialCell glial(cfg);

    // Too many spikes -> increase threshold
    {
        int total_spikes = 500; // 50%
        size_t total_neurons = 1000;
        float old_threshold = glial.get_threshold();
        float new_threshold = glial.regulate(total_spikes, total_neurons);
        bool passed = new_threshold > old_threshold;
        report_test({"GlialCell regulate (high sparsity)", passed, "Threshold did not increase"});
    }

    // Too few spikes -> decrease threshold
    {
        int total_spikes = 10; // 1%
        size_t total_neurons = 1000;
        float old_threshold = glial.get_threshold();
        float new_threshold = glial.regulate(total_spikes, total_neurons);
        bool passed = new_threshold < old_threshold;
        report_test({"GlialCell regulate (low sparsity)", passed, "Threshold did not decrease"});
    }
}

// ============================================================
// Test 6: Hippocampus
// ============================================================
void test_hippocampus(s::queue &q)
{
    std::cout << "
=== Test 6: Hippocampus ===
";

    using namespace neurobit::components;

    const size_t input_dim = 8;
    const size_t hidden_dim = 4;

    Hippocampus::Config cfg;
    cfg.batch_size = 1;
    cfg.input_dim = input_dim;
    cfg.hidden_dim = hidden_dim;
    cfg.learning_rate = 0.1f;

    Hippocampus hippo(cfg);

    std::vector<float> h_X(input_dim, 0.5f);
    std::vector<float> h_W_fast(input_dim * hidden_dim, 0.1f);
    std::vector<float> h_W_slow(input_dim * hidden_dim, 0.1f);

    s::buffer<float, 1> buf_X{h_X.data(), s::range<1>(h_X.size())};
    s::buffer<float, 1> buf_W_fast{h_W_fast.data(), s::range<1>(h_W_fast.size())};
    s::buffer<float, 1> buf_W_slow{h_W_slow.data(), s::range<1>(h_W_slow.size())};

    auto result = hippo.compute_familiarity(q, buf_X, buf_W_fast, buf_W_slow);
    bool passed = (result.familiarity_score >= 0.0f && result.familiarity_score <= 1.0f);
    report_test({"Hippocampus compute_familiarity", passed, "Invalid familiarity score"});

    std::vector<float> h_Z(hidden_dim, 1.0f);
    s::buffer<float, 1> buf_Z{h_Z.data(), s::range<1>(h_Z.size())};
    hippo.learn(q, buf_X, buf_Z, buf_W_fast, 1.0f);
    q.wait();
    report_test({"Hippocampus learn", true, ""});
}

// ============================================================
// Test 7: BitBrainLayer Integration
// ============================================================
void test_bit_brain(s::queue &q)
{
    std::cout << "
=== Test 7: BitBrainLayer Integration ===
";
    using namespace neurobit::layers;
    using namespace neurobit::components;

    BitBrainLayer::Config config;
    config.input_dim = 16;
    config.hidden_dim = 32;
    config.output_dim = 8;
    config.batch_size = 1;

    try
    {
        BitBrainLayer brain(q, config);
        report_test({"BitBrainLayer Construction", true, ""});
    }
    catch (const std::exception &e)
    {
        report_test({"BitBrainLayer Construction", false, e.what()});
    }
}

// ============================================================
// Main Function
// ============================================================
int main()
{
    try
    {
        s::queue q{s::gpu_selector_v};
        auto device = q.get_device();

        std::cout << "
========================================
";
        std::cout << "PNB Master Benchmark (Revised)
";
        std::cout << "========================================
";
        std::cout << "Device: " << device.get_info<s::info::device::name>() << "
";
        std::cout << "========================================
";

        // Phase 1: Functional Tests
        test_types();
        test_bit_linear_xmx(q);
        test_slm_tiling_comparison(q);
        test_ssm_scan(q);
        test_spike_neuron(q);
        test_glial_cell();
        test_hippocampus(q);
        test_bit_brain(q);

        std::cout << "
========================================
";
        std::cout << "Test Summary
";
        std::cout << "========================================
";
        std::cout << "Total: " << total_tests << "
";
        std::cout << "Passed: " << passed_tests << " (" << std::fixed << std::setprecision(1)
                  << (100.0 * passed_tests / total_tests) << "%)
";
        std::cout << "Failed: " << (total_tests - passed_tests) << "
";

        if (passed_tests != total_tests)
        {
            std::cout << "
[WARN] Skipping performance tests due to failures.
";
            return 1;
        }

        // Phase 2: Performance Scaling Tests
        std::cout << "
========================================
";
        std::cout << "Performance Scaling Tests
";
        std::cout << "========================================
";

        // BitLinearXMX Scaling
        std::cout << "
=== BitLinearXMX Scaling ===
";
        std::cout << "+---------------------+------------+-----------+------------+----------+----------+
";
        std::cout << "| Config              |   Elements |  Time(ms) |       TOPS |    GB/s  | Mode     |
";
        std::cout << "+---------------------+------------+-----------+------------+----------+----------+
";

        using bf16 = neurobit::layers::bfloat16;
        for (auto [b, m, n] : std::vector<std::tuple<size_t, size_t, size_t>>{{1, 256, 256},
                                                                              {1, 1024, 1024},
                                                                              {1, 4096, 4096},
                                                                              {16, 1024, 1024},
                                                                              {64, 1024, 1024},
                                                                              {256, 1024, 1024},
                                                                              {512, 2048, 2048},
                                                                              {1024, 2048, 2048},
                                                                              {2048, 4096, 4096},
                                                                              {4096, 4096, 4096},
                                                                              {4096, 8192, 8192},
                                                                              {8192, 8192, 8192},
                                                                              {16384, 16384, 16384}})
        {
            neurobit::layers::BitLinearXMX<bf16> layer(q, m, n, true);
            std::vector<int8_t> h_W(m * n, 1);
            layer.set_weights(h_W);
            std::vector<bf16> h_X(b * m, bf16(1.0f));
            std::vector<bf16> h_Y(b * n);
            s::buffer<bf16> buf_X{h_X.data(), s::range<1>(h_X.size())};
            s::buffer<bf16> buf_Y{h_Y.data(), s::range<1>(h_Y.size())};

            for (int i = 0; i < 3; ++i)
                layer.forward(buf_X, buf_Y, b);
            q.wait();

            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 10; ++i)
                layer.forward(buf_X, buf_Y, b);
            q.wait();
            auto end = std::chrono::high_resolution_clock::now();

            double time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 / 10.0;
            double ops = static_cast<double>(b) * n * m * 2.0;
            double tops = ops / (time_ms / 1000.0) / 1e12;
            double bytes = static_cast<double>(b) * m * 2 + static_cast<double>(m) * n + static_cast<double>(b) * n * 2;
            double gb_s = bytes / (time_ms / 1000.0) / 1e9;

            // Get mode string
            auto mode_enum = layer.get_mode(b);
            std::string mode_str = "UNKNOWN";
            if (mode_enum == neurobit::layers::BitLinearXMX<bf16>::Mode::Standard)
                mode_str = "STD";
            else if (mode_enum == neurobit::layers::BitLinearXMX<bf16>::Mode::XMX_Universal)
                mode_str = "XMX_UNI";
            else if (mode_enum == neurobit::layers::BitLinearXMX<bf16>::Mode::XMX_SLM_Medium)
                mode_str = "XMX_SLM_Medium";
            else if (mode_enum == neurobit::layers::BitLinearXMX<bf16>::Mode::XMX_SLM_Large)
                mode_str = "XMX_SLM_Large";

            char config[32];
            snprintf(config, sizeof(config), "%zux%zux%zu", b, m, n);
            printf("| %-19s | %10zu | %9.3f | %10.3f | %8.2f | %-8s |
", config, b * m * n, time_ms, tops, gb_s,
                   mode_str.c_str());
        }
        std::cout << "+---------------------+------------+-----------+------------+----------+----------+
";

        // SSMScan Scaling
        std::cout << "
=== SSMScan Scaling ===
";
        std::cout << "+---------------------+------------+-----------+--------------+
";
        std::cout << "| Config (BxLxD)      |   Elements |  Time(ms) |    M elem/s  |
";
        std::cout << "+---------------------+------------+-----------+--------------+
";

        for (auto [b, l, d] : std::vector<std::tuple<size_t, size_t, size_t>>{
                 {1, 1, 4096}, {1, 1, 65536}, {64, 1, 4096}, {256, 1, 4096}})
        {
            neurobit::layers::SSMScan::Config cfg{b, l, d, true};
            neurobit::layers::SSMScan layer(cfg);
            std::vector<float> h_X(b * d, 1.0f);
            std::vector<float> h_A(d, 0.5f);
            std::vector<float> h_Y(b * d);
            std::vector<float> h_State(b * d, 0.0f);
            s::buffer<float> buf_X{h_X.data(), s::range<1>(h_X.size())};
            s::buffer<float> buf_A{h_A.data(), s::range<1>(h_A.size())};
            s::buffer<float> buf_Y{h_Y.data(), s::range<1>(h_Y.size())};
            s::buffer<float> buf_State{h_State.data(), s::range<1>(h_State.size())};

            for (int i = 0; i < 3; ++i)
                layer.forward(q, buf_X, buf_A, buf_Y, buf_State);
            q.wait();

            auto start = std::chrono::high_resolution_clock::now();
            const int iters = 100;
            for (int i = 0; i < iters; ++i)
                layer.forward(q, buf_X, buf_A, buf_Y, buf_State);
            q.wait();
            auto end = std::chrono::high_resolution_clock::now();

            double total_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
            double avg_time_ms = total_time_ms / iters;
            // CORRECTED: Total elements processed over all iters / total time
            double total_elems = static_cast<double>(b) * d * iters;
            double m_elem_s = total_elems / (total_time_ms / 1000.0) / 1e6;

            char config[32];
            snprintf(config, sizeof(config), "%zux%zux%zu", b, l, d);
            printf("| %-19s | %10zu | %9.4f | %12.2f |
", config, b * l * d, avg_time_ms, m_elem_s);
        }
        std::cout << "+---------------------+------------+-----------+--------------+
";

        // SpikeNeuron Scaling
        std::cout << "
=== SpikeNeuron Scaling ===
";
        std::cout << "+-----------------+------------+-----------+--------------+------------+
";
        std::cout << "| Config (BxN)    |    Neurons |  Time(ms) |  M neurons/s |     Spikes |
";
        std::cout << "+-----------------+------------+-----------+--------------+------------+
";

        for (auto [b, n] : std::vector<std::pair<size_t, size_t>>{{1, 4096}, {1, 65536}, {64, 16384}})
        {
            neurobit::layers::SpikeNeuron::Config cfg{b, 1, n, 1.0f, 0.5f, true};
            neurobit::layers::SpikeNeuron layer(q, cfg);
            std::vector<float> h_X(b * n, 2.0f);
            std::vector<float> h_V(b * n, 0.0f);
            std::vector<float> h_Z(b * n);
            std::vector<float> h_OU(b * n, 0.0f);
            std::vector<int> h_Act(1, 0);

            s::buffer<float> buf_X{h_X.data(), s::range<1>(h_X.size())};
            s::buffer<float> buf_V{h_V.data(), s::range<1>(h_V.size())};
            s::buffer<float> buf_Z{h_Z.data(), s::range<1>(h_Z.size())};
            s::buffer<float> buf_OU{h_OU.data(), s::range<1>(h_OU.size())};
            s::buffer<int> buf_Act{h_Act.data(), s::range<1>(1)};

            for (int i = 0; i < 3; ++i)
                layer.forward_single(q, buf_X, buf_V, buf_Z, buf_OU, buf_Act);
            q.wait();

            // RESET SPIKE COUNT BEFORE BENCHMARK
            {
                s::host_accessor acc(buf_Act, s::write_only);
                acc[0] = 0;
            }

            auto start = std::chrono::high_resolution_clock::now();
            const int iters = 100;
            for (int i = 0; i < iters; ++i)
                layer.forward_single(q, buf_X, buf_V, buf_Z, buf_OU, buf_Act);
            q.wait();
            auto end = std::chrono::high_resolution_clock::now();

            s::host_accessor acc_act(buf_Act, s::read_only);
            int spike_count = acc_act[0];

            double total_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
            double avg_time_ms = total_time_ms / iters;
            // CORRECTED: Total neurons processed / total time
            double total_neurons = static_cast<double>(b) * n * iters;
            double m_neurons_s = total_neurons / (total_time_ms / 1000.0) / 1e6;

            char config[32];
            snprintf(config, sizeof(config), "%zux%zu", b, n);
            printf("| %-15s | %10zu | %9.4f | %12.2f | %10d |
", config, b * n, avg_time_ms, m_neurons_s, spike_count);
        }
        std::cout << "+-----------------+------------+-----------+--------------+------------+
";

        return 0;
    }
    catch (const s::exception &e)
    {
        std::cerr << "
[FATAL] SYCL exception: " << e.what() << "
";
        return 1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "
[FATAL] Exception: " << e.what() << "
";
        return 1;
    }
}
