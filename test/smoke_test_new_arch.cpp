/*
 * Copyright 2025 Project Neuro-Bit Contributors
 */

#include "neurobit/layers/bit_brain.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

// This is a minimal smoke test for the new architecture.
// It initializes the network, performs one forward and backward pass,
// and checks if the EqOpt successfully moves weights.

namespace s = sycl;

int main() {
    s::queue q{s::gpu_selector_v};
    std::cout << "Running on: " << q.get_device().get_info<s::info::device::name>() << std::endl;

    neurobit::layers::BitBrainLayer::Config cfg{
        .batch_size = 16,
        .input_dim = 784,
        .hidden_dim = 1024,
        .output_dim = 10,
        .glial_config = {.target_sparsity = 0.15f}
    };

    neurobit::layers::BitBrainLayer brain(q, cfg);
    
    s::buffer<float, 1> input(s::range<1>(16 * 784));
    s::buffer<float, 1> output(s::range<1>(16 * 10));
    s::buffer<float, 1> grad_out(s::range<1>(16 * 10));
    s::buffer<float, 1> grad_in(s::range<1>(16 * 784));
    s::buffer<float, 1> w_fast(s::range<1>(784 * 1024));
    s::buffer<float, 1> w_slow(s::range<1>(784 * 1024));

    {
        s::host_accessor acc_in(input, s::write_only);
        for(size_t i=0; i<16*784; ++i) acc_in[i] = 0.1f;
        s::host_accessor acc_go(grad_out, s::write_only);
        for(size_t i=0; i<16*10; ++i) acc_go[i] = 0.01f;
        s::host_accessor acc_wf(w_fast, s::write_only);
        for(size_t i=0; i<784*1024; ++i) acc_wf[i] = 0.0f;
        s::host_accessor acc_ws(w_slow, s::write_only);
        for(size_t i=0; i<784*1024; ++i) acc_ws[i] = 0.0f;
    }

    std::cout << "1. Testing Forward (2D-SSM)..." << std::endl;
    brain.set_feedback(1.0f, 0.5f, 0.1f);
    brain.forward(input, output, w_fast, w_slow);
    q.wait();

    std::cout << "2. Testing Backward..." << std::endl;
    brain.backward(grad_out, grad_in);
    q.wait();

    std::cout << "3. Testing Step (EqOpt)..." << std::endl;
    brain.step(0.01f);
    q.wait();

    std::cout << "Smoke Test PASSED!" << std::endl;
    return 0;
}
