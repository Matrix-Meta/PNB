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

#pragma once

#include <sycl/sycl.hpp>
#include <cmath>
#include <vector>

namespace neurobit {
namespace core {

namespace s = sycl;

/**
 * @brief Synaptic Equilibrium Optimizer (SEO)
 * 
 * Implements a biologically-inspired weight update rule that integrates:
 * 1. Hebbian/Gradient Learning (modulated by Vigilance)
 * 2. Structural Attractors (Quantization Force)
 * 3. Synaptic Consolidation (Core Synapse / LTM Protection)
 * 4. Homeostatic Scaling (Activity-dependent)
 */
class SynapticOptimizer {
public:
    struct Config {
        float base_lr = 0.001f;
        float attract_strength = 0.01f;     // Strength of quantization force
        float core_synapse_threshold = 1.5f;// Weight threshold to become a Core Synapse (+2)
        float consolidation_force = 0.1f;   // Force to lock consolidated synapses
        float forgetting_threshold = -0.5f; // Negative gradient needed to break consolidation
        float target_active_rate = 0.15f;   // For homeostatic scaling
        float scaling_factor = 0.0001f;     // Strength of homeostatic scaling
    };

    SynapticOptimizer(const Config& cfg) : cfg_(cfg) {}

    void set_learning_rate(float lr) { cfg_.base_lr = lr; }

    /**
     * @brief Update weights based on gradients and biological plasticity rules.
     * 
     * @param q SYCL queue
     * @param weights Weight buffer
     * @param grads Gradient buffer
     * @param num_params Number of parameters
     * @param vigilance Current vigilance level (0.0 - 1.0)
     * @param active_rate Current layer active rate (for homeostasis)
     * @param enable_core_synapse Whether to allow +2 state consolidation
     * @param profile_events Optional profiling
     */
    void update(s::queue& q, 
                s::buffer<float, 1>& weights, 
                s::buffer<float, 1>& grads, 
                size_t num_params,
                float vigilance,
                float active_rate,
                bool enable_core_synapse,
                std::vector<s::event>* profile_events = nullptr) {
        
        Config cfg = cfg_; // Capture for kernel

        auto ev = q.submit([&](s::handler& h) {
            s::accessor w(weights, h, s::read_write);
            s::accessor g(grads, h, s::read_only);

            h.parallel_for(s::range<1>{num_params}, [=](s::id<1> idx) {
                float weight = w[idx];
                float grad = s::clamp(g[idx], -1.0f, 1.0f); // Clip gradients locally
                
                // 1. Homeostatic Scaling
                // If activity is high, scale down weights slightly (Depression)
                // If activity is low, scale up weights slightly (Potentiation)
                float scaling = 1.0f - cfg.scaling_factor * (active_rate - cfg.target_active_rate);
                
                // 2. Ideal State Calculation (Attractor Target)
                // States: -1, 0, +1, +2
                float ideal = s::round(weight);
                
                // 3. Quantization Force (Attractor)
                float q_err = ideal - weight;
                float force = q_err * cfg.attract_strength;
                
                // 4. Effective Learning Rate Modulation
                float effective_lr = cfg.base_lr * vigilance;

                // 5. Core Synapse Consolidation (+2 State)
                if (enable_core_synapse && ideal == 2.0f) {
                    // Stronger lock to 2.0
                    force = (2.0f - weight) * cfg.consolidation_force;
                    
                    // Freeze mechanism: Only allow strong negative gradients (Forgetting)
                    if (grad > cfg.forgetting_threshold) {
                        effective_lr = 0.0f; 
                    }
                }

                // 6. Update Rule
                float new_w = weight * scaling - effective_lr * grad + force;
                
                // 6.5 Digital Pacemaker (Anti-Collapse)
                // If weight dies (magnitude < 1e-4), shock it back to life.
                // This prevents layer collapse in deep/recurrent networks.
                if (s::fabs(new_w) < 1e-4f) {
                    // Deterministic reset based on neuron index to avoid random generator overhead
                    new_w = (idx[0] % 2 == 0) ? 0.01f : -0.01f;
                }
                
                // 7. Clamp
                // Allow up to 2.2 for Core Synapse, down to -1.2
                // Note: bit_packing treats 11 as +2, 00 as -1.
                w[idx] = s::clamp(new_w, -1.2f, 2.2f);
            });
        });

        if (profile_events) profile_events->push_back(ev);
    }

    /**
     * @brief Update continuous parameters (like SSM A/Delta) using standard SGD.
     * No quantization, no core synapse logic.
     */
    void update_continuous(s::queue& q, 
                           s::buffer<float, 1>& params, 
                           s::buffer<float, 1>& grads, 
                           size_t num_params,
                           float lr_scaler = 1.0f,
                           float min_val = -1000.0f,
                           float max_val = 1000.0f,
                           std::vector<s::event>* profile_events = nullptr) {
        
        Config cfg = cfg_;
        auto ev = q.submit([&](s::handler& h) {
            s::accessor w(params, h, s::read_write);
            s::accessor g(grads, h, s::read_only);

            h.parallel_for(s::range<1>{num_params}, [=](s::id<1> idx) {
                float grad = s::clamp(g[idx], -1.0f, 1.0f);
                float effective_lr = cfg.base_lr * lr_scaler;
                
                // Simple SGD
                float new_w = w[idx] - effective_lr * grad;
                
                w[idx] = s::clamp(new_w, min_val, max_val);
            });
        });
        if (profile_events) profile_events->push_back(ev);
    }

private:
    Config cfg_;
};

} // namespace core
} // namespace neurobit
