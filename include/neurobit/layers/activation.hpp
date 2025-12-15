#pragma once

#include <cmath>
#include <sycl/sycl.hpp>

namespace neurobit
{
namespace layers
{

namespace s = sycl;

/**
 * M-DSiLU (Memory-modulated Dynamic SiLU)
 *
 * Formula: y = SiLU(x - (theta_base + vigilance * (1 - familiarity)))
 *
 * - theta_base: Provided by Glial Cell (Homeostasis)
 * - familiarity: Provided by Hippocampus (0.0 = Novel, 1.0 = Familiar)
 * - vigilance: Hyperparameter controlling how much novelty suppresses activity
 *
 * Characteristics:
 * - Smooth, non-monotonic (like SiLU)
 * - Dynamic sparsity (controlled by theta)
 * - Context-aware (controlled by familiarity)
 */
class MDSiLU
{
  public:
    float vigilance;

    MDSiLU(float vigilance_factor = 0.5f) : vigilance(vigilance_factor)
    {
    }

    void forward(s::queue &q, s::buffer<float, 1> &input, s::buffer<float, 1> &output, size_t size,
                 float base_threshold, float familiarity)
    {
        float v = vigilance;
        q.submit([&](s::handler &h) {
            s::accessor in(input, h, s::read_only);
            s::accessor out(output, h, s::write_only, s::no_init);

            h.parallel_for(s::range<1>(size), [=](s::id<1> idx) {
                // Effective Threshold = Base + Vigilance * (1 - F)
                float eff_th = base_threshold + v * (1.0f - familiarity);

                float x_shifted = in[idx] - eff_th;
                float sigmoid = 1.0f / (1.0f + s::exp(-x_shifted));

                out[idx] = x_shifted * sigmoid;
            });
        });
    }

    void backward(s::queue &q, s::buffer<float, 1> &grad_output, s::buffer<float, 1> &input,
                  s::buffer<float, 1> &grad_input, size_t size, float base_threshold, float familiarity)
    {
        float v = vigilance;
        q.submit([&](s::handler &h) {
            s::accessor dy(grad_output, h, s::read_only);
            s::accessor x(input, h, s::read_only);
            s::accessor dx(grad_input, h, s::write_only, s::no_init);

            h.parallel_for(s::range<1>(size), [=](s::id<1> idx) {
                float eff_th = base_threshold + v * (1.0f - familiarity);
                float x_shifted = x[idx] - eff_th;

                float exp_val = s::exp(-x_shifted);
                float sigmoid = 1.0f / (1.0f + exp_val);

                // SiLU derivative: sigma(x) + x * sigma(x) * (1 - sigma(x))
                //                = sigma(x) * (1 + x * (1 - sigma(x)))
                float d_silu = sigmoid * (1.0f + x_shifted * (1.0f - sigmoid));

                dx[idx] = dy[idx] * d_silu;
            });
        });
    }
};

} // namespace layers
} // namespace neurobit
