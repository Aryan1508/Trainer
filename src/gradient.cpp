#include "gradient.h"
#include <cmath>

static constexpr float BETA_1 = 0.9f;
static constexpr float BETA_2 = 0.999f;
static constexpr float LR     = 0.01f;

namespace Trainer
{
    float calculate_gradient(Gradient& grad)
    {
        if (!grad.value)
            return 0;
        
        grad.moment_1 = grad.moment_1 * BETA_1 + grad.value * (1 - BETA_1);
        grad.moment_2 = grad.moment_2 * BETA_2 + (grad.value * grad.value) * (1 - BETA_2);

        return LR * grad.moment_1 / (std::sqrt(grad.moment_2) + 1e-8);
    }

    void apply_gradient(Gradient& grad, float& parameter)
    {
        parameter -= calculate_gradient(grad);
        grad.value = 0;
    }
}