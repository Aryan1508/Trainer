#pragma once 
#include <cmath>

inline float LEARNING_RATE = 0.01f;

struct Gradient 
{
    float value = 0; 
    float moment_1 = 0;
    float moment_2 = 0;
};

inline void update_gradient(Gradient& grad, const float delta)
{   
    grad.value += delta;
}

inline float calculate_gradient(Gradient& grad)
{
    constexpr float BETA_1 = 0.9f;
    constexpr float BETA_2 = 0.999f;

    if (!grad.value)
        return 0;
    
    grad.moment_1 = grad.moment_1 * BETA_1 + grad.value * (1.0f - BETA_1);
    grad.moment_2 = grad.moment_2 * BETA_2 + (grad.value * grad.value) * (1.0f - BETA_2);

    return LEARNING_RATE * grad.moment_1 / (std::sqrt(grad.moment_2) + 1e-8);
}

inline void apply_gradient(Gradient& grad, float& parameter)
{
    parameter -= calculate_gradient(grad);
    grad.value = 0.0f;
}

inline void reset_gradient(Gradient& grad)
{
    grad.value = 0;
}