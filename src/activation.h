#pragma once
#include <math.h>
#include <algorithm>

typedef float(*ActivationFunction)(float);

constexpr float sigmoid_scale = 4.0f / 1024.0f;

inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(sigmoid_scale * -x));
}

inline float sigmoid_prime(float x)
{
    return x * (1.0f - x) * sigmoid_scale;
}

inline float relu(float x)
{
    return std::max<float>(x, 0.0f);
}

inline float relu_prime(float x)
{
    return x > 0.0f;
}