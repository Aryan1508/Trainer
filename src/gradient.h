#pragma once 

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

float calculate_gradient(Gradient&);

void apply_gradient(Gradient&, float& parameter);