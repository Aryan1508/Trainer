#pragma once 
#include "matrix.h"
#include "gradient.h"
#include <vector>
#include <utility>

struct Sample;
class Network;

struct Gradients 
{
    Gradients(std::vector<int> const& topology);

    std::vector<Matrix<Gradient>> weight_gradients;
    std::vector<Matrix<Gradient>>   bias_gradients;
};

void apply_gradients(Network&, Gradients&);

void reset_gradients(Gradients&);

void calculate_gradients(Sample const&, Network&, Gradients&);

float calculate_output_gradient(Sample const&, Network&);

float calculate_output_gradient(Sample const&, const float output);

float calculate_cost_gradient(Sample const&, const float output);