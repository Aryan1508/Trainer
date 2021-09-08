#pragma once
#include "matrix.h"

#include <vector>

struct Input;

struct Network
{
    Network(std::vector<int> const& topology);

    std::vector<Matrix<float>> neurons;
    std::vector<Matrix<float>> weights;
    std::vector<Matrix<float>> biases;
};

float forward_propagate(Input const& input, Network&);

std::vector<int> get_topology(Network const&);