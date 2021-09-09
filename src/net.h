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

void save_network(Network const&, std::string_view path);
void load_network(Network&      , std::string_view path);