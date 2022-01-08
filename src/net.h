#pragma once
#include "matrix.h"
#include "position.h"

#include <vector>

struct Neurons;
struct Gradients;

struct Network
{
    Network(std::vector<int> const& topology);

    std::vector<Matrix<float>> weights;
    std::vector<Matrix<float>> biases;
};

float forward_propagate(Features const& input, Network const&, Neurons& neurons);

void save_network(Network const&, Gradients const&, std::string_view path); 
void load_network(Network&      , Gradients&      , std::string_view path);
void save_network(Network const&, std::string_view path);
void load_network(Network&      , std::string_view path);

std::vector<int> get_topology(Network const&);