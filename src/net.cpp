#include "net.h"
#include "cost.h"
#include "sample.h"
#include "matmul.h"

#include <vector>
#include <random>

void he_initialize_matrix(Matrix<float>& mat, const int n_input_neurons)
{
    const float g = 2.0f / sqrtf(static_cast<float>(n_input_neurons));

    std::random_device rd;
    std::normal_distribution<float> distrib(0.0f, g);
    std::mt19937 rng(1234);

    for(int i = 0;i < mat.size();i++)
        mat(i) = distrib(rng);
}

Network::Network(std::vector<int> const& topology)
{
    for(std::size_t i = 1;i < topology.size();i++)
    {
        const int output_size = topology[i];
        const int input_size  = topology[i - 1];

        neurons.push_back(Matrix<float>(output_size));
        biases .push_back(Matrix<float>(output_size));
        weights.push_back(Matrix<float>(output_size, input_size));
        
        he_initialize_matrix(weights.back(), input_size);
        he_initialize_matrix(biases.back() , input_size);
        neurons.back().set(0.0f);
    }
}

float forward_propagate(Input const& input, Network& network)
{
    forward_propagate(input,
                      network.neurons[0],
                      network.weights[0],
                      network.biases[0],
                      relu);

    for(std::size_t i = 1;i < network.neurons.size();i++)
    {
        const auto activation = i == network.neurons.size() - 1 ? sigmoid 
                                                                : relu;
        forward_propagate(network.neurons[i - 1],
                          network.neurons[i],
                          network.weights[i],
                          network.biases[i],
                          activation);
    }

    return network.neurons.back()(0);
}