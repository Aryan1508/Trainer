#include "net.h"
#include "cost.h"
#include "sample.h"
#include "matmul.h"
#include "neurons.h"
#include "optimize.h"

#include <vector>
#include <random>

void he_initialize_matrix(Matrix<float>& mat, const int n_input_neurons)
{
    const float g = 2.0f / sqrtf(static_cast<float>(n_input_neurons));

    std::random_device rd;
    std::normal_distribution<float> distrib(0.0f, g);
    std::mt19937 rng(rd());

    for(int i = 0;i < mat.size();i++)
        mat(i) = distrib(rng);
}

Network::Network(std::vector<int> const& topology)
{
    for(std::size_t i = 1;i < topology.size();i++)
    {
        const int output_size = topology[i];
        const int input_size  = topology[i - 1];

        biases .push_back(Matrix<float>(output_size));
        weights.push_back(Matrix<float>(output_size, input_size));
        
        he_initialize_matrix(weights.back(), input_size);
        he_initialize_matrix(biases.back() , input_size);
    }
}

float forward_propagate(Input const& input, Network const& network, Neurons& neurons)
{
    forward_propagate(input,
                      neurons.activations[0],
                      network.weights[0],
                      network.biases[0],
                      relu);

    for(std::size_t i = 1;i < neurons.activations.size();i++)
    {
        const auto activation = i == neurons.activations.size() - 1 ? sigmoid 
                                                                    : relu;
        forward_propagate(neurons.activations[i - 1],
                          neurons.activations[i],
                          network.weights[i],
                          network.biases[i],
                          activation);
    }

    return neurons.activations.back()(0);
}

std::vector<int> get_topology(Network const& network)
{
    std::vector<int> topology;

    topology.push_back(network.weights[0].cols()); // Input layer

    for(auto const& layer : network.biases)
        topology.push_back(layer.size());

    return topology;
}

std::uint32_t hash_network(Network const& network)
{
    float sum = 0.0f;

    for(auto& weights : network.weights)
        for(int i = 0;i < weights.size();i++)
            sum += weights(i);

    return std::hash<float>{}(sum);
}

void save_network(Network const& network, FILE* file) 
{
    const std::uint32_t hash     = hash_network(network);
    const std::uint32_t n_layers = network.biases.size() + 1;

    fwrite(&hash, sizeof(std::uint32_t), 1, file);  
    for(std::size_t i = 0;i < n_layers - 1;i++)
    {
        fwrite(network.weights[i].raw(), sizeof(float), network.weights[i].size(), file);
        fwrite(network.biases[i].raw(), sizeof(float), network.biases[i].size(), file);
    }
}

void save_network(Network const& network, std::string_view path)
{
    const char* p = path.data();
    FILE* file = fopen(p, "wb");
    save_network(network, file);
    fclose(file);
}

// write gradients for re-loading 
void save_network(Network const& network, Gradients const& gradients, std::string_view path) 
{
    const char* p = path.data();
    FILE* file = fopen(p, "wb");
    save_network(network, file);

    const std::uint32_t n_layers = network.biases.size() + 1;
    for(std::size_t i = 0;i < n_layers - 1;i++) 
    {
        fwrite(gradients.weight_gradients[i].raw(), sizeof(Gradient), gradients.weight_gradients[i].size(), file);
        fwrite(gradients.bias_gradients[i].raw(), sizeof(Gradient), gradients.bias_gradients[i].size(), file);
    }
    fclose(file);
}

std::uint32_t load_network(Network& network, FILE* file) 
{
    std::uint32_t hash = 0;
    fread(&hash, sizeof(hash), 1, file);
    for(std::size_t i = 0;i < network.biases.size();i++)
    {
        fread(network.weights[i].raw(), sizeof(float), network.weights[i].size(), file);
        fread(network.biases[i].raw(), sizeof(float), network.biases[i].size(), file);
    }
    return hash;
}

void load_network(Network& network, Gradients& gradients, std::string_view path) 
{
    const char* p = path.data();
    FILE* file = fopen(p, "rb");
    const std::uint32_t hash  = load_network(network, file);

    for(std::size_t i = 0;i < network.biases.size();i++)
    {
        fread(gradients.weight_gradients[i].raw(), sizeof(Gradient), gradients.weight_gradients[i].size(), file);
        fread(gradients.bias_gradients[i].raw(), sizeof(Gradient), gradients.bias_gradients[i].size(), file);
    }
        
    fclose(file);
    std::cout << std::hex << "Loaded network '" << path << "' / " << hash << std::dec << '\n';
}

void load_network(Network& network, std::string_view path)
{
    const char* p = path.data();
    FILE* file = fopen(p, "rb");
    const std::uint32_t hash  = load_network(network, file);
    fclose(file);
    std::cout << std::hex << "Loaded network '" << path << "' / " << hash << std::dec << '\n';
}