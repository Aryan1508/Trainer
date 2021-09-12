#include "net.h"
#include "cost.h"
#include "sample.h"
#include "matmul.h"
#include "neurons.h"

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

constexpr std::uint32_t network_key = 1113194752;

void save_network(Network const& network, std::string_view path)
{
    const char* p = path.data();
    FILE* file = fopen(p, "wb");

    const auto topology = get_topology(network);
    const std::uint32_t hash     = hash_network(network);
    const std::uint32_t n_layers = static_cast<std::uint32_t>(topology.size());

    fwrite(&network_key  , sizeof(std::uint32_t), 1, file);
    fwrite(&hash         , sizeof(std::uint32_t), 1, file);  
    fwrite(&n_layers     , sizeof(std::uint32_t), 1, file);

    for(const std::uint32_t layer_size : topology)
        fwrite(&layer_size, sizeof(std::uint32_t), 1, file);

    for(std::size_t i = 0;i < n_layers - 1;i++)
    {
        fwrite(network.weights[i].raw(), sizeof(float), network.weights[i].size(), file);
        fwrite(network.biases[i].raw(), sizeof(float), network.biases[i].size(), file);
    }

    fclose(file);
}

void load_network(Network& network, std::string_view path)
{
    const char* p = path.data();
    FILE* file = fopen(p, "rb");

    std::uint32_t key = 0, n_layers = 0;
    fread(&key, sizeof(key), 1, file);

    if (key != network_key)
        throw std::runtime_error("Inavlid network file");
    
    fread(&key, sizeof(key), 1, file);
    fread(&n_layers, sizeof(n_layers), 1, file);

    std::vector<std::uint32_t> topology;
    topology.resize(n_layers);

    for(auto& layer_size : topology)
        fread(&layer_size, sizeof(layer_size), 1, file);

    for(std::size_t i = 0;i < n_layers - 1;i++)
    {
        fread(network.weights[i].raw(), sizeof(float), network.weights[i].size(), file);
        fread(network.biases[i].raw(), sizeof(float), network.biases[i].size(), file);
    }

    std::cout << std::hex << "Loaded network '" << path << "' / " << key << std::dec << '\n';
    fclose(file);
}