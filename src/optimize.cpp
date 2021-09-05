#include "net.h"
#include "cost.h"
#include "sample.h"
#include "optimize.h"
#include "activation.h"

Gradients::Gradients(std::vector<int> const& topology)
{
    for(std::size_t i = 1;i < topology.size();i++)
    {
        const int size          = topology[i];
        const int previous_size = topology[i - 1];

        weight_gradients.push_back(Matrix<Gradient>(size, previous_size));
        bias_gradients  .push_back(Matrix<Gradient>(size, 1)); 
    }
}

float calculate_output_gradient(Sample const& sample, Network& network)
{
    const float output = forward_propagate(sample.input, network);
    return calculate_output_gradient(sample, output);
}

float calculate_output_gradient(Sample const& sample, const float output)
{
    return calculate_cost_gradient(sample, output) * sigmoid_prime(output);
}

void apply_gradients(Matrix<float>& values, 
                        Matrix<Gradient>& gradients)
{
    for(int i = 0;i < values.size();i++)
        apply_gradient(gradients(i), values(i));
}

void apply_gradients(Network& network, Gradients& gradients)
{
    for(std::size_t layer = 0;layer < network.neurons.size();layer++)
    {
        apply_gradients(network.weights[layer], gradients.weight_gradients[layer]);
        apply_gradients(network.biases[layer], gradients.bias_gradients[layer]);
    }
}

void calculate_errors(Sample const& sample, Network& network, std::vector<Matrix<float>>& errors)
{
    const float output_gradient = calculate_output_gradient(sample, network);

    errors.back()(0) = output_gradient;

    for(std::size_t layer = network.neurons.size() - 1;layer --> 0;)
    {
        auto const& neurons = network.neurons[layer];
        auto const& weights = network.weights[layer + 1];

        auto const& output_errors = errors[layer + 1];
        auto      & layer_errors  = errors[layer];

        for(int i = 0;i < errors[layer].size();i++)
        {
            layer_errors(i) = 0.0f;

            for(int j = 0;j < output_errors.size();j++)
                layer_errors(i) += output_errors(j) * weights(j, i) * relu_prime(neurons(i));
        }
    }        
}

void calculate_hidden_gradients(Sample const& sample, Gradients& gradients, std::vector<Matrix<float>>& errors)
{
    auto& weight_gradients = gradients.weight_gradients[0];
    auto& bias_gradients   = gradients.bias_gradients[0];
    auto& hidden_errors    = errors[0];

    for(auto input_layer_index : sample.input.indices)
    {
        for(int i = 0;i < hidden_errors.size();i++)
            update_gradient(weight_gradients(i, input_layer_index), hidden_errors(i));
    }

    for(int i = 0;i < hidden_errors.size();i++)
        update_gradient(bias_gradients(i), hidden_errors(i));
}

void calculate_gradients(Sample const& sample, Network& network, Gradients& gradients)
{
    auto errors = network.neurons;
    
    calculate_errors(sample, network, errors);
    calculate_hidden_gradients(sample, gradients, errors);

    for(std::size_t layer = 1;layer < network.neurons.size();layer++)
    {
        auto& weight_grads = gradients.weight_gradients[layer];
        auto& bias_grads   = gradients.bias_gradients[layer];
        auto& neurons      = network.neurons[layer - 1];

        for(int i = 0;i < bias_grads.size();i++)
        {
            const float gradient = errors[layer](i);
            update_gradient(bias_grads(i), gradient);

            for(int j = 0;j < weight_grads.rows();j++)
            {
                const float gradient = neurons(i) * errors[layer](j);
                update_gradient(weight_grads(j, i), gradient);
            }
        }
    }
}