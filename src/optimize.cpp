#include "net.h"
#include "cost.h"
#include "sample.h"
#include "optimize.h"
#include "activation.h"

namespace Trainer
{
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

    void calculate_gradients(Sample const& sample, Network& network, Gradients& gradients)
    {
        const float output_gradient = calculate_output_gradient(sample, network);

        for (int i = 0; i < network.neurons[0].size(); i++)
        {
            if (network.neurons[0](i) > 0)
            {
                float hidden_error = output_gradient * network.weights[1](i);

                update_gradient(gradients.weight_gradients[1](i), network.neurons[0](i) * output_gradient);
                update_gradient(gradients.bias_gradients[0](i), hidden_error);
            }
        }

        for(auto activated_input_index : sample.input.indices)
        {
            for(int i = 0;i < network.neurons[0].size();i++)
            {
                if (network.neurons[0](i) > 0)
                    update_gradient(gradients.weight_gradients[0](i, activated_input_index),
                                    output_gradient * network.weights[1](i));
            }
        }

        update_gradient(gradients.bias_gradients[1](0), output_gradient);
    }
}