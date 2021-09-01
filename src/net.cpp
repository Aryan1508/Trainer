#include "net.h"
#include "cost.h"
#include "sample.h"
#include "matmul.h"
#include "gradients.h"

#include <vector>
#include <random>

static void he_initialize_matrix(Trainer::Matrix<float>& mat, const int n_input_neurons)
{
    const float g = 2.0f / sqrtf(static_cast<float>(n_input_neurons));

    std::random_device rd;
    std::normal_distribution<float> distrib(0.0f, g);
    std::mt19937 rng(rd());

    for(int i = 0;i < mat.size();i++)
        mat(i) = distrib(rng);
}

namespace Trainer
{
    Network::Network(std::vector<int> const& topology)
    {
        for(std::size_t i = 1;i < topology.size();i++)
        {
            const int output_size = topology[i];
            const int input_size  = topology[i - 1];

            neurons.push_back(Matrix<float>(output_size));
            biases.push_back(Matrix<float>(output_size));

            weights.push_back(Matrix<float>(output_size, input_size));
            
            weight_gradients.push_back(Matrix<Parameter>(output_size, input_size));
            bias_gradients.push_back(Matrix<Parameter>(output_size));

            he_initialize_matrix(weights.back(), input_size);
            he_initialize_matrix(biases.back() , input_size);
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

    void Network::update_gradients(Sample const& sample)
    {
        const float output_gradient = calculate_output_gradient(sample, *this);

        for (int i = 0; i < neurons[0].size(); i++)
        {
            if (neurons[0](i) > 0)
            {
                float hidden_error = output_gradient * weights[1](i);

                weight_gradients[1](i).update_gradient(neurons[0](i) * output_gradient);
                bias_gradients[0](i).update_gradient(hidden_error);
            }
        }

        for(auto activated_input_index : sample.input.indices)
        {
            for(int i = 0;i < neurons[0].size();i++)
            {
                if (neurons[0](i) > 0)
                    weight_gradients[0](i, activated_input_index).update_gradient(output_gradient * weights[1](i));
            }
        }

        bias_gradients[1](0).update_gradient(output_gradient);
    }

    template<typename T1, typename T2>
    void apply_gradients(T1& values, T2& gradients)
    {
        for (int i = 0; i < values.size(); i++)
            values(i) += gradients(i).get_final_gradient();
    }

    void Network::apply_gradients()
    {
        Trainer::apply_gradients(weights[0], weight_gradients[0]);
        Trainer::apply_gradients(biases[0], bias_gradients[0]);
        Trainer::apply_gradients(weights[1], weight_gradients[1]);
        Trainer::apply_gradients(biases[1], bias_gradients[1]);
    }
}