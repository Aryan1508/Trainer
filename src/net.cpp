#include "net.h"
#include "cost.h"
#include "sample.h"

#include <thread>
#include <vector>

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
        }
    }

    float Network::feed(Sample const& sample)
    {
        neurons[0].set_zero();

        for (auto index : sample.input.indices)
        {
            for (int i = 0; i < neurons[0].rows(); i++)
                neurons[0](i) += weights[0](i, index);
        }

        for (int i = 0; i < neurons[1].size(); i++)
            neurons[0](i) = relu(neurons[0](i) + biases[0](i));

        for(std::size_t i = 1;i < neurons.size();i++)
        {
            auto const& input   = neurons[i - 1];
            auto const& weight  = weights[i];
            auto const& bias    = biases[i];
            auto& output  = neurons[i];

            for(int i = 0;i < output.size();i++)
            {
                output(i) = 0.0f;

                for(int j = 0;j < input.size();j++)
                {
                    output(i) += input(j) * weight(i, j);
                }
                output(i) = sigmoid(output(i) + bias(i));
            }
        }

        return neurons.back()(0);
    }

    void Network::update_gradients(Sample const& sample)
    {
        float error = calculate_cost_gradient(sample, *this) * sigmoid_prime(get_output());

        for (int i = 0; i < neurons[0].size(); i++)
        {
            if (neurons[0](i) > 0)
            {
                float hidden_error = error * weights[1](i);

                weight_gradients[1](i).update_gradient(neurons[0](i) * error);
                bias_gradients[0](i).update_gradient(hidden_error);
            }
        }

        for(auto activated_input_index : sample.input.indices)
        {
            for(int i = 0;i < neurons[0].size();i++)
            {
                if (neurons[0](i) > 0)
                    weight_gradients[0](i, activated_input_index).update_gradient(error * weights[1](i));
            }
        }

        bias_gradients[1](0).update_gradient(error);
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