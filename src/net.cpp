#include "net.h"
#include <thread>
#include <vector>

namespace 
{
    template<int input_neuron_count, typename T> 
    void randomize_matrix(T& matrix)
    {
        float g = 2 / sqrtf(static_cast<float>(input_neuron_count));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> distrib(0.0f, g);

        for(int i = 0;i < matrix.size();i++)
            matrix.get(i) = distrib(gen);
    }
}

namespace Trainer
{
    Network::Network()
    {
        randomize_matrix<INPUT_SIZE>(hidden_weights);
        randomize_matrix<INPUT_SIZE>(hidden_biases);
        randomize_matrix<HIDDEN_SIZE>(output_weights);
        randomize_matrix<HIDDEN_SIZE>(output_bias);

        hidden_neurons.set(0.0f);
        output_neuron .set(0.0f);
    }

    void Network::save_network(std::string_view path)
    {
        const char* p = path.data();
        FILE* f = fopen(p, "wb");

        uint64_t count = 0;

        count += hidden_weights.size();
        count += output_weights.size();
        count += hidden_biases.size();
        count += output_bias.size();

        fwrite(&count, sizeof(uint64_t), 1, f);

        fwrite(hidden_weights.raw(), sizeof(Parameter), hidden_weights.size(), f);
        fwrite(output_weights.raw(), sizeof(Parameter), output_weights.size(), f);

        fwrite(hidden_biases.raw(), sizeof(Parameter), hidden_biases.size(), f);
        fwrite(output_bias.raw(), sizeof(Parameter), output_bias.size(), f);

        fclose(f);
    }

    void Network::load_network(std::string_view path)
    {
        const char* p = path.data();
        FILE* f = fopen(p, "rb");

        uint64_t count = 0;

        count += hidden_weights.size();
        count += output_weights.size();
        count += hidden_biases.size();
        count += output_bias.size();

        uint64_t fileCount = 0;
        fread(&fileCount, sizeof(uint64_t), 1, f);

        if (count != fileCount)
        {
            std::cerr << "Error loading network" << std::endl;
            std::terminate();
        }

        fread(hidden_weights.raw(), sizeof(Parameter), hidden_weights.size(), f);
        fread(output_weights.raw(), sizeof(Parameter), output_weights.size(), f);

        fread(hidden_biases.raw(), sizeof(Parameter), hidden_biases.size(), f);
        fread(output_bias.raw(), sizeof(Parameter), output_bias.size(), f);

        fclose(f);
    }

    void Network::feed(NetworkInput const& sample)
    {
        hidden_neurons.set(0.0f);

        for (auto index : sample.activated_input_indices)
        {
            for (int i = 0; i < hidden_weights.total_rows(); i++)
                hidden_neurons.get(i) += hidden_weights.get(i, index);
        }

        for (int i = 0; i < hidden_neurons.size(); i++)
            hidden_neurons.get(i) = relu(hidden_neurons.get(i) + hidden_biases.get(i));

        for (int k = 0; k < output_weights.total_cols(); k++)
            output_neuron.get(0) +=  hidden_neurons.get(k) * output_weights.get(k);

        output_neuron.get(0) = sigmoid(output_neuron.get(0) + output_bias.get(0));
    }

    void Network::update_gradients(NetworkInput const& sample)
    {
        float error = (get_output() - sample.target) * sigmoid_prime(get_output()) * 2;

        for (int i = 0; i < hidden_neurons.size(); i++)
        {
            if (hidden_neurons.get(i) > 0)
            {
                float hidden_error = error * output_weights.get(i);
                output_weights.get(i).update_gradient(hidden_neurons.get(i) * error);
                hidden_biases.get(i).update_gradient(hidden_error);
            }
        }

        for(auto activated_input_index : sample.activated_input_indices)
        {
            for(int i = 0;i < hidden_neurons.size();i++)
            {
                if (hidden_neurons.get(i) > 0)
                    hidden_weights.get(i, activated_input_index).update_gradient(error * output_weights.get(i));
            }
        }

        output_bias.get(0).update_gradient(error);
    }

    template<typename T>
    void apply_gradients(T& matrix)
    {
        for (int i = 0; i < matrix.size(); i++)
            matrix.get(i).apply_gradient();
    }

    void Network::apply_gradients()
    {
        Trainer::apply_gradients(hidden_weights);
        Trainer::apply_gradients(hidden_biases);
        Trainer::apply_gradients(output_weights);
        output_bias.get(0).apply_gradient();
    }
}