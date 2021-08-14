#include "net.h"

namespace Trainer
{
	Network::Network()
	{
		hidden_weights.randomize(InputVector::size());
		hidden_biases .randomize(InputVector::size());
		output_weights.randomize(HiddenVector::size());
		output_bias   .randomize(HiddenVector::size());

		hidden_neurons      .set(Parameter());
		output_neuron       .set(Parameter());
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

	void Network::feed(std::vector<int> const& input_indices)
	{
		for (int i = 0; i < hidden_neurons.size(); i++) hidden_neurons.get(i).value = 0;

		for (auto index : input_indices)
		{
			for (int i = 0; i < hidden_weights.total_rows(); i++)
				hidden_neurons.get(i).value += hidden_weights.get(i, index).value;
		}

		for (int i = 0; i < hidden_neurons.size(); i++)
			hidden_neurons.get(i).value = relu(hidden_neurons.get(i).value + hidden_biases.get(i).value);

		for (int i = 0; i < output_weights.total_rows(); i++)
		{
			float sum = 0;

			for (int k = 0; k < output_weights.total_cols(); k++)
				sum += output_weights.get(i, k).value * hidden_neurons.get(k).value;

			output_neuron.get(i).value = sigmoid(sum + output_bias.get(i).value);
		}
	}

	void Network::update_gradients(InputVector const& sample, float target)
	{
		float error = (get_output() - target) * sigmoid_prime(get_output()) * 2;

		for (int i = 0; i < hidden_neurons.size(); i++)
		{
			if (hidden_neurons.get(i).value > 0)
			{
				float hidden_error = error * output_weights.get(i).value;

				for (int j = 0; j < sample.total_rows(); j++)
				{
					hidden_weights.get(i, j).gradient += hidden_error * sample.get(j).value;
				}
				output_weights.get(i).gradient += hidden_neurons.get(i).value * error;
				hidden_biases.get(i).gradient += hidden_error;
			}
		}
		output_bias.get(0).gradient += error;
	}

	template<typename T>
	void apply(T& mat)
	{
		for (int i = 0; i < mat.size(); i++)
			mat.get(i).apply_gradient();
	}

	void Network::apply_gradients()
	{
		apply(hidden_weights);
		apply(hidden_biases);
		apply(output_weights);
		apply(output_bias);
	}
}