#include "net.h"

template<typename T1, typename T2, typename T3, typename Callable>
static void forward_propagate(
	T1 const& weights,
	T2 const& neurons,
	T3 const& biases,
	T3& result_neurons, Callable activation)
{
	assert(weights.total_cols() == neurons.total_rows());

	for (int i = 0; i < weights.total_rows(); i++)
	{
		float sum = 0;
		for (int k = 0; k < weights.total_cols(); k++)
			sum += weights.get(i, k) * neurons.get(k);

		result_neurons.get(i) = activation(sum + biases.get(i));
	}
}

namespace Trainer
{
	Network::Network()
	{
		hidden_weights.randomize(InputVector::size());
		hidden_biases .randomize(InputVector::size());
		output_weights.randomize(HiddenVector::size());
		output_bias   .randomize(HiddenVector::size());

		hidden_neurons      .set(0);
		hidden_errors       .set(0);
		output_neuron       .set(0);
		output_error        .set(0);
		output_bias_deltas  .set(0);
		hidden_bias_deltas  .set(0);
		output_weight_deltas.set(0);
		hidden_weight_deltas.set(0);
	}

	void Network::feed(std::vector<int> const& input_indices)
	{
		hidden_neurons.set(0);

		for (auto index : input_indices)
		{
			for (int i = 0; i < hidden_weights.total_rows(); i++)
				hidden_neurons.get(i) += hidden_weights.get(i, index);
		}

		for (int i = 0; i < hidden_neurons.size(); i++)
			hidden_neurons.get(i) = relu(hidden_neurons.get(i) + hidden_biases.get(i));

		forward_propagate(output_weights, hidden_neurons, output_bias, output_neuron, sigmoid);
	}

	void Network::calculate_gradients(InputVector const& sample)
	{
		for (int i = 0; i < hidden_neurons.total_rows(); i++)
		{
			for (int j = 0; j < sample.total_rows(); j++)
			{
				hidden_weight_deltas.get(i, j) += sample.get(j) * hidden_errors.get(i);
			}

			hidden_bias_deltas.get(i)   += hidden_errors.get(i);
			output_weight_deltas.get(i) += hidden_neurons.get(i) * output_error.get(0);
		}
		output_bias_deltas.get(0) += output_error.get(0);
	}

	void Network::apply_gradients()
	{
		float rate = 0.01f;
		for (int i = 0; i < hidden_neurons.total_rows(); i++)
		{
			for (int j = 0; j < InputVector::size(); j++)
			{
				hidden_weights.get(i, j) -= rate * hidden_weight_deltas.get(i, j);
				hidden_weight_deltas.get(i, j) = 0;
			}

			hidden_biases.get(i) -= rate * hidden_bias_deltas.get(i);
			hidden_bias_deltas.get(i) = 0;

			output_weights.get(i) -= rate * output_weight_deltas.get(i);
			output_weight_deltas.get(i) = 0;
		}
		output_bias.get(0) -= rate * output_bias_deltas.get(0);
		output_bias_deltas.get(0) = 0;
	}

	void Network::calculate_errors(float target)
	{
		output_error.get(0) = (output_neuron.get(0) - target) * sigmoid_prime(output_neuron.get(0)) * 2;

		for (int i = 0; i < hidden_errors.size(); i++)
			hidden_errors.get(i) = hidden_neurons.get(i) > 0 ? output_error.get(0) * output_weights.get(i) : 0;
	}
}