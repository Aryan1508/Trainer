#pragma once
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include "matrix.h"
#include <math.h>

namespace Trainer
{
	float sigmoid(float x)
	{
		return 1.0f / (1.0f + expf(-x * (2.5f / 400.0f)));
	}

	float sigmoid_prime(float x)
	{
		return x * (1.0f - x) * (2.5f / 400.0f);
	}

	float relu(float x)
	{
		return std::max<float>(x, 0.0f);
	}

	float reluD(float x)
	{
		return x > 0.0f ? 1.0f : 0.0f;
	}

	constexpr int N_INPUT_NEURONS = 768;
	constexpr int N_HIDDEN_NEURONS = 512;


	template<typename T1, typename T2, typename T3, typename Callable>
	inline void forward_propagate(
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

	inline void forward_propagate(
		Trainer::Matrix<N_HIDDEN_NEURONS, N_INPUT_NEURONS, true> const& weights,
		std::vector<int>        const& indices,
		Trainer::Matrix<N_HIDDEN_NEURONS, 1> const& biases,
		Trainer::Matrix<N_HIDDEN_NEURONS, 1>& result_neurons)
	{
		result_neurons.set(0);

		for (auto index : indices)
		{
			for (int i = 0; i < weights.total_rows(); i++)
			{
				result_neurons.get(i) += weights.get(i, index);
			}
		}

		for (int i = 0; i < weights.total_rows(); i++)
		{
			result_neurons.get(i) = relu(result_neurons.get(i) + biases.get(i));
		}
	}
	
	class Network
	{
	public:
		Matrix<N_HIDDEN_NEURONS, 1> hidden_neurons;
		Matrix<  1             , 1> output_neuron;
		
		Matrix<N_HIDDEN_NEURONS, N_INPUT_NEURONS, true>  hidden_weights;
		Matrix<N_HIDDEN_NEURONS, N_INPUT_NEURONS> hidden_weight_deltas;
		Matrix<  1            , N_HIDDEN_NEURONS> output_weights, output_weight_deltas;

		Matrix<N_HIDDEN_NEURONS, 1> hidden_biases, hidden_bias_deltas;
		Matrix<  1             , 1> output_bias, output_bias_deltas;

		Matrix<N_HIDDEN_NEURONS, 1> hidden_error;
		float output_error;

		Network()
		{
			hidden_weights.randomize(N_INPUT_NEURONS);
			hidden_weight_deltas.set(0);

			output_weights.randomize(N_HIDDEN_NEURONS);
			output_weight_deltas.set(0);

			hidden_biases.randomize(N_INPUT_NEURONS);
			hidden_bias_deltas.set(0);

			output_bias.randomize(N_HIDDEN_NEURONS);
			output_bias_deltas.set(0);

			hidden_neurons.set(0);
			hidden_error.set(0);
			output_error = 0;
			output_neuron.set(0);
		}

		float feed(std::vector<int> const& input_indices)
		{
			forward_propagate(hidden_weights, input_indices, hidden_biases, hidden_neurons);
			forward_propagate(output_weights, hidden_neurons, output_bias, output_neuron, sigmoid);
			return output_neuron.get(0);
		}

		void calculate_errors(float target)
		{
			output_error = (output_neuron.get(0) - target) * sigmoid_prime(output_neuron.get(0)) * 2;
			
			for (int i = 0; i < hidden_error.size(); i++)
				hidden_error.get(i) = hidden_neurons.get(i) > 0 ? output_error * output_weights.get(i) : 0;
		}

		void calculate_gradients(Matrix<N_INPUT_NEURONS, 1> const& sample)
		{
			for (int i = 0; i < hidden_neurons.total_rows(); i++)
			{
				for (int j = 0; j < sample.total_rows(); j++)
				{
					hidden_weight_deltas.get(i, j) += sample.get(j) * hidden_error.get(i);
				}

				hidden_bias_deltas  .get(i) += hidden_error  .get(i);
				output_weight_deltas.get(i) += hidden_neurons.get(i) * output_error;
			}
			output_bias_deltas.get(0) += output_error;
		}

		void apply()
		{
			float rate = 0.01f;
			for (int i = 0; i < hidden_neurons.total_rows(); i++)
			{
				for (int j = 0; j < N_INPUT_NEURONS; j++)
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

		void back_propagate(Matrix<N_INPUT_NEURONS, 1> const& sample, std::vector<int> const& input_indices, float target)
		{
			feed(input_indices);
			calculate_errors(target);
			calculate_gradients(sample);
		}
	};
}