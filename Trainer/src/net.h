#pragma once
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include "matrix.h"
#include <math.h>

namespace Trainer
{
	double sigmoid(double x)
	{
		return 1 / (1 + exp(-x * (2.5 / 400.0f)));
	}

	double sigmoid_prime(double x)
	{
		return x * (1 - x) * (2.5 / 400.0f);
	}

	double relu(double x)
	{
		return std::max<double>(x, 0);
	}

	double reluD(double x)
	{
		return x > 0 ? 1 : 0;
	}

	constexpr int N_INPUT_NEURONS = 768;
	constexpr int N_HIDDEN_NEURONS = 8;


	template<int X1, int Y1, int X2, int Y2, typename Callable>
	inline void forward_propagate(
		Trainer::Matrix<X1, Y1> const& weights, 
		Trainer::Matrix<X2, Y2> const& neurons, 
		Trainer::Matrix<X1, Y2> const& biases, 
		Trainer::Matrix<X1, Y2>& result_neurons, Callable activation)
	{
		assert(weights.totalCols() == neurons.totalRows());

		for (int i = 0; i < weights.totalRows(); i++)
		{
			double sum = 0;
			for (int k = 0; k < weights.totalCols(); k++)
				sum += weights.get(i, k) * neurons.get(k);

			result_neurons.get(i) = activation(sum + biases.get(i));
		}
	}

	inline void forward_propagate(
		Trainer::Matrix<N_HIDDEN_NEURONS, N_INPUT_NEURONS> const& weights,
		std::vector<int>        const& indices,
		Trainer::Matrix<N_HIDDEN_NEURONS, 1> const& biases,
		Trainer::Matrix<N_HIDDEN_NEURONS, 1>& result_neurons)
	{
		assert(weights.totalCols() == neurons.totalRows());

		for (int i = 0; i < weights.totalRows(); i++)
		{
			double sum = 0;
			for (auto index : indices)
				sum += weights.get(i, index);

			result_neurons.get(i) = relu(sum + biases.get(i));
		}
	}
	
	class Network
	{
	public:
		Matrix<N_HIDDEN_NEURONS, 1> hidden_neurons;
		Matrix<  1             , 1> output_neuron;
		
		Matrix<N_HIDDEN_NEURONS, N_INPUT_NEURONS> hidden_weights, hidden_weight_deltas;
		Matrix<  1            , N_HIDDEN_NEURONS> output_weights, output_weight_deltas;

		Matrix<N_HIDDEN_NEURONS, 1> hidden_biases, hidden_bias_deltas;
		Matrix<  1             , 1> output_bias, output_bias_deltas;

		Matrix<N_HIDDEN_NEURONS, 1> hidden_error;
		double output_error;


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

		double feed(Matrix<N_INPUT_NEURONS, 1> const& sample, std::vector<int> const& input_indices)
		{
			forward_propagate(hidden_weights, input_indices, hidden_biases, hidden_neurons);
			forward_propagate(output_weights, hidden_neurons, output_bias, output_neuron, sigmoid);
			return output_neuron.get(0);
		}

		void calculate_errors(double target)
		{
			output_error = (output_neuron.get(0) - target) * sigmoid_prime(output_neuron.get(0)) * 2;
			
			for (int i = 0; i < hidden_error.size(); i++)
				hidden_error.get(i) = hidden_neurons.get(i) > 0 ? output_error * output_weights.get(i) : 0;
		}

		void updateWeights(Matrix<N_INPUT_NEURONS, 1> const& sample)
		{
			for (int i = 0; i < hidden_neurons.totalRows(); i++)
			{
				for (int j = 0; j < sample.totalRows(); j++)
				{
					hidden_weight_deltas.get(i, j) += sample.get(j) * hidden_error.get(i);
				}

				hidden_bias_deltas.get(i) += hidden_error.get(i);
				output_weight_deltas.get(i) += output_error * hidden_neurons.get(i);
			}
			output_bias_deltas.get(0) += output_error;
		}

		void apply()
		{
			double rate = 0.05;
			for (int i = 0; i < hidden_neurons.totalRows(); i++)
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

		void back_propagate(Matrix<N_INPUT_NEURONS, 1> const& sample, std::vector<int> const& input_indices, double target)
		{
			feed(sample, input_indices);
			calculate_errors(target);
			updateWeights(sample);
		}
	};
}