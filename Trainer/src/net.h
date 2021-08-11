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

	constexpr int N_INPUT_NEURONS = 768;
	constexpr int N_HIDDEN_NEURONS = 16;

	class Network
	{
	public:
		Matrix<N_HIDDEN_NEURONS, 1> hidden_neurons;
		Matrix<  1             , 1> output_neuron;
		
		Matrix<N_HIDDEN_NEURONS, N_INPUT_NEURONS> hidden_weights;
		Matrix<  1            , N_HIDDEN_NEURONS> output_weights;

		Matrix<N_HIDDEN_NEURONS, 1> hidden_biases;
		Matrix<  1             , 1> output_bias;

		Matrix<N_HIDDEN_NEURONS, 1> hidden_error;
		double output_error;

		Network()
		{
			hidden_weights.randomize(N_INPUT_NEURONS);
			output_weights.randomize(N_HIDDEN_NEURONS);

			hidden_biases.randomize(N_INPUT_NEURONS);
			output_bias.randomize(N_HIDDEN_NEURONS);

			hidden_neurons.set(0);
			hidden_error.set(0);
			output_error = 0;
			output_neuron.set(0);
		}

		double feed(Matrix<N_INPUT_NEURONS, 1> const& sample)
		{
			forward_propagate(hidden_weights, sample, hidden_biases, hidden_neurons, relu);
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
			double rate = 0.2;
			for (int i = 0; i < hidden_neurons.totalRows(); i++)
			{
				for (int j = 0; j < sample.totalRows(); j++)
				{
					hidden_weights.get(i, j) -= rate * sample.get(j) * hidden_error.get(i);
				}
				hidden_biases.get(i) -= rate * hidden_error.get(i);
				output_weights.get(i) -= rate * output_error * hidden_error.get(i);
			}
		}

		double cost(Matrix<N_INPUT_NEURONS, 1> const& sample, double target)
		{
			return pow(feed(sample) - target, 2);
		}

		void back_propagate(Matrix<N_INPUT_NEURONS, 1> const& sample, double target)
		{
			feed(sample);
			calculate_errors(target);
			updateWeights(sample);
		}
	};
}