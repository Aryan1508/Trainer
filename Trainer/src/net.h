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

	constexpr int TOTAL_INPUTS = 

	class Network
	{
	public:
		Matrix<512, 1> hidden_neurons;
		double output_neuron;
		
		Matrix<512, 768> hidden_weights;
		Matrix<  1, 512> output_weights;

		Matrix<512, 1> hidden_biases;
		Matrix<  1, 1> output_bias;

		Matrix<512, 1> hidden_error;
		double         output_error;

		Network()
		{
			hidden_weights.randomize(768);
			output_weights.randomize(512);

			hidden_biases.randomize(768);
			output_bias.randomize(512);

			hidden_neurons.set(0);
			hidden_error.set(0);
			output_error = 0;
			output_neuron = 0;
		}

		double feed(Matrix<768, 1> const& sample)
		{
			hidden_neurons = (hidden_weights * sample + hidden_biases).for_each(relu);
			output_neuron =  (output_weights * hidden_neurons + output_bias).for_each(sigmoid).get(0);
			return output_neuron;
		}

		void calculate_errors(double target)
		{
			output_error = (output_neuron - target) * sigmoid_prime(output_neuron) * 2;
			
			for (int i = 0; i < hidden_error.size(); i++)
				hidden_error.get(i) = hidden_neurons.get(i) > 0 ? output_error * output_weights.get(i) : 0;
		}

		void updateWeights(Matrix<768, 1> const& sample)
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

				/*double E1 = cost(sample, target);
				output_weights.get(i) += 1e-6;

				double E2 = cost(sample, target);
				output_weights.get(i) -= 1e-6;

				double fd = (E2 - E1) / 1e-6;
				double avg = (fd + gradient) / 2;
				double deviation = std::abs(fd - avg) / avg * 100;

				if (!isnan(deviation) && std::fabs(fd) > 0.00000001)
				{
					std::cout << deviation << '\n';
				}*/
			}
		}

		double cost(Matrix<768, 1> const& sample, double target)
		{
			return pow(feed(sample) - target, 2);
		}

		void back_propagate(Matrix<768, 1> const& sample, double target)
		{
			feed(sample);
			calculate_errors(target);
			//update();
			updateWeights(sample);
		}
	};
}