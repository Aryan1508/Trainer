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

	class Network
	{
	public:
		std::vector<Matrix> neurons;
		std::vector<Matrix> weights;
		std::vector<Matrix> biases;
		std::vector<Matrix> errors;
		std::vector<Matrix> wgradients;
		std::vector<Matrix> bgradients;
		std::vector<Matrix> wgrad_total;
		std::vector<Matrix> bgrad_total;

		Network(std::vector<int> const& topology)
		{
			for (int i = 0; i < topology.size(); i++)
			{
				neurons.push_back(Matrix(topology[i], 1));
				errors.push_back(Matrix(topology[i], 1));

				neurons.back().set(0);
				errors.back().set(0);

				if (i > 0)
				{
					biases.push_back(Matrix(topology[i], 1));
					biases.back().randomize();
					weights.push_back(Matrix(topology[i], topology[i - 1]));
					weights.back().randomize();
				}
			}
		}

		Matrix const& feed(Matrix const& sample)
		{
			neurons[0] = sample;

			for (int i = 1; i < neurons.size(); i++)
			{
				Matrix z = (weights[i - 1] * neurons[i - 1] + biases[i - 1]);
				neurons[i] = i == neurons.size() - 1 ? z.for_each(sigmoid) : z.for_each(relu);
			}

			return neurons.back();
		}

		void calculate_errors(Matrix const& target)
		{
			errors.back() = (neurons.back() - target) * neurons.back().for_each(sigmoid_prime) * 2;

			for (std::size_t layer = neurons.size() - 2; layer > 0; layer--)
			{
				for (int neuron = 0; neuron < neurons[layer].totalRows(); neuron++)
				{
					double sum = 0;

					for (int next_neuron = 0; next_neuron < neurons[layer + 1].totalRows(); next_neuron++)
						sum += weights[layer].get(next_neuron, neuron) * errors[layer + 1].get(next_neuron);

					errors[layer].get(neuron) = sum * reluD(neurons[layer].get(neuron));
				}
			}
		}

		void validateGradients(Matrix const& sample, Matrix const& target)
		{
			for (std::size_t i = 1; i < neurons.size(); i++)
			{
				for (int neuron = 0; neuron < neurons[i].totalCols(); neuron++)
				{
					double errorSignal = errors[i].get(neuron);

					for (int prevNeuron = 0; prevNeuron < neurons[i - 1].totalRows(); prevNeuron++)
					{
						double gradient = neurons[i - 1].get(prevNeuron) * errorSignal;

						double delta = 0.001;

						double E1 = cost(sample, target);

						weights[i - 1].get(neuron, prevNeuron) += delta;

						double E2 = cost(sample, target);
						weights[i - 1].get(neuron, prevNeuron) -= delta;

						double FD = (E2 - E1) / delta;

						double avg = FD + gradient;
						double diff = std::abs(FD - gradient);
                        double deviation = diff / avg * 100;

						if (std::abs(diff / avg * 100) >= 1 && !isnan(diff / avg * 100))
						{
							std::cerr << "ERROR weight gradient\n";
							std::terminate();
							std::cin.get();
						}
					}
					double gradient = errorSignal;
					double delta = 0.001;

					double E1 = cost(sample, target);

					biases[i - 1].get(neuron) += delta;

					double E2 = cost(sample, target);
					biases[i - 1].get(neuron) -= delta;


					double FD = (E2 - E1) / delta;

					double avg = FD + gradient;
					double diff = std::abs(FD - gradient);
					double deviation = diff / avg * 100;

					if (deviation >= 1 && !std::isnan(deviation))
					{
						std::cout << std::fixed << std::setprecision(7) << diff / avg * 100 << "%\n";
						std::cerr << "ERROR bias gradient\n";
						std::terminate();
						std::cin.get();
					}
				}
			}
		}

		double cost(Matrix const& sample, Matrix const& target)
		{
			return pow(feed(sample).get(0) - target.get(0), 2);
		}

		void update()
		{
			for (std::size_t i = 1; i < neurons.size(); i++)
			{
				for (int neuron = 0; neuron < neurons[i].totalCols(); neuron++)
				{
					double error_signal = errors[i].get(neuron);

					for (int prev_neuron = 0; prev_neuron < neurons[i - 1].totalRows(); prev_neuron++)
					{
						double gradient = neurons[i - 1].get(prev_neuron) * error_signal;

						weights[i - 1].get(neuron, prev_neuron) -= 0.01 * gradient;
					}
					biases[i - 1].get(neuron) -= 0.01 * error_signal;
				}
			}
		}


		void back_propagate(Matrix const& sample, Matrix const& target)
		{
			feed(sample);
			calculate_errors(target);
			 update();
			//validateGradients(sample, target);
		}
	};
}