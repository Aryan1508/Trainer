#pragma once
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include "matrix.h"

double sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

double sigmoidPrime(double x)
{
	return x * (1 - x);
}

class Network
{
public:
	std::vector<Matrix> neurons;
	std::vector<Matrix> weights;
	std::vector<Matrix> biases;
	std::vector<Matrix> errors;

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
				weights.push_back(Matrix(topology[i], topology[i - 1]));
			}
		}
		init();
	}

	void init()
	{
		std::mt19937 gen(4312987);
		std::uniform_real_distribution distrib(-1.0f, 1.0f);

		auto r = [&](double) { return distrib(gen); };

		for (auto& weight : weights) weight.randomize();
		for (auto& bias : biases)    bias.randomize();
	}

	Matrix const& feed(Matrix const& sample)
	{
		neurons[0] = sample;

		for (int i = 1; i < neurons.size(); i++)
			neurons[i] = (weights[i - 1] * neurons[i - 1] + biases[i - 1]).forEach(sigmoid);

		return neurons.back();
	}

	void calculateErrors(Matrix const& target)
	{
		errors.back() = (neurons.back() - target) * neurons.back().forEach(sigmoidPrime) * 2;

		for (int layer = neurons.size() - 2; layer > 0; layer--)
		{
			for (int neuron = 0; neuron < neurons[layer].totalRows(); neuron++)
			{
				double sum = 0;

				for (int nextNeuron = 0; nextNeuron < neurons[layer + 1].totalRows(); nextNeuron++)
					sum += weights[layer].get(nextNeuron, neuron) * errors[layer + 1].get(nextNeuron);

				errors[layer].get(neuron) = sum * neurons[layer].get(neuron) * (1 - neurons[layer].get(neuron));
			}
		}
	}

	double cost(Matrix const& sample, Matrix const& target)
	{
		return pow(feed(sample).get(0) - target.get(0), 2);
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

					if (std::abs(diff / avg * 100) >= 1 && !isnan(diff / avg * 100))
					{
						std::cout << std::fixed << std::setprecision(7) << diff / avg * 100 << "%\n";
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

				if (std::abs(diff / avg * 100) >= 1 && !isnan(diff / avg * 100))
				{
					std::cout << std::fixed << std::setprecision(7) << diff / avg * 100 << "%\n";
					std::cerr << "ERROR bias gradient\n";
					std::terminate();
					std::cin.get();
				}
			}
		}
	}

	void update()
	{
		for (std::size_t i = 1; i < neurons.size(); i++)
		{
			for (int neuron = 0; neuron < neurons[i].totalCols(); neuron++)
			{
				double errorSignal = errors[i].get(neuron);

				for (int prevNeuron = 0; prevNeuron < neurons[i - 1].totalRows(); prevNeuron++)
				{
					double gradient = neurons[i - 1].get(prevNeuron) * errorSignal;
					weights[i - 1].get(neuron, prevNeuron) -= gradient;
				}
				biases[i - 1].get(neuron) -= errorSignal;
			}
		}
	}

	void backPropagate(Matrix const& sample, Matrix const& target)
	{
		feed(sample);
		calculateErrors(target);
		update();
		//validateGradients(sample, target);
	}
};