#include "net.h"
#include <iostream>
#include <iomanip>

double getCost(Network& net, std::vector<std::vector<float>>& samples)
{
	Matrix sample(2, 1);
	Matrix target(1, 1);

	double cost = 0.0f;
	for (auto S : samples)
	{
		sample.get(0) = S[0];
		sample.get(1) = S[1];
		target.get(0) = S[2];

		Matrix prediction = net.feed(sample);
		cost += pow(S[2] - prediction.get(0), 2);

		net.backPropagate(sample, target);
	}
	return cost;
}

int main()
{
	Network net({ 2, 3, 1 });

	std::vector<std::vector<float>> samples
	{
		{ 1.0f, 1.0f, 1.0f },
		{ 1.0f, 0.0f, 1.0f },
		{ 0.0f, 1.0f, 1.0f },
		{ 0.0f, 0.0f, 0.0f },
	};

	int i = 10000;
	while (i--)
	{
		double cost = getCost(net, samples);
		std::cout << "Cost: " << std::fixed << std::setprecision(8) << ": " << cost << '\n';
	}

	Matrix sample(2, 1);

	for (auto S : samples)
	{
		sample.get(0) = S[0];
		sample.get(1) = S[1];
		std::cout << "Prediction: " << S[0] << " ^ " << S[1] << ": " << net.feed(sample).get(0) << '\n';
	}
}