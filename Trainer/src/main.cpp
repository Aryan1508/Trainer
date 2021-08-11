#include "net.h"
#include <iostream>
#include <iomanip>
#include "position.h"
#include "mappings.h"
#include <fstream>
#include "stopwatch.h"

#define NPOSITIONS 5000

double get_cost(Trainer::Network& net)
{
	double cost = 0;

	int i = 0;
	Position position;
	std::ifstream fil("C:/tuning/shuffled_depth_6");

	for (std::string line; std::getline(fil, line);)
	{
		if (++i >= NPOSITIONS)
			break;

		double target = 0;

		std::string fen = line.substr(0, line.find("[") - 1);

		if (line.find("[1.0]") != line.npos)        target = 1.0;
		else if (line.find("[0.0]") != line.npos)   target = 0.0;
		else                                        target = 0.5;

		position.set_fen(fen);

		auto sample = Trainer::position_to_input(position);

	    net.back_propagate(sample, target);
		cost += pow(target - net.output_neuron.get(0), 2);
	}
	fil.close();

	return cost;
}

template<int X1, int X2>
using M = Trainer::Matrix<X1, X2>;

int main()
{
	Trainer::Network* net = new Trainer::Network;

	while (1)
		std::cout << "Cost over " << NPOSITIONS << " positions: " << std::fixed << std::setprecision(8) << get_cost(*net) << '\n';
}