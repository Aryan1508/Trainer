#include "net.h"
#include <iostream>
#include <iomanip>
#include "position.h"
#include "mappings.h"
#include <fstream>
#include "stopwatch.h"

#define NPOSITINOS 5000

double get_cost(Trainer::Network& net)
{
	double cost = 0;

	int i = 0;
	Position position;
	std::ifstream fil("C:/tuning/lichess-quiet.txt");

	for (std::string line; std::getline(fil, line);)
	{
		if (++i >= NPOSITINOS)
			break;

		double target = 0;

		std::string fen = line.substr(0, line.find("[") - 1);

		if (line.find("[1.0]") != line.npos)        target = 1.0;
		else if (line.find("[0.0]") != line.npos)   target = 0.0;
		else                                        target = 0.5;

		position.set_fen(fen);

		auto sample = Trainer::position_to_input(position);

		cost += pow(target - net.output_neuron, 2);
	    net.back_propagate(sample, target);
	}
	fil.close();

	return cost;
}

int main()
{
	Trainer::Network* net = new Trainer::Network;

    while (1)
    {
        std::cout << "Cost over " << NPOSITINOS << " positions: " << std::fixed << std::setprecision(8) << get_cost(*net) << '\n';
    }
}