#include "net.h"
#include <iostream>
#include <iomanip>
#include "position.h"
#include "mappings.h"
#include <fstream>
#include "stopwatch.h"

#define NPOSITIONS 5000

void load_positions(std::vector<Position>& positions)
{
	positions.reserve(NPOSITIONS);

	int i = 0;
	Position position;
	std::ifstream fil("C:/tuning/shuffled_depth_6");

	for (std::string line; std::getline(fil, line);)
	{
		if (++i >= NPOSITIONS)
			break;

		position.set_fen(line.substr(0, line.find("[") - 1));

		if (line.find("[1.0]") != line.npos)        position.result = 1.0;
		else if (line.find("[0.0]") != line.npos)   position.result = 0.0;
		else                                        position.result = 0.5;

		positions.push_back(position);
	}
	fil.close();
}

void fit(Trainer::Network& net, std::vector<Position>& positions)
{
    double cost = 0;

    int i = 0;

    for (auto const& position : positions)
	{
        i++;

	    Trainer::InputVector sample;
		std::vector<int> indices;

		Trainer::position_to_input(position, sample, indices);

	    net.back_propagate(sample, indices, position.result);
		cost += pow(position.result - net.get_output(), 2);
	}
    net.apply_gradients();
    std::cout << "Cost: " << cost << '\n';            
}

double get_cost(Trainer::Network& net, std::vector<Position>& positions)
{
	double cost = 0;

	for (auto const& position : positions)
	{
	    Trainer::InputVector sample;
		std::vector<int> indices;

		Trainer::position_to_input(position, sample, indices);

	    net.back_propagate(sample, indices, position.result);
		cost += pow(position.result - net.get_output(), 2);
	}
	net.apply_gradients();

	return cost;
}

template<int X1, int X2>
using M = Trainer::Matrix<X1, X2>;

int main()
{
	Trainer::Network* net = new Trainer::Network;

    std::vector<Position> positions;
    load_positions(positions);

    while(1)
        fit(*net, positions);
}
