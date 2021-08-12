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

double get_cost(Trainer::Network& net, std::vector<Position>& positions)
{
	double cost = 0;

	for (auto const& position : positions)
	{
	    Trainer::Matrix<Trainer::N_INPUT_NEURONS, 1> sample;
		std::vector<int> indices;

		Trainer::position_to_input(position, sample, indices);

	    net.back_propagate(sample, indices, position.result);
		cost += pow(position.result - net.output_neuron.get(0), 2);
	}
	net.apply();

	return cost;
}

template<int X1, int X2>
using M = Trainer::Matrix<X1, X2>;

int main()
{
	Trainer::Network* net = new Trainer::Network;

	std::vector<Position> positions;
	load_positions(positions);

	StopWatch watch;
	watch.go();

	for (int i = 0; i < 200000; i++)
	{
		Trainer::Matrix<Trainer::N_INPUT_NEURONS, 1> sample;
		std::vector<int> indices;

		Trainer::position_to_input(positions[0], sample, indices);
		net->feed(sample, indices);
	}

	watch.stop();

	std::cout << watch.elapsed_time().count() << '\n';

	watch.reset();
	watch.go();
	int i = 5;
	while (i--)
		std::cout << "Cost: " << get_cost(*net, positions) << '\n';

	watch.stop();

	std::cout << watch.elapsed_time().count() << '\n';
}