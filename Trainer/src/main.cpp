#include "net.h"
#include "reader.h"
#include "mappings.h"

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

int main()
{
	std::unique_ptr<Trainer::Network> net = std::make_unique<Trainer::Network>();

	auto positions = Trainer::load_positions("C:/tuning/shuffled_depth_6", 5000);

    while(1)
        fit(*net, positions);
}
