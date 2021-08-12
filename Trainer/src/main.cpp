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

		if (i % 16000 == 0)
		{
			net.apply_gradients();
			std::cout << "Cost: " << cost << '\n';
		}

	    Trainer::InputVector sample;
		std::vector<int> indices;

		Trainer::position_to_input(position, sample, indices);

	    net.back_propagate(sample, indices, position.result);
		cost += pow(position.result - net.get_output(), 2);
	}
}

int main()
{
	std::unique_ptr<Trainer::Network> net = std::make_unique<Trainer::Network>();
	net->load_network("test.nn");

	auto positions = Trainer::load_positions("C:/tuning/shuffled_depth_6", 10000000);

	for (int i = 0; i < 1000; i++)
	{
		fit(*net, positions);
		net->save_network("test.nn");
	}
}