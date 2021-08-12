#include "net.h"
#include "reader.h"
#include "mappings.h"

void fit(Trainer::Network& net, std::vector<Position>& positions)
{
    int i = 0;

	long double cost = 0;
    for (auto const& position : positions)
	{
        if (++i % 256 == 0)
        {
            net.apply_gradients();
            std::cout << "\rEvaluated [ " << i << " ]";
        }

	    Trainer::InputVector sample;
		std::vector<int> indices;

		Trainer::position_to_input(position, sample, indices);

	    net.back_propagate(sample, indices, position.result);
		cost += powf(net.get_output() - position.result, 2.0f);
	}
    
	std::cout << "Cost: " << std::fixed << std::setprecision(8) << (cost / (double)positions.size()) << '\n';
}

int main()
{
	std::unique_ptr<Trainer::Network> net = std::make_unique<Trainer::Network>();

	auto positions = Trainer::load_positions("C:/tuning/shuffled_depth_6", 5000000);

	for (int i = 0; i < 100000; i++)
	{
		fit(*net, positions);
		net->save_network("test.nn");
	}
}