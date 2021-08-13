#include "net.h"
#include "reader.h"
#include "mappings.h"
#include "stopwatch.h"

void fit(Trainer::Network& net, std::vector<Position>& positions)
{
    int i = 0;
	StopWatch watch;
	watch.go();

	long double cost = 0;
    for (auto const& position : positions)
	{
        if (++i % 4096 == 0)
        {
			net.apply_gradients();
			double eps = i / (double)watch.elapsed_time().count() * 1000;
            std::cout << "\rEvaluated [ " << i << " ]" << "EPS [ " << eps << " ] ";
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
	//net->load_network("shuffled_depth_6.nn");

	auto positions = Trainer::load_positions("C:/tuning/shuffled_depth_6.book", 10000);

	for (int i = 0; i < 100000; i++)
	{
		fit(*net, positions);
		//net->save_network("shuffled_depth_6.nn");
	}
}