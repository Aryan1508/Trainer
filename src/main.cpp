#include "net.h"
#include "reader.h"
#include "mappings.h"
#include "stopwatch.h"
#include <memory>
#include <iomanip>

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
            std::cout << "\rEvaluated [ " << i << " ]" << " EPS [ " << eps << " ] ";
        }

        auto sample = Trainer::position_to_input(position);

        net.back_propagate(sample, position.result);
        cost += powf(net.get_output() - position.result, 2.0f);
    }
    
    std::cout << " Cost [ " << (cost / (double)positions.size()) << " ]\n";
}

int main()
{
    std::cout << std::fixed << std::setprecision(8);

    std::unique_ptr<Trainer::Network> net = std::make_unique<Trainer::Network>();
    net->load_network("14_8_256h.nn");

    auto positions = Trainer::load_positions("C:/tuning/lichess-big3.txt", 10000);

    for (int i = 0; i < 100000; i++)
    {
        fit(*net, positions);
    }
}