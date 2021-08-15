#include "net.h"
#include "reader.h"
#include "stopwatch.h"
#include <memory>
#include <iomanip>

void fit(Trainer::Network& net, std::vector<Trainer::NetworkInput>& inputs)
{
    int i = 0;
    StopWatch watch;
    watch.go();

    long double cost = 0;
    for (auto const& input : inputs)
    {
        if (++i % 4096 == 0)
        {
            net.apply_gradients();
            double eps = i / (double)watch.elapsed_time().count() * 1000;
            std::cout << "\rEvaluated [ " << i << " ]" << " EPS [ " << eps << " ] ";
        }

        net.back_propagate(input);
        cost += powf(net.get_output() - input.target, 2.0f);
    }
    
    std::cout << " Cost [ " << (cost / (double)inputs.size()) << " ]\n";
}

int main()
{
    std::cout << std::fixed << std::setprecision(8);

    std::unique_ptr<Trainer::Network> net = std::make_unique<Trainer::Network>();
    net->load_network("14_8_256h.nn");

    auto positions = Trainer::load_inputs("C:/tuning/lichess-big3.txt", 100000);

    for (int i = 0; i < 100000; i++)
    {
        fit(*net, positions);
    }
}