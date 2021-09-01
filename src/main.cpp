#include "net.h"
#include "cost.h"
#include "sample.h"
#include "stopwatch.h"
#include <memory>
#include <iomanip>
#include <iostream>

void fit(Trainer::Network& net, std::vector<Trainer::Sample>& inputs, int epoch)
{
    int i = 0;
    StopWatch watch;
    watch.go();

    for (auto const& input : inputs)
    {
        if (++i % 16384 == 0)
        {
            net.apply_gradients();
            double eps = i / (double)watch.elapsed_time().count() * 1000;
            std::cout << "\rEpoch #" << epoch << " Evaluated [ " << i << " ]" << " EPS [ " << eps << " ] ";
        }

        net.back_propagate(input);
    }
}

int main()
{
    std::cout << std::fixed << std::setprecision(8);

    auto net = std::unique_ptr<Trainer::Network>(new Trainer::Network({768, 128, 1}));
    
    auto positions = Trainer::load_inputs("C:/tuning/8b5ed5872e.txt", 2000000);

    for (int i = 0; i < 200; i++)
    {
        fit(*net, positions, i); 
        std::cout << std::setprecision(8) << " Cost [ " << Trainer::calculate_cost(positions, *net) << " ]\n";
    }
}