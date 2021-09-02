#include "net.h"
#include "cost.h"
#include "sample.h"
#include "optimize.h"
#include "stopwatch.h"
#include <memory>
#include <iomanip>
#include <iostream>

void fit(Trainer::Network& net, Trainer::Gradients& gradients, std::vector<Trainer::Sample>& inputs, int epoch)
{
    int i = 0;
    StopWatch watch;
    watch.go();

    for (auto const& input : inputs)
    {
        if (++i % 16384 == 0)
        {
            Trainer::apply_gradients(net, gradients);

            double eps = 16384 / (double)watch.elapsed_time().count() * 1000;
            std::cout << "\rEpoch #" << epoch << " Evaluated [ " << i << " ]" << " EPS [ " << eps << " ] ";
            
            watch.reset();
            watch.go();
        }

        Trainer::calculate_gradients(input, net, gradients);
    }
}

int main()
{
    std::cout << std::fixed << std::setprecision(8);

    auto net = std::unique_ptr<Trainer::Network>(new Trainer::Network({768, 128, 1}));
    
    auto positions = Trainer::load_samples("C:/tuning/8b5ed5872e.txt", 200000);

    auto gradients = std::unique_ptr<Trainer::Gradients>(new Trainer::Gradients({768, 128, 1}));

    for (int i = 0; i < 200; i++)
    {
        std::cout << std::setprecision(8) << " Cost [ " << Trainer::calculate_cost(positions, *net) << " ]\n";
        fit(*net, *gradients, positions, i); 
    }
}