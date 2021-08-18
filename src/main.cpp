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
        if (++i % 16384 == 0)
        {
            net.apply_gradients();
            double eps = i / (double)watch.elapsed_time().count() * 1000;
            std::cout << "\rEvaluated [ " << i << " ]" << " EPS [ " << eps << " ] ";
        }

        net.back_propagate(input);
        cost += net.get_cost(input);
    }
    
    std::cout << " Cost [ " << (cost / (double)inputs.size()) << " ]\n";
}

int main()
{
    std::cout << std::fixed << std::setprecision(8);

    auto net = std::make_unique<Trainer::Network>();

    auto positions = Trainer::load_inputs("C:/tuning/koi-6.15-d10.book");

    for (int i = 0; i < 100; i++)
    {
        fit(*net, positions);
        net->save_network("a33e3d1337.nn");
    }
}