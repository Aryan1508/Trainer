#include "net.h"
#include "cost.h"
#include "dataset.h"
#include "optimize.h"

#include <iostream>

int main()
{
    Dataset   dataset("C:/tuning/Datasets/8b5ed5872e.txt", 2000000);
    Network   network({768, 128, 1});
    Gradients gradients({768, 128, 1});
    
    std::cout << "Training size=" << dataset.training.size() << '\n';

    while (true)
    {
        for(std::size_t i = 0;i < dataset.training.size();i++)
        {
            calculate_gradients(dataset.training[i], network, gradients);

            if (i % 16384 == 0)
            {
                apply_gradients(network, gradients);
                std::cout << "\rEvaluated [" << i + 1 << "]";
            }
        }

        std::cout << '\r' << std::string(' ', 32) << std::endl;
        std::cout.precision(8);

        const float validation_cost = calculate_cost(dataset.validation, network);
        const float training_cost   = calculate_cost(dataset.training, network);

        std::cout << "Validation [" << validation_cost << "] ";
        std::cout << "Training   [" << training_cost   << "]\n";
    }
}