#include "net.h"
#include "cost.h"
#include "table.h"
#include "dataset.h"
#include "optimize.h"
#include "training.h"

#include <sstream>
#include <iomanip>

namespace 
{
    constexpr int batch_size = 16384;
    constexpr int max_epochs = 256;

    std::string ftos(float value)
    {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(8) << value;
        return ss.str();
    }

    void print_cost(Table& table, Network& network, Dataset const& dataset,int epoch)
    {
        const float training_cost   = calculate_cost(dataset.training, network);
        const float validation_cost = calculate_cost(dataset.validation, network);

        std::vector<std::string> values;
        values.push_back(std::to_string(epoch));
        values.push_back(ftos(training_cost));
        values.push_back(ftos(validation_cost));

        table.print_value_row(values);
    }

    void run_epoch(Network& network, Dataset const& dataset, Gradients& gradients)
    {
        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);

        for(std::size_t i = 0;i < dataset.training.size();i++)
        {
            calculate_gradients(dataset.training[i], network, gradients);

            if(i && i % batch_size == 0)
            {
                apply_gradients(network, gradients);
                
                const double completed = i / static_cast<double>(dataset.training.size()) * 100;
                std::cout << "\rCompleted " << completed << "%";
            }
        }
        std::cout << '\r';
    }
}

void train_network(Network& network, Dataset const& dataset, Gradients& gradients)
{
    Table table(std::cout, 32, {"Epoch", "Training", "Validation"});

    table.print_headers();
    for(int epoch = 1;epoch <= max_epochs;epoch++)
    {
        run_epoch(network, dataset, gradients);
        print_cost(table, network, dataset, epoch);
    }
}