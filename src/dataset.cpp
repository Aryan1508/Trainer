#include "cost.h"
#include "dataset.h"

#include <fstream>
#include <iomanip>
#include <iostream>

static std::size_t calculate_training_size(std::size_t dataset_size)
{
    return static_cast<std::size_t>(0.9f * dataset_size);
}

Dataset::Dataset(std::string_view path, std::size_t limit)
{   
    std::cout.precision(2);

    std::ifstream file(path.data());

    if (!file)
        throw std::invalid_argument("Cannot open " + std::string(path));

    std::size_t total_read = 0;

    for(std::string line; std::getline(file, line);)
    {   
        if (total_read >= calculate_training_size(limit))
            validation.push_back(Sample(line));
        else 
            training.push_back(Sample(line));

        if (++total_read >= limit)
            break;

        if (total_read % 32768 == 0)
        {
            const float p = total_read / static_cast<float>(limit) * 100.0f;
            std::cout << "\rLoading samples " << p << '%' << std::flush;
        }
    }
    std::cout << '\r' << std::string(' ', 32);
    std::cout << "\rLoaded dataset '" << path << "' (" << total_read << ')' << std::endl;
}