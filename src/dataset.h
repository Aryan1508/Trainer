#pragma once 
#include "sample.h"

#include <vector>
#include <fstream>
#include <iostream>
#include <string_view>

struct Dataset
{   
    Dataset(std::string_view path, std::size_t limit)
    {   
        std::ifstream file(path.data());

        if (!file)
            throw std::invalid_argument("Cannot open " + std::string(path));

        std::cout.precision(2);

        const std::size_t training_size = 0.9 * limit;

        std::size_t total_read = 0;
        training.reserve(training_size);

        for(std::string line; std::getline(file, line);)
        {
            if (total_read >= training_size)
                validation.push_back(Sample(line));
            else 
                training.push_back(Sample(line));
                
            if (++total_read >= limit)
                break;

            if (total_read % 131072 == 0)
            {
                const float p = total_read / static_cast<float>(limit) * 100.0f;
                std::cout << '\r' << std::string(' ', 32);
                std::cout << "\rLoading samples " << p << "% (" << total_read << ")" << std::flush;
            }
        }

        std::cout << '\r' << std::string(' ', 32);
        std::cout << "\rLoaded dataset '" << path << "' (" << total_read << " samples)" << std::endl;
    }

    std::vector<Sample> training;
    std::vector<Sample> validation;
};