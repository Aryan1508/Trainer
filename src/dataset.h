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
        const std::size_t training_size = 0.9 * limit;

        std::cout.precision(2);

        std::ifstream file(path.data());

        if (!file)
            throw std::invalid_argument("Cannot open " + std::string(path));

        std::size_t total_read = 0;

        for(std::string line; std::getline(file, line);)
        {   
            if (total_read >= training_size)
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

    std::vector<Sample> training;
    std::vector<Sample> validation;
};