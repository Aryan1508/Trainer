#pragma once 
#include "matrix.h"

#include <vector>

struct Neurons
{
    Neurons(std::vector<int> const& topology)
    {
        for(std::size_t i = 1;i < topology.size();i++)
            activations.push_back(Matrix<float>(topology[i]));
    }

    std::vector<Matrix<float>> activations;
};