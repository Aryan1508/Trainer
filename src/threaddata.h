#pragma once
#include "net.h"
#include "optimize.h"

#include <vector>

struct ThreadData
{
    ThreadData(std::vector<int> const& topology)
        : neurons(topology), gradients(topology)
    {}

    Neurons   neurons;
    Gradients gradients;
};