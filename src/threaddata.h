#pragma once
#include "net.h"
#include "optimize.h"

#include <vector>

struct ThreadData
{
    ThreadData(std::vector<int> const& topology)
        : network(topology), gradients(topology)
    {}

    Network   network;
    Gradients gradients;
};