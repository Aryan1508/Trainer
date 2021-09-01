#pragma once
#include <vector>

namespace Trainer
{
    class Sample;
    class Network;

    float calculate_cost(std::vector<Sample> const& samples, Network& network);

    float calculate_cost(Sample const& sample, Network& network);

    float calculate_cost_gradient(Sample const& sample, Network& network);
}