#pragma once
#include <vector>

#include "net.h"
#include "sample.h"
#include "neurons.h"

#include <thread>

constexpr float COST_EVAL_WEIGHT = 0.5f;
constexpr float COST_WDL_WEIGHT  = 1.0f - COST_EVAL_WEIGHT;

inline float calculate_cost(Sample const& sample, Network const& network, Neurons& neurons)
{
    const float output = forward_propagate(sample.input, network, neurons);
    
    return  COST_EVAL_WEIGHT * powf(output - sample.eval_target, 2.0f)
          + COST_WDL_WEIGHT  * powf(output - sample.wdl_target , 2.0f);
}

inline float calculate_cost(std::vector<Sample> const& samples, Network const& network)
{
    float cost = 0.0f;
    Neurons neurons(get_topology(network));

    for(auto const& sample : samples)
        cost += calculate_cost(sample, network, neurons);

    return cost / static_cast<float>(samples.size());
}

inline float calculate_cost_gradient(Sample const& sample, const float output)
{
    return    2.0f * COST_EVAL_WEIGHT * (output - sample.eval_target)
            + 2.0f * COST_WDL_WEIGHT  * (output - sample.wdl_target);
}