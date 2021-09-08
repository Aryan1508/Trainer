#pragma once
#include <vector>

#include "net.h"
#include "sample.h"

constexpr float COST_EVAL_WEIGHT = 0.5f;
constexpr float COST_WDL_WEIGHT  = 1.0f - COST_EVAL_WEIGHT;

inline float calculate_cost(Sample const& sample, Network& network)
{
    const float output = forward_propagate(sample.input, network);
    
    return  COST_EVAL_WEIGHT * powf(output - sample.eval_target, 2.0f)
          + COST_WDL_WEIGHT  * powf(output - sample.wdl_target , 2.0f);
}

inline float calculate_cost(std::vector<Sample> const& samples, Network& network)
{
    float cost = 0.0f;

    for(auto const& sample : samples)
        cost += calculate_cost(sample, network);

    return cost / static_cast<float>(samples.size());
}

inline float calculate_cost_gradient(Sample const& sample, const float output)
{
    return    2.0f * COST_EVAL_WEIGHT * (output - sample.eval_target)
            + 2.0f * COST_WDL_WEIGHT  * (output - sample.wdl_target);
}