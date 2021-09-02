#include "net.h"
#include "cost.h"
#include "sample.h"

#include <iostream>

static constexpr float cost_eval_weight = 0.5f;
static constexpr float cost_wdl_weight  = 1.0f - cost_eval_weight;

namespace Trainer
{
    float calculate_cost(std::vector<Sample> const& samples, Network& network)
    {
        float cost = 0.0f;

        for(auto const& sample : samples)
            cost += calculate_cost(sample, network);

        return cost / static_cast<float>(samples.size());
    }
    
    float calculate_cost(Sample const& sample, Network& network)
    {
        const float output = forward_propagate(sample.input, network);
        
        return   cost_eval_weight * powf(output - sample.eval_target, 2.0f)
               + cost_wdl_weight  * powf(output - sample.wdl_target , 2.0f);
    }

    float calculate_cost_gradient(Sample const& sample, const float output)
    {
        return   2.0f * cost_eval_weight * (output - sample.eval_target)
               + 2.0f * cost_wdl_weight  * (output - sample.wdl_target);
    }
}