#pragma once
#include <vector>

#include "net.h"
#include "sample.h"
#include "neurons.h"

#include <thread>
#include <algorithm>

constexpr float COST_EVAL_WEIGHT = 0.5f;
constexpr float COST_WDL_WEIGHT  = 1.0f - COST_EVAL_WEIGHT;

inline float calculate_cost(Sample const& sample, Network const& network, Neurons& neurons)
{
    const float output = forward_propagate(sample.position.to_features(), network, neurons);
    
    return  COST_EVAL_WEIGHT * powf(output - sample.eval_target, 2.0f)
          + COST_WDL_WEIGHT  * powf(output - calculate_wdl_target(sample.wdl_value), 2.0f);
}

inline float calculate_total_cost(Network const& network, 
    std::vector<Sample>::const_iterator begin, std::vector<Sample>::const_iterator end, float& sum)
{
    Neurons neurons(get_topology(network));

    while(begin++ != end)
        sum += calculate_cost(*begin, network, neurons);

    return sum;
}

inline float calculate_average_cost(Network const& network, std::vector<Sample> const& samples, const int n_threads)
{
    const std::size_t local_size = samples.size() / n_threads;
    const std::size_t total      = local_size * n_threads;

    std::vector<std::thread> threads;
    std::vector<float> sums(n_threads, 0.0f);

    for(int i = 0;i < n_threads;i++)
    {
        const auto begin = samples.begin() + i * local_size;
        const auto end   = begin + local_size; 
        threads.emplace_back(calculate_total_cost, std::ref(network), begin, end, std::ref(sums[i]));
    }

    for(auto& thread : threads) thread.join();

    return std::accumulate(sums.begin(), sums.end(), 0.0f) / static_cast<float>(total);
}

inline float calculate_cost_gradient(Sample const& sample, const float output)
{
    return    2.0f * COST_EVAL_WEIGHT * (output - sample.eval_target)
            + 2.0f * COST_WDL_WEIGHT  * (output - calculate_wdl_target(sample.wdl_value));
}