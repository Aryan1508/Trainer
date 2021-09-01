#pragma once 

namespace Trainer
{
    class Sample;
    class Network;

    float calculate_output_gradient(Sample const& sample, Network& net);

    float calculate_output_gradient(Sample const& sample, const float output);

    float calculate_cost_gradient(Sample const& sample, const float output);
}