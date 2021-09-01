#include "net.h"
#include "cost.h"
#include "sample.h"
#include "gradients.h"
#include "activation.h"

namespace Trainer
{
    float calculate_output_gradient(Sample const& sample, Network& network)
    {
        const float output = forward_propagate(sample.input, network);

        return calculate_output_gradient(sample, output);
    }

    float calculate_output_gradient(Sample const& sample, const float output)
    {
        return calculate_cost_gradient(sample, output) * sigmoid_prime(output);
    }
}