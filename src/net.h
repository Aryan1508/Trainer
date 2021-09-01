#pragma once
#include <vector>
#include "sample.h"
#include "matrix.h"
#include "optimize.h"
#include "activation.h"	

namespace Trainer
{
    class Network;

    float forward_propagate(Input const& input, Network&);
    
    class Network
    {
    public:
        Network(std::vector<int> const& topology);

        void back_propagate(Sample const& sample)
        {
            forward_propagate(sample.input, *this);
            update_gradients(sample);
        }

        void update_gradients(Sample const& input);
        void apply_gradients();

        std::vector<Matrix<float>> neurons;
        std::vector<Matrix<float>> weights;
        std::vector<Matrix<float>> biases;

        std::vector<Matrix<Parameter>> weight_gradients;
        std::vector<Matrix<Parameter>> bias_gradients;
    };
}