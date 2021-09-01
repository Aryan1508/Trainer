#pragma once
#include <vector>
#include "sample.h"
#include "matrix.h"
#include "optimize.h"
#include "activation.h"	

namespace Trainer
{
    class Sample;
    
    class Network
    {
    public:
        static constexpr int INPUT_SIZE  = 768;
        static constexpr int HIDDEN_SIZE = 128;
        static constexpr int OUTPUT_SIZE = 1;

        Network(std::vector<int> const& topology);

        float feed(Sample const&);

        void back_propagate(Sample const& sample)
        {
            feed(sample);
            update_gradients(sample);
        }

        float get_output() const
        {
            return neurons.back()(0);
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