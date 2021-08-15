#pragma once
#include <vector>

#include "matrix.h"
#include "mappings.h"
#include "optimize.h"
#include "activation.h"	

namespace Trainer
{
    constexpr int INPUT_SIZE  = 768;
    constexpr int HIDDEN_SIZE = 256;

    template<typename T, int X>
    using ColVector = Matrix<T, X, 1>;

    using HiddenVector = ColVector<Parameter, HIDDEN_SIZE>;
    using OutputVector = ColVector<Parameter, 1>;

    class Network
    {
    public:
        Network();

        void save_network(std::string_view fil);
        
        void load_network(std::string_view fil);

        void feed(std::vector<int> const& input_indices);

        void back_propagate(NetworkInput const& sample, float target)
        {
            feed(sample);
            update_gradients(sample, target);
        }

        float get_output() const
        {
            return output_neuron.get(0).value;
        }

        void update_gradients(NetworkInput const& input, float target);
        void apply_gradients();

        HiddenVector  hidden_biases;
        HiddenVector  hidden_neurons;

        OutputVector output_bias;
        OutputVector output_neuron;

        ColMajorMatrix<Parameter, HIDDEN_SIZE, INPUT_SIZE>   hidden_weights;
        ColMajorMatrix<Parameter,  1        , HIDDEN_SIZE>  output_weights;
    };
}