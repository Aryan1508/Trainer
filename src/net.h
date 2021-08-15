#pragma once
#include <vector>

#include "matrix.h"
#include "reader.h"
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

        void feed(NetworkInput const&);

        void back_propagate(NetworkInput const& sample)
        {
            feed(sample);
            update_gradients(sample);
        }

        float get_output() const
        {
            return output_neuron.get(0).value;
        }

        void update_gradients(NetworkInput const& input);
        void apply_gradients();

        HiddenVector  hidden_biases;
        HiddenVector  hidden_neurons;

        OutputVector output_bias;
        OutputVector output_neuron;

        ColMajorMatrix<Parameter, HIDDEN_SIZE, INPUT_SIZE>   hidden_weights;
        ColMajorMatrix<Parameter,  1        , HIDDEN_SIZE>  output_weights;
    };
}