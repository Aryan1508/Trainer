#pragma once
#include <vector>

#include "matrix.h"
#include "reader.h"
#include "optimize.h"
#include "activation.h"	

namespace Trainer
{
    
    class Network
    {
    public:
        static constexpr int INPUT_SIZE  = 768;
        static constexpr int HIDDEN_SIZE = 256;
        static constexpr int OUTPUT_SIZE = 1;

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
            return output_neuron.get(0);
        }

        void update_gradients(NetworkInput const& input);
        void apply_gradients();


        Matrix<float, HIDDEN_SIZE, 1> hidden_neurons;
        Matrix<float, OUTPUT_SIZE, 1> output_neuron;

        Matrix<Parameter, OUTPUT_SIZE, 1> output_bias;
        Matrix<Parameter, HIDDEN_SIZE, 1> hidden_biases;

        Matrix<Parameter, HIDDEN_SIZE, INPUT_SIZE>   hidden_weights;
        Matrix<Parameter, OUTPUT_SIZE, HIDDEN_SIZE>  output_weights;
    };
}