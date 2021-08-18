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
        static constexpr int HIDDEN_SIZE = 128;
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
        
        float get_cost(NetworkInput const& sample)
        {
            float o = get_output();
            return powf(sample.target - o, 2.0f) * 0.5f + powf(sample.eval_target - o, 2.0f) * 0.5f;
        }

        void update_gradients(NetworkInput const& input);
        void apply_gradients();

        Matrix<float, HIDDEN_SIZE, 1> hidden_neurons;
        Matrix<float, OUTPUT_SIZE, 1> output_neuron;

        Matrix<float, OUTPUT_SIZE, 1> output_bias;
        Matrix<float, HIDDEN_SIZE, 1> hidden_biases;

        Matrix<float, HIDDEN_SIZE, INPUT_SIZE, Arrangement::ColMajor>   hidden_weights;
        Matrix<float, OUTPUT_SIZE, HIDDEN_SIZE>  output_weights;

        Matrix<Parameter, OUTPUT_SIZE, 1> output_bias_gradient;
        Matrix<Parameter, HIDDEN_SIZE, 1> hidden_bias_gradients;

        Matrix<Parameter, HIDDEN_SIZE, INPUT_SIZE, Arrangement::ColMajor>  hidden_weight_gradients;
        Matrix<Parameter, OUTPUT_SIZE, HIDDEN_SIZE> output_weight_gradients;
    };
}