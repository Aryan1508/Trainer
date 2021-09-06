#pragma once
#include "input.h"
#include "matrix.h"
#include "activation.h"

void forward_propagate(Matrix<float> const& input, 
                        Matrix<float>      & output, 
                        Matrix<float> const& weights, 
                        Matrix<float> const& biases,
                        ActivationFunction   activation)
{
    for(int i = 0;i < output.size();i++)
    {
        output(i) = 0.0f;

        for(int j = 0;j < input.size();j++)
            output(i) += input(j) * weights(i, j);

        output(i) = activation(output(i) + biases(i));
    }
}

void forward_propagate(Input         const& input, 
                        Matrix<float>      & output, 
                        Matrix<float> const& weights, 
                        Matrix<float> const& biases,
                        ActivationFunction   activation)
{
    output.set(0.0f);

    for(auto input_index : input.indices)
    {   
        for(int i = 0;i < output.size();i++)
            output(i) += weights(i, input_index);
    }

    for(int i = 0;i < output.size();i++)
        output(i) = activation(output(i) + biases(i));
}