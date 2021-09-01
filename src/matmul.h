#pragma once
#include "matrix.h"
#include "activation.h"

namespace Trainer
{
    class Input;

    void forward_propagate(Matrix<float> const& input, 
                           Matrix<float>      & output, 
                           Matrix<float> const& weights, 
                           Matrix<float> const& biases,
                           ActivationFunction   activation);

    void forward_propagate(Input         const& input, 
                           Matrix<float>      & output, 
                           Matrix<float> const& weights, 
                           Matrix<float> const& biases,
                           ActivationFunction   activation);
}