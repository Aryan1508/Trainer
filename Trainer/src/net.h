#pragma once
#include <vector>
#include <random>
#include <iomanip>
#include <iostream>

#include "matrix.h"
#include "activation.h"

namespace Trainer
{
	template<int X>
	using ColVector = Matrix<X, 1>;
	
	using InputVector  = ColVector<768>;
	using HiddenVector = ColVector<512>;
	using OutputVector = ColVector<1>;

	class Network
	{
	public:
		Network();

		void apply_gradients();
		void feed(std::vector<int> const& input_indices);

		void back_propagate(InputVector const& sample, std::vector<int> const& input_indices, float target)
		{
			feed(input_indices);
			calculate_errors(target);
			calculate_gradients(sample);
		}

		float get_output() const
		{
			return output_neuron.get(0);
		}
	private:
		void calculate_errors(float target);

		void calculate_gradients(InputVector const& sample);

	private:
		HiddenVector  hidden_errors;
		HiddenVector  hidden_biases;
		HiddenVector  hidden_neurons;
		HiddenVector  hidden_bias_deltas;

		OutputVector output_bias;
		OutputVector output_error;
		OutputVector output_neuron;
		OutputVector output_bias_deltas;

		RowMajorMatrix<512, 768>  hidden_weights;
		ColMajorMatrix<  1, 512>  output_weights;
		ColMajorMatrix<512, 768>  hidden_weight_deltas;
		ColMajorMatrix<  1, 512>  output_weight_deltas;
	};
}