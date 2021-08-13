#pragma once
#include <vector>
#include <random>
#include <iomanip>
#include <iostream>

#include "matrix.h"
#include "activation.h"

namespace Trainer
{
    constexpr int INPUT_SIZE  = 768;
    constexpr int HIDDEN_SIZE = 256;

	template<int X>
	using ColVector = Matrix<X, 1>;

	using InputVector  = ColVector<INPUT_SIZE>;
	using HiddenVector = ColVector<HIDDEN_SIZE>;
	using OutputVector = ColVector<1>;

	class Network
	{
	public:
		Network();

		void save_network(std::string_view fil);
		
		void load_network(std::string_view fil);

		//void apply_gradients();
		
		void feed(std::vector<int> const& input_indices);

		void back_propagate(InputVector const& sample, std::vector<int> const& input_indices, float target)
		{
			feed(input_indices);
			calculate_errors(sample, target, input_indices);
		}

		float get_output() const
		{
			return output_neuron.get(0);
		}
	private:
		void calculate_errors(InputVector const& sample, float target, std::vector<int> const& indices);
	private:
		HiddenVector  hidden_biases;
		HiddenVector  hidden_neurons;
		HiddenVector  hidden_bias_deltas;

		OutputVector output_bias;
		OutputVector output_neuron;
		OutputVector output_bias_deltas;

		ColMajorMatrix<HIDDEN_SIZE, INPUT_SIZE>   hidden_weights;
		ColMajorMatrix<  1        , HIDDEN_SIZE>  output_weights;
		ColMajorMatrix<HIDDEN_SIZE, INPUT_SIZE>   hidden_weight_deltas;
		ColMajorMatrix<  1        , HIDDEN_SIZE>  output_weight_deltas;
	};
}