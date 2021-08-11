#pragma once
#include "matrix.h"
#include "position.h"

namespace Trainer
{
	void position_to_input(Position const& position, Matrix<768, 1>& mat, std::vector<int>& input_indices)
	{
		for (int i = 0; i < 12; i++)
		{
			for (int j = 0; j < 64; j++)
			{
				double val = test_bit(position.get_bb(Piece(i)), Square(j));
				int index = i * 64 + j;

				mat.get(index) = val;

				if (val)
					input_indices.push_back(index);
			}
		}
	}
}