#pragma once
#include "matrix.h"
#include "position.h"

namespace Trainer
{
	void position_to_input(Position const& position, Matrix<768, 1>& mat, std::vector<int>& input_indices)
	{
		input_indices.reserve(32);
		for (int i = 0; i < 12; i++)
		{
			for (int j = 0; j < 64; j++)
			{
				float val = position.get_piece(Square(j)) == Piece(i);
				int index = i * 64 + j;

				mat.get(index) = val;

				if (val)
					input_indices.push_back(index);
			}
		}
	}
}