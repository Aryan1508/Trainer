#pragma once
#include "matrix.h"
#include "position.h"

namespace Trainer
{
	Matrix<768, 1> position_to_input(Position const& position)
	{
		Matrix<768, 1> mat;
			
		for (int i = 0; i < 12; i++)
		{
			for (int j = 0; j < 64; j++)
			{
				mat.get(i * 64 + j) = float(test_bit(position.get_bb(Piece(i)), Square(j)));
			}
		}
		return mat;
	}
}