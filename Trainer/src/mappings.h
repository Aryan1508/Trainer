#pragma once
#include "matrix.h"
#include "position.h"

namespace Trainer
{
	Matrix position_to_input(Position const& position)
	{
		Matrix mat(64 * 12, 1);
			
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