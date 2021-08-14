#pragma once
#include "matrix.h"
#include "position.h"

namespace Trainer
{
    void position_to_input(Position const& position, Matrix<Parameter, 768, 1>& mat, std::vector<int>& input_indices)
    {
        input_indices.reserve(32);
        mat.set(Parameter());

        for (int j = 0; j < 64; j++)
        {
            Square sq = Square(j);
                
            if (position.get_piece(Square(j)) != Piece::Empty)
            {
                Piece p = position.get_piece(Square(j));
                int index = p * 64 + sq;

                mat.get(index).value = 1.0f;

                input_indices.push_back(index);
            }
        }
    }
}