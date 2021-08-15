#pragma once
#include "matrix.h"
#include "position.h"

namespace Trainer
{
    typedef std::vector<int> NetworkInput;

    inline NetworkInput position_to_input(Position const& position)
    {
        NetworkInput indices;
        indices.reserve(32);

        for (int j = 0; j < 64; j++)
        {
            Square sq = Square(j);
                
            if (position.get_piece(Square(j)) != Piece::Empty)
            {
                Piece p = position.get_piece(Square(j));
                int index = p * 64 + sq;
                indices.push_back(index);
            }
        }
        return indices;
    }
}