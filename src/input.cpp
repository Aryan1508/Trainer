#include "input.h"
#include "position.h"

namespace Trainer
{
    Input::Input(Position const& position)
    {
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
    }
}