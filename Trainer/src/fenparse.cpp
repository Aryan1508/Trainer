/*
  Bit-Genie is an open-source, UCI-compliant chess engine written by
  Aryan Parekh - https://github.com/Aryan1508/Bit-Genie

  Bit-Genie is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Bit-Genie is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "position.h"
#include "Square.h"
#include <sstream>
#include <vector>

namespace
{
    Piece get_piece(char label)
    {
        char ptl = std::tolower(label);
        PieceType pt = ptl == 'p' ? PieceType::Pawn   :
                       ptl == 'n' ? PieceType::Knight :
                       ptl == 'b' ? PieceType::Bishop :
                       ptl == 'r' ? PieceType::Rook   :
                       ptl == 'q' ? PieceType::Queen  : PieceType::King;
        return std::islower(label) ? make_piece(pt, Color::Black) : make_piece(pt, Color::White);
    }

    std::vector<std::string> split_string(std::string_view str, char delim = ' ')
    {
        std::vector<std::string> parts;
        std::stringstream stream(str.data());
        std::string temp;

        while (std::getline(stream, temp, delim))
        {
            parts.push_back(std::move(temp));
        }
        return parts;
    }

}

std::string Position::get_fen() const 
{
    std::stringstream s;
    int empty = 0;

    auto print_empty = 
    [&]()
    {
        if (empty)
        {
            s << empty;
        }
        empty = 0;
    };

    for (Square sq = Square::A1;sq <= Square::H8;sq++)
    {
        if (sq != Square::A1 && sq % 8 == 0)
        {
            print_empty();
            s << '/';
        }

        Square idx  = flip_square(sq);
        Piece piece = get_piece(idx); 

        if (piece != Piece::Empty)
        {
            print_empty();
            s << piece;
        }
        else    
            empty++;
    }
    print_empty();
    return s.str();
}

void Position::set_fen(std::string_view fen)
{
    pieces.fill(Empty);

    auto parts = split_string(fen);

    Square counter = Square::A1;
    for(std::string_view s_rank : split_string(parts[0], '/'))
    {
        for(char p : s_rank)
        {
            if (std::isdigit(p))    
                counter += p - '0';
            else 
            {
                add_piece(flip_square(counter), ::get_piece(p));
                counter++;
            }
        }
    }

    side = parts[1] == "w";
}