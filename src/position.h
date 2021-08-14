#pragma once
#include "piece.h"

#include <array>
#include <string_view>

class Position
{
public:
    std::array<Piece, 64> pieces; 
    float result = 0;
    bool side = false;
    
    Position();

    void set_fen(std::string_view);

    std::string get_fen() const;

    void add_piece(Square sq, Piece piece)
    {
        get_piece(sq) = piece;
    }

    void remove_piece(Square sq)
    {
        get_piece(sq) = Piece::Empty;
    }

    Piece& get_piece(Square sq) 
    {
        return pieces[sq];
    }

    Piece get_piece(Square sq) const 
    {
        return pieces[sq];
    }

    friend std::ostream& operator<<(std::ostream&, Position const&);
};