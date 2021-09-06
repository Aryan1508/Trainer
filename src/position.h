#pragma once
#include "piece.h"
#include "Square.h"

#include <array>
#include <string_view>

class Position
{
public:
    std::array<Piece, 64> pieces;
    
    Position()
    {
        set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    }

    Position(std::string_view fen)
    {
        set_fen(fen);
    }

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

inline std::ostream& operator<<(std::ostream& o, Position const& position)
{
    return o << position.get_fen();
}