#include <cctype>

#include "position.h"

static void set_bit(uint64_t& bb, const Square sq) {
    assert(sq < SQ_TOTAL);
    bb |= (1ull << sq);
}

static Square get_lsb(uint64_t bb) 
{
    assert(bb);
    return static_cast<Square>(__builtin_ctzll(bb));
}

static Square pop_lsb(uint64_t &bb) 
{
    assert(bb);
    const Square index = get_lsb(bb);
    bb &= (bb - 1);
    return index;
}

Position::Position(std::string_view fen)
{
    set_fen(fen);
}

void Position::set_fen(std::string_view fen) 
{
    occupancy = 0;
    piece_nibbles.fill(HighLowByte());
    
    Square sq = SQ_A8;
    int n = 0;

    for(const char c : fen) 
    {
        if (std::isalpha(c)) 
        {
            Piece pce = char_to_piece(c);
            set_bit(occupancy, sq);

            if (n % 2 == 0)  
                piece_nibbles[n / 2].set_low_nibble(pce);
            else 
                piece_nibbles[n / 2].set_high_nibble(pce);

            n++;
            sq++;
        }
        else if (std::isdigit(c))  {
            sq += c - '0';
        }
        else if (c == '/')  
            continue;
        else if (c == ' ')
            break;
        else 
            throw std::invalid_argument(std::string("invalid fen: ").append(fen));
    }
}

Features Position::to_features() const 
{
    Features features;
    uint64_t b = occupancy;
    int n = 0;
    while(b)  
    {
        Square lsb = pop_lsb(b);
        Piece  pce = n % 2 == 0 ? static_cast<Piece>(piece_nibbles[n / 2].get_low_nibble()) 
                                : static_cast<Piece>(piece_nibbles[n / 2].get_high_nibble());

        features.push_back(pce * 64 + flip_square(lsb));
        n++;
    }
    return features;
}