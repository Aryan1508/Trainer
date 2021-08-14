#include "position.h"

Position::Position()
{
    set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

std::ostream& operator<<(std::ostream& o, Position const& position)
{
    return o << position.get_fen();
}