#pragma once
#include "types.h"
#include "high_low_byte.h"

#include <array>
#include <vector>
#include <string_view>

typedef std::vector<uint16_t> Features;

class Position 
{
public:
    Position(std::string_view fen);

    void set_fen(std::string_view fen);

    std::string get_fen() const;

    Features to_features() const;
private:

    std::array<HighLowByte, 16> piece_nibbles;
    uint64_t occupancy;
};