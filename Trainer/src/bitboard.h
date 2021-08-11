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
#pragma once
#include "bitmask.h"
#include "Square.h"

template <Direction dir>
constexpr uint64_t shift(uint64_t bits) noexcept
{
    return dir == Direction::north   ? bits << 8
           : dir == Direction::south ? bits >> 8
           : dir == Direction::east  ? (bits << 1) & BitMask::not_file_a
                                     : (bits >> 1) & BitMask::not_file_h;
}

constexpr uint64_t shift(uint64_t bits, Direction dir) noexcept
{
    return dir == Direction::north   ? bits << 8
           : dir == Direction::south ? bits >> 8
           : dir == Direction::east  ? (bits << 1) & BitMask::not_file_a
                                     : (bits >> 1) & BitMask::not_file_h;
}

inline Square get_lsb(uint64_t bb) noexcept
{
    return static_cast<Square>(/*__builtin_ctzll*/(bb));
}

inline int popcount64(uint64_t bb) noexcept
{
    return static_cast<int>(__popcnt64(bb));
}

inline Square pop_lsb(uint64_t &bb) noexcept
{
    Square index = get_lsb(bb);
    bb &= (bb - 1);
    return index;
}

constexpr bool test_bit(uint64_t bb, Square sq) noexcept
{
    return (1ull << sq) & bb;
}

constexpr void set_bit(uint64_t &bb, Square sq) noexcept
{
    bb |= (1ull << sq);
}

constexpr void flip_bit(uint64_t& bb, Square sq) noexcept
{
    bb ^= (1ull << sq);
}

constexpr bool is_several(uint64_t bb) noexcept
{
    return bb & (bb - 1);
}