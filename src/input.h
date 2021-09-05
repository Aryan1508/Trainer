#pragma once 
#include <vector>
#include <cstdint>

class Position;
    
struct Input
{
    Input(Position const&);
    
    std::vector<std::uint16_t> indices;
};