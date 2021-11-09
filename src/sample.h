#pragma once
#include "input.h"
#include "position.h"
#include "activation.h"

#include <string>
#include <vector>

constexpr uint8_t WDL_WIN = 2;
constexpr uint8_t WDL_DRAW = 1;
constexpr uint8_t WDL_LOSS = 0;

inline std::string_view extract_fen(std::string_view line)
{
    return line.substr(0, line.find("[") - 1);
}

inline int extract_score(std::string_view line)
{
    return std::stoi(line.substr(line.find("]") + 1).data());
}

inline uint8_t extract_wdl(std::string_view line)
{
    return  line.find("1.0") != line.npos ? WDL_WIN 
          : line.find("0.0") != line.npos ? WDL_LOSS : WDL_DRAW;
}

inline float calculate_wdl_target(uint8_t wdl_value) 
{
    return wdl_value == WDL_WIN ? 1.0f : wdl_value == WDL_LOSS ? 0.0f : 0.5f;
}

struct Sample
{
    Sample(std::string_view str)
        : input(Position(extract_fen(str)))
    {
        eval_target = sigmoid(extract_score(str));
        wdl_value  = extract_wdl(str);
    }   

    Input input;

    float eval_target = 0;
    uint8_t wdl_value = 0;
};