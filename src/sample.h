#pragma once
#include "input.h"
#include "position.h"
#include "activation.h"

#include <string>
#include <vector>

inline std::string_view extract_fen(std::string_view line)
{
    return line.substr(0, line.find("[") - 1);
}

inline int extract_score(std::string_view line)
{
    return std::clamp(std::stoi(line.substr(line.find("]") + 1).data()), -2000, 2000);
}

inline float extract_wdl(std::string_view line)
{
    return  line.find("1.0") != line.npos ? 1.0f 
          : line.find("0.0") != line.npos ? 0.0f : 0.5f;
}

struct Sample
{
    Sample(std::string_view str)
        : input(Position(extract_fen(str)))
    {
        eval_target = sigmoid(extract_score(str));
        wdl_target  = extract_wdl(str);
    }   

    Input input;

    float wdl_target  = 0,
            eval_target = 0;
};