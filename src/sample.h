#pragma once
#include "input.h"

#include <string>
#include <vector>

class Position;

struct Sample
{
    // <fen> [<wdl>] <wpov score>
    Sample(std::string_view);

    Input input;

    float wdl_target  = 0,
            eval_target = 0;
};