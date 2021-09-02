#pragma once
#include "input.h"

#include <string>
#include <vector>

class Position;

namespace Trainer
{
    struct Sample
    {
        // <fen> [<wdl>] <wpov score>
        Sample(std::string_view);

        Input input;

        float wdl_target  = 0,
              eval_target = 0;
    };

    std::vector<Sample> load_samples(std::string_view path, 
                                    std::size_t limit);
}