#pragma once 
#include "sample.h"

#include <vector>
#include <string_view>

struct Sample;

struct Dataset
{   
    Dataset(std::string_view path, std::size_t limit);

    std::vector<Sample> training;
    std::vector<Sample> validation;
};