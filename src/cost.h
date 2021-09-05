#pragma once
#include <vector>

struct Sample;
struct Network;

float calculate_cost(std::vector<Sample> const& samples, Network& network);

float calculate_cost(Sample const& sample, Network& network);