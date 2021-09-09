#pragma once 
#include "threaddata.h"

#include <vector>
#include <string_view>

struct Trainer
{
    Trainer(std::vector<int> const& topology, std::string_view dataset_path, const int n_threads);

    Dataset dataset;
    std::vector<ThreadData> thread_data;  
};

void train_network(Trainer& trainer);