#pragma once 
#include "threaddata.h"

#include <vector>
#include <string_view>

struct Trainer
{
    Trainer(std::vector<int> const& topology, std::string_view dataset_path, const std::size_t sample_size, const int n_threads);

    Dataset dataset;
    Network network;
    std::vector<ThreadData> thread_data;  
    const int n_threads;
};

void train_network(Trainer& trainer, std::string_view output_path, const int n_epochs);