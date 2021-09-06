#include "net.h"
#include "cost.h"
#include "dataset.h"
#include "training.h"
#include "optimize.h"

#include <iomanip>
#include <sstream>
#include <iostream>

int main()
{
    const std::vector<int> topology{768, 128, 1};

    Dataset   dataset("C:/tuning/Datasets/8b5ed5872e.txt", 200000);
    Network   network(topology);
    Gradients gradients(topology);

    train_network(network, dataset, gradients);
}