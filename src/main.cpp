#include "net.h"
#include "cost.h"
#include "dataset.h"
#include "cmdline.h"
#include "training.h"
#include "optimize.h"

#include <iomanip>
#include <sstream>
#include <iostream>

int main(int argc, char** argv)
{
    CommandLineParser cmdline(argc, argv);

    const std::vector<int> topology{768, 128, 1};
    const int n_threads = cmdline.get_option("-threads", 1);

    Trainer trainer(topology, "C:/tuning/Datasets/8b5ed5872e.txt", n_threads);
    train_network(trainer);
}