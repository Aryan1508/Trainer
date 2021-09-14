#include "net.h"
#include "cost.h"
#include "dataset.h"
#include "cmdline.h"
#include "training.h"
#include "optimize.h"

#include <iomanip>
#include <sstream>
#include <iostream>

const std::string HELP_INFO = R"~(
    ./Trainer -threads <number of threads> (default 1)
              -dataset <path to dataset> 
              -samples <number of samples to use> 
              -out     <path to store trained network file> (default trained.nn)
)~";

int main(int argc, char** argv)
{
    CommandLineParser cmdline(argc, argv);

    std::cout << HELP_INFO << std::endl;

    const std::vector<int> topology{768, 128, 1};
    const auto dataset_path = cmdline.get_soption("-dataset", "");
    const auto output_path  = cmdline.get_soption("-out", "trained.nn");
    const auto n_threads    = cmdline.get_ioption("-threads", 1);
    const auto samples      = cmdline.get_ulloption("-samples", 0);

    if (!dataset_path.size())
    {
        std::cerr << "Provide dataset path using option -dataset\n";
        std::cerr << "./Trainer -dataset <path to dataset>" << std::endl;
        return -1;
    }

    if (!samples)
    {
        std::cerr << "Provide number of samples to be used using option -samples\n";
        std::cerr << "./Trainer -samples <number of samples>" << std::endl;
        return -1;
    }

    Trainer trainer(topology, dataset_path, samples, n_threads);
    train_network(trainer, output_path);
}