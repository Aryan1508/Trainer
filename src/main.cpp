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
    -------------------------------------------------------------------------------------------------------------
    |  Option Name   |          Description                  |                   DEFAULT                        |
    |----------------+---------------------------------------+--------------------------------------------------+-
    |  -samples      |    number of samples to read          |  no default, must be provided                    |                      
    |  -dataset      |    path to dataset                    |  no default, must be provided                    |   
    |  -threads      |    number of threads to use           |  1                                               |    
    |  -base         |    path to base network file (.tnn)   |  none (start from randomly initialized network)  |    
    |  -out          |    path to store trained network file | 'trained'                                        |    
    |  -epochs       |    number of epochs to run            |  100                                             |
    |  -batch        |    size of one batch                  |  16384                                           |
    |  -lr           |    initial earning rate               |  default 0.01)                                   |    
    |  -lrdrop       |    number of epochs to drop lr after  |  none (no drop)                                  |
    |  -lrdrop_rate  |    factor by which lr is dropped      |  0.1  (only applicable if -lrdrop is provided)   |

                                USAGE 
         ./Trainer [OPTION NAME] [OPTION VALUE]  
         ./Trainer -threads 8 -dataset C:/datasets/dataset.txt ...
)~";

int main(int argc, char** argv)
{
    CommandLineParser cmdline(argc, argv);

    std::cout << HELP_INFO << std::endl;

    const std::vector<int> topology{768, 256, 1};
    const auto dataset_path = cmdline.get_soption("-dataset", "");
    const auto output_path  = cmdline.get_soption("-out", "trained");
    const auto n_threads    = cmdline.get_ioption("-threads", 1);
    const auto samples      = cmdline.get_ulloption("-samples", 0);
    const auto base_net     = cmdline.get_soption("-base", "");
    const auto epochs       = cmdline.get_ioption("-epochs", 100);
    LR                      = cmdline.get_foption("-lr", 0.01f);
    BATCH_SIZE              = cmdline.get_ioption("-batch", 16384);
    LR_DROP_EPOCHS          = cmdline.get_ioption("-lrdrop", -1);

    if (LR_DROP_EPOCHS != -1) {
        SHOULD_LR_DROP = true;
        LR_DROP_RATE   = cmdline.get_foption("-lrdrop_rate", 0.1f);
    }

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

    if (base_net != "")
        load_network(trainer.network, trainer.thread_data[0].gradients, base_net);

    train_network(trainer, output_path, epochs);
}