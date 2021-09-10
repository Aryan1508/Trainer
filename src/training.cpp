#include "cost.h"
#include "table.h"
#include "dataset.h"
#include "training.h"
#include "stopwatch.h"

#include <chrono>
#include <thread>
#include <sstream>
#include <iomanip>

namespace 
{
    constexpr int batch_size = 16384;
    constexpr int max_epochs = 256;

    std::string ftos(float value)
    {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(8) << value;
        return ss.str();
    }

    void print_cost(Table& table, Network& network, Dataset const& dataset,int epoch)
    {
        const float training_cost   = calculate_cost(dataset.training, network);
        const float validation_cost = calculate_cost(dataset.validation, network);

        std::vector<std::string> values;
        values.push_back(std::to_string(epoch));
        values.push_back(ftos(training_cost));
        values.push_back(ftos(validation_cost));

        table.print_value_row(values);
    }

    void sum_gradients(Matrix<Gradient>& base, Matrix<Gradient>& delta)
    {
        for(int i = 0;i < delta.size();i++)
        {
            update_gradient(base(i), delta(i).value);
            delta(i).value = 0;
        }
    }

    void sum_gradients(Gradients& base, Gradients& delta)
    {
        for(std::size_t layer = 0;layer < base.bias_gradients.size();layer++)
        {
            sum_gradients(base.bias_gradients[layer], delta.bias_gradients[layer]);
            sum_gradients(base.weight_gradients[layer], delta.weight_gradients[layer]);
        }
    }

    void sum_gradients(Trainer& trainer)
    {
        if (trainer.thread_data.size() == 1)
            return;
            
        Gradients& master_gradients = trainer.thread_data[0].gradients;

        for(std::size_t thread = 1;thread < trainer.thread_data.size();thread++)
        {
            Gradients& delta = trainer.thread_data[thread].gradients;
            sum_gradients(master_gradients, delta);
        }
    }

    void complete_local_batch(Dataset const& dataset, Network const& network, ThreadData& thread, const std::size_t start, const std::size_t size)
    {
        const std::size_t end = start + size;

        for(std::size_t i = start;i < end;i++)
        {
            calculate_gradients(dataset.training[i], network, thread.neurons, thread.gradients);
        }
    }

    void complete_batch(Trainer& trainer, const std::size_t start)
    {
        const std::size_t local_batch_size = batch_size / trainer.thread_data.size();

        std::vector<std::thread> threads;

        for(std::size_t i = 0;i < trainer.thread_data.size();i++)
            threads.emplace_back(complete_local_batch, 
                                 std::ref(trainer.dataset),
                                 std::ref(trainer.network),
                                 std::ref(trainer.thread_data[i]),
                                 start + (i * local_batch_size),
                                 local_batch_size);

        for(auto& thread : threads) thread.join();

        sum_gradients(trainer);
    }

    void apply_gradients(Trainer& trainer)
    {
        Gradients& gradients = trainer.thread_data[0].gradients;

        apply_gradients(trainer.network, gradients);
    }

    void complete_epoch(Trainer& trainer)
    {
        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);

        Dataset const& dataset  = trainer.dataset;

        StopWatch watch;
        watch.go();

        for(std::size_t i = 0;i < dataset.training.size();i += batch_size)
        {   
            const float evaluated_percent = i / static_cast<float>(dataset.training.size()) * 100;
            const float elapsed = watch.elapsed_time().count() / 1000.0f;
            const float speed   = i / elapsed;

            std::cout << "\rRunning epoch " << evaluated_percent << "% [" << speed << " /s]";            

            complete_batch(trainer, i);
            apply_gradients(trainer);
        }
        std::cout << '\r';
    }
}

void train_network(Trainer& trainer)
{
    Table table(std::cout, 32, {"Epoch", "Training", "Validation"});
    table.print_headers();

    for(int epoch = 1;epoch <= max_epochs;epoch++)
    {
        complete_epoch(trainer);
        print_cost(table, trainer.network, trainer.dataset, epoch);
    }
}

Trainer::Trainer(std::vector<int> const& topology, std::string_view dataset_path, const int n_threads)
    : dataset(dataset_path), network(topology), thread_data(n_threads, ThreadData(topology))
{
    if (n_threads <= 0)
        throw std::invalid_argument("Invalid thread count");
}