#include "sample.h"
#include "position.h"
#include "activation.h"
#include <fstream>
#include <iomanip>

static std::string_view extract_fen(std::string_view line)
{
    return line.substr(0, line.find("[") - 1);
}

static int extract_score(std::string_view line)
{
    return std::clamp(std::stoi(line.substr(line.find("]") + 1).data()), -2000, 2000);
}

static float extract_wdl(std::string_view line)
{
    return  line.find("1.0") != line.npos ? 1.0f 
          : line.find("0.0") != line.npos ? 0.0f : 0.5f;
}

namespace Trainer
{       
    Sample::Sample(std::string_view str)
        : input(Position(extract_fen(str)))
    {
        eval_target = sigmoid(extract_score(str));
        wdl_target  = extract_wdl(str);
    }   

    std::vector<Sample> load_samples(std::string_view path, 
                                    std::size_t limit)
    {
        std::ifstream file(path.data());

        if (!file)
            throw std::invalid_argument("Cannot open " + std::string(path));

        std::vector<Sample> samples;

        for(std::string line; std::getline(file, line);)
        {   
            samples.push_back(Sample(line));

            if (samples.size() >= limit)
                break;

            if (samples.size() % 32768 == 0)
            {
                const float p = samples.size() / static_cast<float>(limit) * 100.0f;

                std::cout << std::setprecision(2);
                std::cout << "\rLoading samples " << p << '%' << std::flush;
            }
        }
        std::cout << '\r' << std::string(' ', 32);
        std::cout << "\rLoaded all samples (" << samples.size() << ')' << std::endl;

        return samples;
    }
}
