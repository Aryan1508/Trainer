#pragma once
#include "position.h"
#include "activation.h"

#include <string>
#include <vector>
#include <fstream>

namespace Trainer
{
    struct NetworkInput
    {
        std::vector<uint16_t> activated_input_indices;
        float target = 0;
        float eval_target = 0;
    };

    inline NetworkInput position_to_input(Position const& position)
    {
        NetworkInput input;

        input.activated_input_indices.reserve(32);
        for (int j = 0; j < 64; j++)
        {
            Square sq = Square(j);
                
            if (position.get_piece(Square(j)) != Piece::Empty)
            {
                Piece p = position.get_piece(Square(j));
                int index = p * 64 + sq;
                input.activated_input_indices.push_back(index);
            }
        }
        input.activated_input_indices.shrink_to_fit();
        
        return input;
    }

    inline std::vector<NetworkInput> load_inputs(std::string_view file, std::size_t limit = 0)
    {
        Position position;
        std::vector<NetworkInput> inputs;
        inputs.reserve(limit);
        
        std::ifstream fil(file.data());

        if (!fil)
        {
            std::cerr << "Couldn't open " << file << std::endl;
            std::terminate();
        }

        for (std::string line; std::getline(fil, line);)
        {
            if (limit && inputs.size() >= limit)
                break;

            position.set_fen(line.substr(0, line.find("[") - 1));
            
            NetworkInput input = position_to_input(position);

            if (line.find("1.0") != line.npos)        input.target = 1.0;
            else if (line.find("0.0") != line.npos)   input.target = 0.0;
            else if (line.find("0.5") != line.npos)   input.target = 0.5;
            else break;

            int score = std::stoi(line.substr(line.find("]") + 1));

            if (std::abs(score) >= 30000)
                continue;

            score = std::clamp(score, -2000, 2000);

            input.eval_target = sigmoid(score);
            inputs.push_back(input);

            if (inputs.size() % (16384 * 2) == 0)
            {
                std::cout << "\rLoading inputs [" << inputs.size() << "]" << std::flush;
            }
        }
        fil.close();
        inputs.shrink_to_fit();

        std::cout << "\rLoading inputs [" << inputs.size() << "]" << std::flush;
        std::cout << std::endl;

        return inputs;
    }
}