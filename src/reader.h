#pragma once
#include "position.h"

#include <string>
#include <vector>
#include <fstream>

namespace Trainer
{
    struct NetworkInput
    {
        std::vector<int> activated_input_indices;
        float target = 0;
    };

    inline NetworkInput position_to_input(Position const& position)
    {
        NetworkInput input;

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
        return input;
    }

    inline std::vector<NetworkInput> load_inputs(std::string_view file, std::size_t limit = 0)
    {
        Position position;
        std::vector<NetworkInput> inputs;

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

            if (line.find("[1.0]") != line.npos)        input.target = 1.0;
            else if (line.find("[0.0]") != line.npos)   input.target = 0.0;
            else                                        input.target = 0.5;

            inputs.push_back(input);

            if (inputs.size() % 4096 == 0)
                std::cerr << "\rLoading inputs [" << inputs.size() << "]" << std::flush;
        }
        fil.close();
        std::cout << std::endl;

        return inputs;
    }
}