#pragma once
#include "position.h"

#include <string>
#include <vector>
#include <fstream>

namespace Trainer
{
	inline std::vector<Position> load_positions(std::string_view file, int limit = 0)
	{
		Position position;
		std::vector<Position> positions;

		std::ifstream fil(file.data());

		if (!fil)
		{
			std::cerr << "Couldn't open " << file << std::endl;
			std::terminate();
		}

		for (std::string line; std::getline(fil, line);)
		{
			if (limit && positions.size() >= limit)
				break;

			position.set_fen(line.substr(0, line.find("[") - 1));

			if (line.find("[1.0]") != line.npos)        position.result = 1.0;
			else if (line.find("[0.0]") != line.npos)   position.result = 0.0;
			else                                        position.result = 0.5;

			positions.push_back(position);

			if (positions.size() % 4096 == 0)
				std::cerr << "\rLoading positions [" << positions.size() << "]" << std::flush;
		}
		fil.close();
		std::cout << std::endl;

		return positions;
	}
}