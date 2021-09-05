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

Sample::Sample(std::string_view str)
    : input(Position(extract_fen(str)))
{
    eval_target = sigmoid(extract_score(str));
    wdl_target  = extract_wdl(str);
}   