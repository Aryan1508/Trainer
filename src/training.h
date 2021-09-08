#pragma once 
#include <vector>

struct Network;
struct Dataset;
struct Gradients;

void train_network(Network&, Dataset const&, Gradients& gradients);