EXE := Trainer
SRC := *.cpp
CXX := g++

LFLAGS   := -lpthread -fopenmp
WFLAGS   := -Wall -Wextra
CXXFLAGS := -std=c++17 $(WFLAGS) -O3 -DNDEBUG -g -march=native 

default:
	$(CXX) $(CXXFLAGS) $(SRC) $(LFLAGS) -o $(EXE)