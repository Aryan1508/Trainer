#pragma once
#include <array>
#include <random>
#include <cassert>
#include <iostream>
#include <algorithm>

namespace Trainer
{
    template<typename T, int X, int Y> 
    class Matrix
    {
    public:
        Matrix() = default;

        T& get(int row, int col)
        {
            return data[row * Y + col];
        }

        T const& get(int row, int col) const
        {
            return data[row * Y + col];
        }

        T& get(int i)
        {
            return data[i];
        }

        T const& get(int i) const
        {
            return data[i];
        }

        void set(T const& val)
        {
            std::fill(data.begin(), data.end(), val);
        }

        void randomize(int inputs)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> distrib(0.0f, 2 / sqrtf((float)inputs));

            for (auto& val : data) val = distrib(gen);
        }

        T* raw()
        {
            return &data[0];
        }

        static constexpr int total_rows() 
        {
            return X;
        }

        static constexpr int total_cols() 
        {
            return Y;
        }

        static constexpr int size() 
        {
            return static_cast<int>(X * Y);
        }
    private:
        std::array<T, X * Y> data;
    };
}