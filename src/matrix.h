#pragma once
#include <array>
#include <random>
#include <cassert>
#include <iostream>
#include <algorithm>

namespace Trainer
{
    template<typename T, int X, int Y, bool rowmajor = false> 
    class Matrix
    {
    public:
        Matrix() = default;

        T& get(int row, int col)
        {
            return rowmajor ? data[col * X + row] : data[row * Y + col];
        }

        T const& get(int row, int col) const
        {
            return rowmajor ? data[col * X + row] : data[row * Y + col];
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

    template<typename T, int X, int Y>
    using RowMajorMatrix = Matrix<T, X, Y, true>;

    template<typename T, int X, int Y>
    using ColMajorMatrix = Matrix<T, X, Y, false>;
}