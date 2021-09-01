#pragma once
#include <random>
#include <cassert>
#include <cstring>

namespace Trainer
{
    template<typename T> 
    class Matrix
    {
    public:
        Matrix() = default;

        explicit Matrix(int size) 
        {
            create(size, 1);
        }

        Matrix(int rows, int cols)
        {
            create(rows, cols);
        }

        Matrix(Matrix const& rhs) 
        {
            create(rhs.row_count, rhs.col_count);
            std::memcpy(data, rhs.data, sizeof(T) * rhs.size());
        }

        void operator=(Matrix const& rhs) 
        {
            assert(has_same_dimensions(rhs));
            std::memcpy(data, rhs.data, sizeof(T) * rhs.size());
        }

        void resize(int rows, int cols)
        {
            delete[] data;
            create(rows, cols);
        }

        ~Matrix() 
        {
            delete[] data;
        }

        void set_zero()
        {
            std::memset(data, 0, sizeof(T) * size());
        }

        T& operator()(const int index)
        {
            assert(index < size());
            return data[index];
        }

        T const& operator()(const int index) const
        {
            assert(index < size());
            return data[index];
        }

        T& operator()(const int row, const int col)
        {
            assert(row < row_count && col < col_count);
            return data[col * row_count + row];
            // return data[row * col_count + col];
        }

        T const& operator()(const int row, const int col) const
        {
            assert(row < row_count && col < col_count);
            return data[col * row_count + row];
            // return data[row * col_count + col];
        }

        T* raw()
        {
            return data;
        }

        int rows() const 
        {
            return row_count;
        }

        int cols() const 
        {
            return col_count;
        }

        int size() const
        {
            return row_count * col_count;
        }

        bool has_same_dimensions(Matrix const& rhs) const
        {
            return row_count == rhs.row_count && col_count == rhs.col_count;
        }

        void he_init(const int n)
        {
            const float g = 2.0f / sqrtf(static_cast<float>(n));

            std::random_device rd;
            std::normal_distribution<float> distrib(0.0f, g);
            std::mt19937 rng(12345);

            for(int i = 0;i < size();i++)
                this->operator()(i) = distrib(rng);
        }
    private:
        void create(int rows, int cols)
        {
            row_count = rows;
            col_count = cols;
            data = new T[row_count * col_count];
            he_init(768);
        }

        T* data = nullptr;
        int    row_count;
        int    col_count;
    };
}