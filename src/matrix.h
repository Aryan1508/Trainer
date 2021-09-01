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

        void set(T const& val)
        {
            std::fill(data, data + size(), val);
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

    private:
        void create(int rows, int cols)
        {
            row_count = rows;
            col_count = cols;
            data = new T[row_count * col_count];
        }

        T* data = nullptr;
        int    row_count;
        int    col_count;
    };
}