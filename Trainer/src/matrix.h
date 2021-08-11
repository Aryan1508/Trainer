#pragma once
#include <vector>
#include <random>
#include <cassert>
#include <algorithm>

class Matrix
{
public:
	using T = double;

	Matrix(int rows, int cols)
		: rows(rows), cols(cols)
	{
		data.resize(rows * cols, T());
	}

	T& get(int row, int col)
	{
		return data[row * cols + col];
	}
	
	T const& get(int row, int col) const
	{
		return data[row * cols + col];
	}

	T& get(int i)
	{
		return data[i];
	}

	T const& get(int i) const
	{
		return data[i];
	}

	int totalRows() const
	{
		return rows;
	}

	int totalCols() const
	{
		return cols;
	}

	template<typename Callable>
	Matrix forEach(Callable F) const
	{
		Matrix out(rows, cols);
		
		for (int i = 0; i < data.size(); i++)
			out.get(i) = F(get(i));
		return out;
	}

	void set(T const& val)
	{
		std::fill(data.begin(), data.end(), val);
	}

	void randomize()
	{
		std::mt19937 gen(4312987);
		std::uniform_real_distribution distrib(-1.0f, 1.0f);
	
		*this = forEach([&](T const& x) { return distrib(gen); });
	}

	int size() const
	{
		return data.size();
	}
private:
	std::vector<T> data;
	int rows, cols;
};

inline Matrix operator+(Matrix const& lhs, Matrix const& rhs)
{
	assert(lhs.totalRows() == rhs.totalRows() && lhs.totalCols() == rhs.totalCols());
	Matrix out(lhs.totalRows(), lhs.totalCols());

	for (int i = 0; i < lhs.size(); i++)
		out.get(i) = lhs.get(i) + rhs.get(i);

	return out;
}

inline Matrix operator-(Matrix const& lhs, Matrix const& rhs)
{
	assert(lhs.totalRows() == rhs.totalRows() && lhs.totalCols() == rhs.totalCols());
	Matrix out(lhs.totalRows(), lhs.totalCols());

	for (int i = 0; i < lhs.size(); i++)
		out.get(i) = lhs.get(i) - rhs.get(i);

	return out;
}

inline Matrix operator*(Matrix const& lhs, Matrix const& rhs)
{
	assert(lhs.totalCols() == rhs.totalRows());

	Matrix out(lhs.totalRows(), rhs.totalCols());

	for (int i = 0; i < lhs.totalRows(); i++)
	{
		for (int j = 0; j < rhs.totalCols(); j++)
		{
			Matrix::T sum = 0;
			for (int k = 0; k < lhs.totalCols(); k++)
				sum += lhs.get(i, k) * rhs.get(k, j);

			out.get(i, j) = sum;
		}
	}
	return out;
}

inline Matrix operator*(Matrix const& lhs, Matrix::T scalar)
{
	return lhs.forEach([=](Matrix::T x) { return x * scalar; });
}