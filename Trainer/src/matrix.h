#pragma once
#include <vector>
#include <random>
#include <cassert>
#include <algorithm>

namespace Trainer
{

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
		Matrix for_each(Callable F) const
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
			std::random_device rd;
			std::mt19937 gen(43199807);
			std::uniform_real_distribution distrib(-1.0f, 1.0f);

			*this = for_each([&](T const&) { return distrib(gen); });
		}

		int size() const
		{
			return static_cast<int>(data.size());
		}
	private:
		std::vector<T> data;
		int rows, cols;
	};
}

inline Trainer::Matrix operator+(Trainer::Matrix const& lhs, Trainer::Matrix const& rhs)
{
	assert(lhs.totalRows() == rhs.totalRows() && lhs.totalCols() == rhs.totalCols());
	Trainer::Matrix out(lhs.totalRows(), lhs.totalCols());

	for (int i = 0; i < lhs.size(); i++)
		out.get(i) = lhs.get(i) + rhs.get(i);

	return out;
}

inline Trainer::Matrix operator-(Trainer::Matrix const& lhs, Trainer::Matrix const& rhs)
{
	assert(lhs.totalRows() == rhs.totalRows() && lhs.totalCols() == rhs.totalCols());
	Trainer::Matrix out(lhs.totalRows(), lhs.totalCols());

	for (int i = 0; i < lhs.size(); i++)
		out.get(i) = lhs.get(i) - rhs.get(i);

	return out;
}

inline Trainer::Matrix operator*(Trainer::Matrix const& lhs, Trainer::Matrix const& rhs)
{
	assert(lhs.totalCols() == rhs.totalRows());

	Trainer::Matrix out(lhs.totalRows(), rhs.totalCols());

	for (int i = 0; i < lhs.totalRows(); i++)
	{
		for (int j = 0; j < rhs.totalCols(); j++)
		{
			Trainer::Matrix::T sum = 0;
			for (int k = 0; k < lhs.totalCols(); k++)
				sum += lhs.get(i, k) * rhs.get(k, j);

			out.get(i, j) = sum;
		}
	}
	return out;
}

inline Trainer::Matrix operator*(Trainer::Matrix const& lhs, Trainer::Matrix::T scalar)
{
	return lhs.for_each([=](Trainer::Matrix::T x) { return x * scalar; });
}