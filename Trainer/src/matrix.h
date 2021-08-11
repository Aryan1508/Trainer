#pragma once
#include <vector>
#include <array>
#include <random>
#include <cassert>
#include <algorithm>

namespace Trainer
{
	template<int X, int Y> 
	class Matrix
	{
	public:
		using T = double;

		Matrix()
		{
			data.resize(X * Y);
		}

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

		constexpr int totalRows() const
		{
			return X;
		}

		constexpr int totalCols() const
		{
			return Y;
		}

		template<typename Callable>
		Matrix<X, Y> for_each(Callable F) const
		{
			Matrix<X, Y> out;

			for (int i = 0; i < data.size(); i++)
				out.get(i) = F(get(i));
			return out;
		}

		Matrix<Y, X> transpose() const
		{
			Matrix<Y, X> out;

			for (int i = 0; i < X; i++)
			{
				for (int j = 0; j < Y; j++)
					out.get(j, i) = get(i, j);
			}
			return out;
		}

		void set(T const& val)
		{
			std::fill(data.begin(), data.end(), val);
		}

		void randomize()
		{
			std::random_device rd;
			std::mt19937 gen(431);
			std::uniform_real_distribution distrib(0.0f, 1.0f);

			*this = for_each([&](T const&) { return distrib(gen); });
		}

		int size() const
		{
			return static_cast<int>(data.size());
		}
	private:
		std::vector<double> data;
	};
}

template<int X1, int Y1> 
inline Trainer::Matrix<X1, Y1> operator+(Trainer::Matrix<X1, Y1> const& lhs, Trainer::Matrix<X1, Y1> const& rhs)
{
	assert(lhs.totalRows() == rhs.totalRows() && lhs.totalCols() == rhs.totalCols());
	Trainer::Matrix<X1, Y1> out;

	for (int i = 0; i < lhs.size(); i++)
		out.get(i) = lhs.get(i) + rhs.get(i);

	return out;
}

template<int X1, int Y1>
inline Trainer::Matrix<X1, Y1> operator-(Trainer::Matrix<X1, Y1> const& lhs, Trainer::Matrix<X1, Y1> const& rhs)
{
	Trainer::Matrix<X1, Y1> out;

	for (int i = 0; i < lhs.size(); i++)
		out.get(i) = lhs.get(i) - rhs.get(i);

	return out;
}

template<int X1, int Y1, int X2, int Y2>
inline Trainer::Matrix<X1, Y2> operator*(Trainer::Matrix<X1, Y1> const& lhs, Trainer::Matrix<X2, Y2> const& rhs)
{
	assert(lhs.totalCols() == rhs.totalRows());

	Trainer::Matrix<X1, Y2> out;

	for (int i = 0; i < lhs.totalRows(); i++)
	{
		for (int j = 0; j < rhs.totalCols(); j++)
		{
			double sum = 0;
			for (int k = 0; k < lhs.totalCols(); k++)
				sum += lhs.get(i, k) * rhs.get(k, j);

			out.get(i, j) = sum;
		}
	}
	return out;
}

template<int X1, int Y1>
inline Trainer::Matrix<X1, Y1> operator*(Trainer::Matrix<X1, Y1> const& lhs, double scalar)
{
	return lhs.for_each([=](double x) { return x * scalar; });
}