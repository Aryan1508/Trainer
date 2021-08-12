#pragma once
#include <vector>
#include <array>
#include <random>
#include <cassert>
#include <iostream>
#include <algorithm>

namespace Trainer
{
	template<int X, int Y> 
	class Matrix
	{
	public:
		using T = float;

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

		constexpr int total_rows() const
		{
			return X;
		}

		constexpr int total_cols() const
		{
			return Y;
		}

		void set(T const& val)
		{
			std::fill(data.begin(), data.end(), val);
		}

		void randomize(int inputs)
		{
			float a = 2 / sqrtf((float)inputs);

			std::mt19937 gen(754232);
			std::normal_distribution distrib(0.0f, a);

			for (auto& val : data) val = distrib(gen);
		}

		int size() const
		{
			return static_cast<int>(data.size());
		}
	private:
		std::vector<float> data;
	};

	template<int X1, int X2>
	std::ostream& operator<<(std::ostream& o, Matrix<X1, X2> const& mat)
	{
		for (int i = 0; i < mat.total_rows(); i++)
		{
			for (int j = 0; j < mat.total_cols(); j++)
			{
				o << mat.get(i, j) << ' ';
			}
			o << '\n';
		}
		return o;
	}
}