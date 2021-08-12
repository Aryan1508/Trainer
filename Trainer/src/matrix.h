#pragma once
#include <array>
#include <random>
#include <cassert>
#include <iostream>
#include <algorithm>

namespace Trainer
{
	template<int X, int Y, bool rowmajor = false> 
	class Matrix
	{
	public:
		Matrix() = default;

		float& get(int row, int col)
		{
			return rowmajor ? data[col * X + row] : data[row * Y + col];
		}

		float get(int row, int col) const
		{
			return rowmajor ? data[col * X + row] : data[row * Y + col];
		}

		float& get(int i)
		{
			return data[i];
		}

		float get(int i) const
		{
			return data[i];
		}

		void set(float val)
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
		std::array<float, X * Y> data;
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

	template<int X, int Y>
	using RowMajorMatrix = Matrix<X, Y, true>;

	template<int X, int Y>
	using ColMajorMatrix = Matrix<X, Y, false>;
}