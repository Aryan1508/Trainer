#pragma once
#include <math.h>
#include <algorithm>

namespace Trainer
{
	inline float sigmoid(float x)
	{
		return 1.0f / (1.0f + expf(-x));
	}

	inline float sigmoid_prime(float x)
	{
		return x * (1.0f - x);
	}

	inline float relu(float x)
	{
		return std::max<float>(x, 0.0f);
	}
}