#pragma once
#include <math.h>
#include <algorithm>

namespace Trainer
{
	inline float sigmoid(float x)
	{
		return 1.0f / (1.0f + expf(-x * (2.5f / 400.0f)));
	}

	inline float sigmoid_prime(float x)
	{
		return x * (1.0f - x) * (2.5f / 400.0f);
	}

	inline float relu(float x)
	{
		return std::max<float>(x, 0.0f);
	}
}