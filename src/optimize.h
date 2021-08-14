#pragma once
#include <math.h>

namespace Trainer
{
	constexpr double BETA_1 = 0.9f;
	constexpr double BETA_2 = 0.999f;

	struct Parameter
	{
		float value = 0;
		double learning_rate = 0.01f;
		double gradient = 0;
		double M = 0, V = 0;

		Parameter() = default;
		
		void operator=(float v)
		{
			value = v;
		}

		void apply_gradient()
		{
			if (!gradient)
				return;

			M = M * BETA_1 + gradient * (1 - BETA_1);
			V = V * BETA_2 + (gradient * gradient) * (1 - BETA_2);

			value += -learning_rate * M / (sqrt(V) + 1e-8);
			gradient = 0;
		}
	};
}