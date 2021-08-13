#pragma once

namespace Trainer
{
	struct Parameter
	{
		float value = 0;
		float delta = 0;
		float learning_rate = 0;
		float decay_rate = 0;

		Parameter()
		{
			learning_rate = 0.1f;
			decay_rate = 0.8f;
		}

		void operator=(float v)
		{
			value = v;
		}

		void apply_gradient(float& gradient)
		{
			delta = -learning_rate * gradient + delta * decay_rate;
			value += delta;
			gradient = 0;
		}
			

	};
}