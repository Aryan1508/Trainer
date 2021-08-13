#pragma once

namespace Trainer
{
	struct Parameter
	{
		float value = 0;
		float delta = 0;
		float learning_rate = 0.1f;
		float decay_rate = 0.8f;
		float gradient = 0;

		Parameter() = default;
		
		void operator=(float v)
		{
			value = v;
		}

		void apply_gradient()
		{
			delta = -learning_rate * gradient + delta * decay_rate;
			value += delta;
			gradient = 0;
		}
	};
}