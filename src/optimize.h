#pragma once
#include <math.h>

namespace Trainer
{
    constexpr double BETA_1 = 0.9f;
    constexpr double BETA_2 = 0.999f;

    class Parameter
    {
    private:
        float value = 0;
        double learning_rate = 0.01f;
        double gradient = 0;
        double M = 0, V = 0;

    public:
        Parameter() = default;

        explicit Parameter(float value) 
            : value(value)
        {}

        void operator=(float v)
        {
            value = v;
        }

        float get_value() const
        {
            return value;
        }

        void update_gradient(float delta)
        {
            gradient += delta;
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

    inline bool operator>(Parameter const& lhs, Parameter const& rhs)  
    {
        return lhs.get_value() > rhs.get_value();
    }

    inline float& operator+=(float& value, Parameter const& param)
    {
        value += param.get_value();
        return value;
    }

    inline float operator+(float value, Parameter const& param)
    {
        return value + param.get_value();
    }

    inline float operator*(float value, Parameter const& param)
    {
        return value * param.get_value();
    }
}