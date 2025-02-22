#pragma once

/// @brief Base class for activation functions and their derivatives
/// @details This class is implemented with Identity as the default
class ActivationFunction
{
public:
    virtual ~ActivationFunction() = default;
    virtual double operator()(double x);
    virtual double derivative(double x);
};

class Relu final : public ActivationFunction
{
};

class Sigmoid final : public ActivationFunction
{
};

class Tanh final : public ActivationFunction
{
};

class LeakyRelu final : public ActivationFunction
{
};
