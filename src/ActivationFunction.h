#pragma once

#include <cmath>
#include <vector>
#include "export.h"

/// @brief Base class for activation functions and their derivatives
/// @details This class is implemented with Identity as the default
class DllExport ActivationFunction
{
public:
    virtual ~ActivationFunction() = default;
    virtual double operator()(double x) { return x; };
    virtual double derivative(double x) { return 1; };
};

class DllExport Relu final : public ActivationFunction
{
public:
    double operator()(double x) override { return x >= 0 ? x : 0; };
    double derivative(double x) override { return x >= 0 ? 1 : 0; };
};

class DllExport Sigmoid final : public ActivationFunction
{
public:
    double operator()(double x) override { return 1 / (1 + exp(-x)); };
    double derivative(double x) override
    {
        double g = this->operator()(x);
        return g / (1 - g);
    };
};

class DllExport Tanh final : public ActivationFunction
{
public:
    double operator()(double x) override { return 2.0 / (1 + exp(-2 * x)) - 1; };
    double derivative(double x) override { return 1 - pow(this->operator()(x), 2); };
};

class DllExport LeakyRelu final : public ActivationFunction
{
public:
    double operator()(double x) override { return x >= 0 ? x : 0.01 * x; };
    double derivative(double x) override { return x >= 0 ? 1 : 0.01; };
};
