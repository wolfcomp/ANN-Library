#pragma once
#include <vector>
#include "export.h"

struct DllExport Neuron
{
    double bias = 0.0;
    double output = 0.0;
    double N = 0.0;
    double errorGradient = 0.0;
    std::vector<double> weights = {};

    Neuron() {}
    Neuron(double bias, std::vector<double> weights) : bias(bias), weights(weights) {}

    void calculateN(const std::vector<double> &input);
};