#pragma once
#include <vector>
#include "export.h"

struct DllExport Neuron
{
    double bias;
    double output;
    double N;
    double dN;
    std::vector<double> weights;

    Neuron() {}
    Neuron(double bias, std::vector<double> weights) : bias(bias), weights(weights) {}

    void calculateN(const std::vector<double> &input);
};