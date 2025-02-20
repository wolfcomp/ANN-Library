#pragma once
#include <vector>
#include "ActivationFunction.h"

struct Neuron
{
    double bias;
    double output;
    double errorGradient;
    double N;
    std::vector<double> weights;

    Neuron() {}
    Neuron(double bias, std::vector<double> weights) : bias(bias), weights(weights) {}

    void calculateN(std::vector<double> input);
};