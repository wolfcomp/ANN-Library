#pragma once
#include <vector>

struct Neuron
{
    double bias;
    double output;
    double N;
    double errorGradient;
    std::vector<double> weights;

    Neuron();
    Neuron(double bias, std::vector<double> weights);

    void calculateN(const std::vector<double> &input);
};