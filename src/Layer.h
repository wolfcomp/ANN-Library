#pragma once

#include <vector>
#include <memory>
#include "export.h"

struct Neuron;
class ActivationFunction;

struct DllExport Layer
{
    std::vector<std::shared_ptr<Neuron>> neurons;
    std::shared_ptr<ActivationFunction> activationFunction;

    Layer() = default;
    Layer(std::vector<std::shared_ptr<Neuron>> neurons, std::shared_ptr<ActivationFunction> activation) : neurons(neurons), activationFunction(activation) {}
    ~Layer() = default;

    void addNeuron(std::shared_ptr<Neuron> neuron) { neurons.push_back(neuron); }
    void calculateNeuron(const std::vector<double> &input);
    std::vector<double> calculateOutput(const std::vector<double> &input);
    std::vector<double> calculatedGradients();
};