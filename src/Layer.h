#pragma once

#include <vector>
#include <memory>
#include "ActivationFunction.h"

struct Neuron;

struct Layer
{
    std::vector<std::shared_ptr<Neuron>> neurons;

    Layer() = default;
    Layer(std::vector<std::shared_ptr<Neuron>> neurons) : neurons(neurons) {}
    ~Layer() = default;

    void addNeuron(std::shared_ptr<Neuron> neuron) { neurons.push_back(neuron); }
    void calculateN(std::vector<double> input);
    std::vector<double> calculateOutput(std::vector<double> input);
};