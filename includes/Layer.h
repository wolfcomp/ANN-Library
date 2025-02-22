#pragma once

#include <vector>
#include <memory>

struct Neuron;
class ActivationFunction;

struct Layer
{
    std::vector<std::shared_ptr<Neuron>> neurons;
    std::shared_ptr<ActivationFunction> activationFunction;

    Layer() = default;
    Layer(std::vector<std::shared_ptr<Neuron>> neurons, std::shared_ptr<ActivationFunction> activation) : neurons(neurons), activationFunction(activation) {}
    ~Layer() = default;

    void addNeuron(std::shared_ptr<Neuron> neuron) { neurons.push_back(neuron); }
    std::vector<double> calculateOutput(const std::vector<double> &input);
};