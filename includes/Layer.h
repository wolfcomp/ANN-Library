#pragma once

#include <vector>
#include <memory>

struct Neuron;
class ActivationFunction;

struct Layer
{
    std::vector<std::shared_ptr<Neuron>> neurons;
    std::shared_ptr<ActivationFunction> activationFunction;

    Layer();
    Layer(std::vector<std::shared_ptr<Neuron>> neurons, std::shared_ptr<ActivationFunction> activation);
    ~Layer();

    void addNeuron(std::shared_ptr<Neuron> neuron);
    std::vector<double> calculateOutput(const std::vector<double> &input);
};