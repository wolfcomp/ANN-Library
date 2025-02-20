#pragma once

#include <vector>
#include <memory>
#include "ActivationFunction.h"

struct Neuron;
struct Layer;

struct NeuralNetwork
{
    std::vector<std::shared_ptr<Layer>> layers;
    ActivationFunction activationFunction = RELU;
    NeuralNetwork() = default;
    NeuralNetwork(int numLayers, int numNeuronsPerLayer, int numInputs, int numOutputs, ActivationFunction activation);
    NeuralNetwork(std::vector<std::shared_ptr<Layer>> layers, ActivationFunction activation) : layers(layers), activationFunction(activation) {}
    ~NeuralNetwork() = default;
    void calculateOutput(std::vector<double> input);
    void train(std::vector<double> input, std::vector<double> output);
    std::vector<double> compute(std::vector<double> input);
};
