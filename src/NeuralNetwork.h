#pragma once

#include <vector>
#include <memory>
#include "export.h"

struct Neuron;
struct Layer;
class ActivationFunction;

struct DllExport NeuralNetwork
{
    std::vector<std::shared_ptr<Layer>> layers;
    std::vector<double> learningRates;
    double learningRate = 0.1;
    NeuralNetwork();
    NeuralNetwork(int numLayers, int numNeuronsPerLayer, int numInputs, int numOutputs, double rate, std::shared_ptr<ActivationFunction> activation, std::shared_ptr<ActivationFunction> outputActivation);
    NeuralNetwork(const std::vector<std::shared_ptr<Layer>> &layers) : layers(layers) {}
    ~NeuralNetwork() = default;
    std::vector<double> calculateOutput(const std::vector<double> &input);
    std::vector<double> train(const std::vector<double> &input, const std::vector<double> &output);
    void setLearningRate(double rate);
    void initRandomWeights();
    double getRandomWeight(double min, double max);
};
