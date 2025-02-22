#pragma once

#include <vector>
#include <memory>

struct Neuron;
struct Layer;
class ActivationFunction;

struct NeuralNetwork
{
    std::vector<std::shared_ptr<Layer>> layers;
    double learningRate = 0.1;
    NeuralNetwork();
    NeuralNetwork(int numLayers, int numNeuronsPerLayer, int numInputs, int numOutputs, double rate, std::shared_ptr<ActivationFunction> activation, std::shared_ptr<ActivationFunction> outputActivation);
    NeuralNetwork(const std::vector<std::shared_ptr<Layer>> &layers) : layers(layers) {}
    ~NeuralNetwork() = default;
    std::vector<double> calculateOutput(const std::vector<double> &input);
    std::vector<double> train(const std::vector<double> &input, const std::vector<double> &output);
    void setLearningRate(double rate) { learningRate = rate; }
    void initRandomWeights();
};
