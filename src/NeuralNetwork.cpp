#include "NeuralNetwork.h"
#include "Layer.h"
#include "Neuron.h"
#include "ActivationFunction.h"
#include <algorithm>
#include <random>

NeuralNetwork::NeuralNetwork()
{
    layers = std::vector<std::shared_ptr<Layer>>();
}

NeuralNetwork::NeuralNetwork(int numLayers, int numNeuronsPerLayer, int numInputs, int numOutputs, double rate, std::shared_ptr<ActivationFunction> activation, std::shared_ptr<ActivationFunction> outputActivation)
{
    layers = std::vector<std::shared_ptr<Layer>>();
    layers.reserve(numLayers + 1);
    learningRate = rate;
    // setup input layer
    auto layer = std::make_shared<Layer>(std::vector<std::shared_ptr<Neuron>>(), activation);
    layers.push_back(layer);
    layer->neurons.reserve(numNeuronsPerLayer);
    for (int i = 0; i < numNeuronsPerLayer; i++)
    {
        auto neuron = std::make_shared<Neuron>(0, std::vector<double>());
        neuron->weights.resize(numInputs);
        layer->addNeuron(neuron);
    }

    // setup hidden layers
    for (int i = 0; i < numLayers - 1; i++)
    {
        layer = std::make_shared<Layer>(std::vector<std::shared_ptr<Neuron>>(), activation);
        layers.push_back(layer);
        layer->neurons.reserve(numNeuronsPerLayer);
        auto prevLayerNeuronCount = layers[i]->neurons.size();
        for (int j = 0; j < numNeuronsPerLayer; j++)
        {
            auto neuron = std::make_shared<Neuron>(0, std::vector<double>());
            neuron->weights.resize(prevLayerNeuronCount);
            layer->addNeuron(neuron);
        }
    }

    // setup output layer
    auto lastHiddenLayerNeuronSize = layers.back()->neurons.size();
    auto outputLayer = std::make_shared<Layer>(std::vector<std::shared_ptr<Neuron>>(), outputActivation);
    layers.push_back(outputLayer);
    for (int i = 0; i < numOutputs; i++)
    {
        auto neuron = std::make_shared<Neuron>(0, std::vector<double>());
        neuron->weights.resize(lastHiddenLayerNeuronSize);
        outputLayer->addNeuron(neuron);
    }

    // initialize random weights
    initRandomWeights();
}

std::vector<double> NeuralNetwork::calculateOutput(const std::vector<double> &input)
{
    auto inputLayer = layers[0];
    auto outputFromLastLayer = inputLayer->calculateOutput(input);
    for (int i = 1; i < layers.size(); i++)
    {
        outputFromLastLayer = layers[i]->calculateOutput(outputFromLastLayer);
    }
    return outputFromLastLayer;
}

std::vector<double> NeuralNetwork::train(const std::vector<double> &input, const std::vector<double> &output)
{
    auto calculatedOutput = calculateOutput(input);

    for (int layerIndex = static_cast<int>(layers.size()) - 1; layerIndex >= 0; layerIndex--)
    {
        std::vector<double> inputs;
        if (layerIndex > 0)
        {
            auto nextLayer = layers[layerIndex - 1];
            inputs.reserve(nextLayer->neurons.size());
            for (size_t prevNeuronIndex = 0; prevNeuronIndex < nextLayer->neurons.size(); prevNeuronIndex++)
            {
                inputs.push_back(nextLayer->neurons[prevNeuronIndex]->output);
            }
        }
        else
        {
            inputs = input;
        }
        auto currentLayer = layers[layerIndex];
        for (size_t neuronIndex = 0; neuronIndex < currentLayer->neurons.size(); neuronIndex++)
        {
            auto &neuron = currentLayer->neurons[neuronIndex];
            double error = 0.0;
            if (layerIndex == layers.size() - 1)
            {
                error = output[neuronIndex] - neuron->output;
                neuron->dN = error * currentLayer->activationFunction->derivative(neuron->N);
                for (size_t weightIndex = 0; weightIndex < neuron->weights.size(); weightIndex++)
                {
                    neuron->weights[weightIndex] += learningRate * error * inputs[weightIndex];
                }
            }
            else
            {
                auto prevLayer = layers[layerIndex + 1];
                for (size_t prevNeuronIndex = 0; prevNeuronIndex < prevLayer->neurons.size(); prevNeuronIndex++)
                {
                    error += prevLayer->neurons[prevNeuronIndex]->weights[neuronIndex] * prevLayer->neurons[prevNeuronIndex]->dN;
                }
                neuron->dN = error * currentLayer->activationFunction->derivative(neuron->N);
                for (size_t weightIndex = 0; weightIndex < neuron->weights.size(); weightIndex++)
                {
                    neuron->weights[weightIndex] += learningRate * neuron->dN * inputs[weightIndex];
                }
            }
            neuron->bias += learningRate * neuron->dN;
        }
    }

    return calculatedOutput;
}

void NeuralNetwork::initRandomWeights()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (auto &layer : layers)
    {
        for (auto &neuron : layer->neurons)
        {
            for (auto &weight : neuron->weights)
            {
                weight = dis(gen);
            }
            neuron->bias = dis(gen);
        }
    }
}
