#include "NeuralNetwork.h"
#include "Layer.h"
#include "Neuron.h"

NeuralNetwork::NeuralNetwork(int numLayers, int numNeuronsPerLayer, int numInputs, int numOutputs, ActivationFunction activation)
{
    activationFunction = activation;
    layers = std::vector<std::shared_ptr<Layer>>();
    // setup input layer
    layers.push_back(std::make_shared<Layer>(std::vector<std::shared_ptr<Neuron>>()));
    auto inputLayer = layers[0];
    for (int i = 0; i < numInputs; i++)
    {
        auto neuron = std::make_shared<Neuron>(0, std::vector<double>());
        neuron->weights.resize(numInputs);
        inputLayer->addNeuron(neuron);
    }

    // setup hidden layers
    for (int i = 0; i < numLayers; i++)
    {
        auto layer = std::make_shared<Layer>(std::vector<std::shared_ptr<Neuron>>());
        layers.push_back(layer);
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
    auto outputLayer = std::make_shared<Layer>(std::vector<std::shared_ptr<Neuron>>());
    layers.push_back(outputLayer);
    for (int i = 0; i < numOutputs; i++)
    {
        auto neuron = std::make_shared<Neuron>(0, std::vector<double>());
        neuron->weights.resize(lastHiddenLayerNeuronSize);
        outputLayer->addNeuron(neuron);
    }
}

void NeuralNetwork::calculateOutput(std::vector<double> input)
{
    auto inputLayer = layers[0];
    inputLayer->calculateOutput(input);
}

void NeuralNetwork::train(std::vector<double> input, std::vector<double> output)
{
}