#include "NeuralNetwork.h"
#include "Layer.h"
#include "Neuron.h"
#include "ActivationFunction.h"
#include <algorithm>
#include <random>

std::vector<std::shared_ptr<ActivationFunction>> NeuralNetwork::activationFunctions = {
    std::make_shared<ActivationFunction>(),
    std::make_shared<Relu>(),
    std::make_shared<Sigmoid>(),
    std::make_shared<Tanh>(),
    std::make_shared<LeakyRelu>()};

NeuralNetwork::NeuralNetwork()
{
    layers = std::vector<std::shared_ptr<Layer>>();
}

NeuralNetwork::NeuralNetwork(int numLayers, int numNeuronsPerLayer, int numInputs, int numOutputs, double rate, std::shared_ptr<ActivationFunction> activation, std::shared_ptr<ActivationFunction> outputActivation)
{
    layers = std::vector<std::shared_ptr<Layer>>();
    learningRate = rate;
    // setup input layer
    layers.push_back(std::make_shared<Layer>(std::vector<std::shared_ptr<Neuron>>(), activation));
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
        auto layer = std::make_shared<Layer>(std::vector<std::shared_ptr<Neuron>>(), activation);
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
    auto errorDifference = output;
    // Output Layer Error correction
    std::transform(errorDifference.begin(), errorDifference.end(), calculatedOutput.begin(), errorDifference.begin(), std::minus<double>());
    for (auto &difference : errorDifference)
        difference *= learningRate;
    auto gradients = layers.back()->calculatedGradients();
    std::transform(gradients.begin(), gradients.end(), errorDifference.begin(), gradients.begin(), std::multiplies<double>());
    auto referencedLayer = layers[layers.size() - 2];
    auto inputs = std::vector<double>(referencedLayer->neurons.size(), 0.0);
    std::transform(referencedLayer->neurons.begin(), referencedLayer->neurons.end(), inputs.begin(), [](auto &neuron)
                   { return neuron->output; });

    referencedLayer = layers[layers.size() - 1];
    for (size_t i = 0; i < referencedLayer->neurons.size(); i++)
    {
        auto &neuron = referencedLayer->neurons[i];
        for (size_t j = 0; j < neuron->weights.size(); j++)
        {
            neuron->weights[j] += errorDifference[i] * inputs[j];
        }
        neuron->bias += gradients[i];
    }

    // Hidden Layer and Input Layer Error correction
    for (int i = static_cast<int>(layers.size()) - 2; i >= 0; i--)
    {
        auto newGradients = layers[i]->calculatedGradients();
        if (i == 0)
        {
            inputs = input;
        }
        else
        {
            referencedLayer = layers[i - 1];
            inputs = std::vector<double>(referencedLayer->neurons.size(), 0.0);
            std::transform(referencedLayer->neurons.begin(), referencedLayer->neurons.end(), inputs.begin(), [](auto &neuron)
                           { return neuron->output; });
        }
        referencedLayer = layers[i + 1];
        for (size_t j = 0; j < newGradients.size(); j++)
        {
            double error = 0.0;
            for (size_t k = 0; k < referencedLayer->neurons.size(); k++)
            {
                error += referencedLayer->neurons[k]->weights[j] * gradients[k];
            }
            newGradients[j] *= error;
        }
        for (auto &difference : newGradients)
            difference *= learningRate;
        gradients = newGradients;

        for (size_t j = 0; j < layers[i]->neurons.size(); j++)
        {
            auto &neuron = layers[i]->neurons[j];
            for (size_t k = 0; k < neuron->weights.size(); k++)
            {
                neuron->weights[k] += inputs[k] * gradients[j];
            }
            neuron->bias += gradients[j];
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
