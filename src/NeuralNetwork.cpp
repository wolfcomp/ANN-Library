#include "NeuralNetwork.h"
#include "Layer.h"
#include "Neuron.h"
#include "ActivationFunction.h"
#include <algorithm>
#include <random>
#include <iostream>
#include <xmmintrin.h>
#include <smmintrin.h>

std::random_device rd;
std::mt19937 gen(rd());

NeuralNetwork::NeuralNetwork()
{
    layers = std::vector<std::shared_ptr<Layer>>();
}

NeuralNetwork::NeuralNetwork(int numLayers, int numNeuronsPerLayer, int numInputs, int numOutputs, double rate, std::shared_ptr<ActivationFunction> activation, std::shared_ptr<ActivationFunction> outputActivation)
{
    layers = std::vector<std::shared_ptr<Layer>>();
    layers.reserve(numLayers + 1);
    learningRate = rate;

    // setup hidden layers
    auto prevLayerNeuronCount = numInputs;
    for (int i = 0; i < numLayers; i++)
    {
        auto layer = std::make_shared<Layer>(std::vector<std::shared_ptr<Neuron>>(), activation);
        layers.push_back(layer);
        layer->neurons.reserve(numNeuronsPerLayer);
        for (int j = 0; j < numNeuronsPerLayer; j++)
        {
            auto neuron = std::make_shared<Neuron>(0, std::vector<double>(prevLayerNeuronCount, 0.0));
            layer->addNeuron(neuron);
        }
        prevLayerNeuronCount = numNeuronsPerLayer;
    }

    // setup output layer
    auto outputLayer = std::make_shared<Layer>(std::vector<std::shared_ptr<Neuron>>(), outputActivation);
    layers.push_back(outputLayer);
    for (int i = 0; i < numOutputs; i++)
    {
        auto neuron = std::make_shared<Neuron>(0, std::vector<double>(prevLayerNeuronCount, 0.0));
        outputLayer->addNeuron(neuron);
    }

    // initialize random weights
    initRandomWeights();
}

std::vector<double> NeuralNetwork::calculateOutput(const std::vector<double> &input)
{
    auto outputFromLastLayer = input;
    for (int i = 0; i < layers.size(); i++)
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
        // Predefine inputs vector
        std::vector<double> inputs;
        if (layerIndex > 0)
        {
            // Get inputs from previous layer
            auto nextLayer = layers[layerIndex - 1];
            inputs.reserve(nextLayer->neurons.size());
            for (size_t prevNeuronIndex = 0; prevNeuronIndex < nextLayer->neurons.size(); prevNeuronIndex++)
            {
                inputs.push_back(nextLayer->neurons[prevNeuronIndex]->output);
            }
        }
        else
        {
            // Get inputs from input
            inputs = input;
        }

        // Get the current layer pointer and loop through its neurons
        auto currentLayer = layers[layerIndex];
        for (size_t neuronIndex = 0; neuronIndex < currentLayer->neurons.size(); neuronIndex++)
        {
            auto currentNeuron = currentLayer->neurons[neuronIndex];
            double error = 0.0;
            // Comments prefixed with _mm_ are from the Intel(R) Intrinsic Guide for SIMD Instructions (https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)

            if (layerIndex == layers.size() - 1)
            {
                // Calculate the error from the desired output
                // _mm_sub_pd
                error = output[neuronIndex] - currentNeuron->output;

                // Calculate the error gradient for the current neuron
                // _mm_dp_pd; flag: 0xFF
                currentNeuron->errorGradient = error * currentLayer->activationFunction->derivative(currentNeuron->N);
                for (size_t weightIndex = 0; weightIndex < currentNeuron->weights.size(); weightIndex++)
                {
                    // Calculate the new weight for the current neuron with its corresponding input
                    // _mm_dp_pd; flag: 0xFF
                    // _mm_dp_pd; flag: 0xFF
                    // _mm_add_pd
                    currentNeuron->weights[weightIndex] += learningRate * inputs[weightIndex] * error;
                }
            }
            else
            {
                // Get the previous layer pointer
                auto prevLayer = layers[layerIndex + 1];
                for (size_t prevNeuronIndex = 0; prevNeuronIndex < prevLayer->neurons.size(); prevNeuronIndex++)
                {
                    // Sum up the error rate from the previous layer for the current neuron based on its weight and error gradient
                    // _mm_dp_pd; flag: 0xFF
                    // _mm_add_pd
                    error += prevLayer->neurons[prevNeuronIndex]->errorGradient * prevLayer->neurons[prevNeuronIndex]->weights[neuronIndex];
                }

                // Calculate the error gradient for the current neuron
                // _mm_dp_pd; flag: 0xFF
                currentNeuron->errorGradient = error * currentLayer->activationFunction->derivative(currentNeuron->N);
                for (size_t weightIndex = 0; weightIndex < currentNeuron->weights.size(); weightIndex++)
                {
                    // Calculate the new weight for the current neuron with its corresponding input
                    // _mm_dp_pd; flag: 0xFF
                    // _mm_dp_pd; flag: 0xFF
                    // _mm_add_pd
                    currentNeuron->weights[weightIndex] += learningRate * inputs[weightIndex] * currentNeuron->errorGradient;
                }
            }

            // Calculate the bias for the current neuron
            // _mm_dp_pd; flag: 0xFF
            // _mm_add_pd
            currentNeuron->bias += learningRate * currentNeuron->errorGradient;
        }
    }

    return calculatedOutput;
}

void NeuralNetwork::setLearningRate(double rate)
{
    learningRate = rate;
    for (auto &learning : learningRates)
        learning = rate;
}

void NeuralNetwork::initRandomWeights()
{
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

double NeuralNetwork::getRandomWeight(double min, double max)
{
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}
