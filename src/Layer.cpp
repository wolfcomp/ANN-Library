#include "Layer.h"
#include "Neuron.h"
#include <algorithm>
#include "ActivationFunction.h"

void Layer::calculateNeuron(const std::vector<double> &input)
{
    for (auto &neuron : neurons)
    {
        neuron->calculateN(input);
        neuron->output = (*activationFunction)(neuron->N);
    }
}

std::vector<double> Layer::calculateOutput(const std::vector<double> &input)
{
    calculateNeuron(input);
    auto ret = std::vector<double>(neurons.size(), 0.0);
    std::transform(neurons.begin(), neurons.end(), ret.begin(), [](auto &neuron)
                   { return neuron->output; });
    return ret;
}

std::vector<double> Layer::calculatedGradients()
{
    auto ret = std::vector<double>(neurons.size(), 0.0);
    std::transform(neurons.begin(), neurons.end(), ret.begin(), [this](auto &neuron)
                   { return activationFunction->derivative(neuron->N); });
    return ret;
}
