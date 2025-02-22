#include "Layer.h"
#include "Neuron.h"
#include <algorithm>
#include "ActivationFunction.h"

std::vector<double> Layer::calculateOutput(const std::vector<double> &input)
{
    auto ret = std::vector<double>();
    ret.reserve(neurons.size());
    for (auto &neuron : neurons)
    {
        neuron->calculateN(input);
        neuron->output = (*activationFunction)(neuron->N);
        ret.push_back(neuron->output);
    }
    return ret;
}