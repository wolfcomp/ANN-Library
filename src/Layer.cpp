#include "Layer.h"
#include "Neuron.h"

void Layer::calculateN(std::vector<double> input)
{
    for (auto &neuron : neurons)
    {
        neuron->calculateN(input);
    }
}

std::vector<double> Layer::calculateOutput(std::vector<double> input)
{
    calculateN(input);
    return std::vector<double>();
}
