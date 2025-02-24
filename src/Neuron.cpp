#include "Neuron.h"
#include <stdexcept>
#include <xmmintrin.h>
#include <smmintrin.h>

void Neuron::calculateN(const std::vector<double> &input)
{
    if (input.size() != weights.size())
    {
        throw std::invalid_argument("Input size does not match weight size");
    }

    if (weights.size() == 0 || input.size() == 0)
    {
        throw std::invalid_argument("Input or weight size is 0");
    }
    N = bias;

    for (int i = 0; i < weights.size(); i++)
    {
        N += weights[i] * input[i];
    }
}
