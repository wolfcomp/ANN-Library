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

    auto weightStart = &weights[0];
    auto inputStart = &input[0];
    auto weightEnd = &weights[weights.size() - 1];

    // initialize simd accumulator
    __m128d acc = _mm_setzero_pd();
    for (; weightStart <= weightEnd; weightStart += 2, inputStart += 2)
    {
        // load 4 doubles from weight
        const auto a = _mm_loadu_pd(weightStart);
        // load 4 doubles from input
        const auto b = _mm_loadu_pd(inputStart);
        // multiply a and b
        const auto dp = _mm_dp_pd(a, b, 0xFF);
        // add to accumilator
        acc = _mm_add_pd(acc, dp);
    }

    // convert accumulator to double and add to output
    N += _mm_cvtsd_f64(acc);
}
