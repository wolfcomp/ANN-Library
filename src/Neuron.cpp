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

    auto weightStart = weights.data();
    auto inputStart = input.data();
    auto weightEnd = weightStart + weights.size();

    // __m128d is a 2 double large vector type used for SIMD instructions
    // for more info see the Intel(R) Intrinsics Guide (https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html) for the respective _mm_ instructions

    // initialize accumulator
    __m128d acc = _mm_setzero_pd();
    for (; weightStart < weightEnd; weightStart += 2, inputStart += 2)
    {
        // load 2 doubles from weight and input
        const auto a = _mm_loadu_pd(weightStart);
        const auto b = _mm_loadu_pd(inputStart);

        // multiply index 0 with 0 and index 1 with 1
        const auto dp = _mm_dp_pd(a, b, 0xFF);

        // add result to accumulator
        acc = _mm_add_pd(acc, dp);
    }

    // convert index 0 of accumulator to double and add to N
    N += _mm_cvtsd_f64(acc);

    // All of this is the same as:
    //
    // for (int i = 0; i < weights.size(); i++)
    //     N += weights[i] * input[i];
    //
    // The method used with _mm_ instructions is just over 100 times faster
}
