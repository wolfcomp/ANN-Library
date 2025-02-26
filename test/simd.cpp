#include <random>
#include <vector>
#include <benchmark/benchmark.h>
#include <numeric>
#include <immintrin.h>
#include <xmmintrin.h>
#include <smmintrin.h>
#include <iostream>

static std::vector<double> weights = {};
static std::vector<double> inputs = {};
constexpr int size = 100000;

static void InitValues(int size)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0, 1);
    weights.resize(size);
    inputs.resize(size);

    for (int i = 0; i < size; i++)
    {
        weights[i] = dist(mt);
        inputs[i] = dist(mt);
    }
};

static void ForMultiply(benchmark::State &state)
{
    for (auto _ : state)
    {
        InitValues(size);
        double output = 0;
        for (int i = 0; i < weights.size(); i++)
        {
            output += weights[i] * inputs[i];
        }
        benchmark::DoNotOptimize(output);
    }
};
BENCHMARK(ForMultiply);

static void TransformMultiply(benchmark::State &state)
{
    std::vector<double> mulResult = {};
    mulResult.resize(size);
    for (auto _ : state)
    {
        InitValues(size);
        std::transform(weights.begin(), weights.end(), inputs.begin(), mulResult.begin(), std::multiplies<int>());
        double output = std::accumulate(mulResult.begin(), mulResult.end(), 0.0);
        benchmark::DoNotOptimize(output);
    }
};
BENCHMARK(TransformMultiply);

static void SSEMultiply(benchmark::State &state)
{
    InitValues(size);
    for (auto _ : state)
    {
        auto acc = _mm_setzero_pd();
        auto weightsStart = weights.data();
        auto inputsStart = inputs.data();
        auto weightsEnd = weightsStart + weights.size();
        for (; weightsStart < weightsEnd; weightsStart += 2, inputsStart += 2)
        {
            const auto a = _mm_loadu_pd(weightsStart);
            const auto b = _mm_loadu_pd(inputsStart);
            const auto dp = _mm_dp_pd(a, b, 0xFF);
            acc = _mm_add_pd(acc, dp);
        }
        double output = _mm_cvtsd_f64(acc);
        if (weightsStart < weightsEnd)
        {
            std::cout << "SSE did not finish off by: " << (weightsEnd - weightsStart) / sizeof(double) << std::endl;
            for (; weightsStart < weightsEnd; weightsStart++)
            {
                output += *weightsStart * *inputsStart;
                inputsStart++;
            }
        }
        benchmark::DoNotOptimize(output);
    }
};
BENCHMARK(SSEMultiply);

static void AVXMultiply(benchmark::State &state)
{
    InitValues(size);
    for (auto _ : state)
    {
        auto acc = _mm256_setzero_pd();
        auto weightsStart = weights.data();
        auto inputsStart = inputs.data();
        auto weightsEnd = weightsStart + weights.size();
        for (; weightsStart < weightsEnd; weightsStart += 4, inputsStart += 4)
        {
            const auto a = _mm256_loadu_pd(weightsStart);
            const auto b = _mm256_loadu_pd(inputsStart);
            const auto mul = _mm256_mul_pd(a, b);
            const auto hmul = _mm256_hadd_pd(mul, mul);
            const auto perm = _mm256_permute4x64_pd(hmul, 0x18);
            const auto dp = _mm256_hadd_pd(perm, perm);
            acc = _mm256_add_pd(acc, dp);
        }
        double output = _mm256_cvtsd_f64(acc);
        if (weightsStart < weightsEnd)
        {
            std::cout << "AVX did not finish off by: " << (weightsEnd - weightsStart) / sizeof(double) << std::endl;
            for (; weightsStart < weightsEnd; weightsStart++)
            {
                output += *weightsStart * *inputsStart;
                inputsStart++;
            }
        }
        benchmark::DoNotOptimize(output);
    }
};
BENCHMARK(AVXMultiply);

BENCHMARK_MAIN();