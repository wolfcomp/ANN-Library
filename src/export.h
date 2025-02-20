#pragma once
#define DllExport __declspec(dllexport)
#define ExternCDllExport extern "C" __declspec(dllexport)
#include <memory>

// predefines
struct Neuron;

// functions
DllExport std::shared_ptr<Neuron> createNeuron(double bias, std::vector<double> weights);