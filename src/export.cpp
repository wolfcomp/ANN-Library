#include "export.h"
#include "Neuron.h"

DllExport std::shared_ptr<Neuron> createNeuron(double bias, std::vector<double> weights) { return std::make_shared<Neuron>(bias, weights); }
