#include "functions.h"
#include "math.h"

double getDefaultActivation(double weightedSum) {
    double eToWSum = pow(M_E, weightedSum);
    return eToWSum / (eToWSum + 1);
}

double getDefaultActivationDerivative(double activationValue) {
    return activationValue * (1 - activationValue);
}

/**
 * start_index can be any value
 *
 * @param neuronValue value of w[column][end_index][start_index]
 * @return delta z[column, end_index] / delta w[column, end_index, start_index]. This is always
 * equivalent to the neuron value itself since weight is multiplied with the neuron value when calculating
 * weighted sum.
 */
double getWeightedSumWeightDerivative(double neuronValue) {
    return neuronValue;
}

double getWeightedSumNeuronValueDerivative(double weight) {
    return weight;
}

double getDefaultInitialWeightValue(double previousLayerSize, double layerSize) {
    return sqrt(2 / (previousLayerSize));
}

double getDefaultInitialBiasValue(double previousLayerSize, double layerSize) {
    return 0;
}

/**
 * @return delta z[column, end_index] / delta b[column, end_index]. This is always
 * constant (1) since the bias is not multiplied by the neuron value when calculating weighted sum.
 */
double getWeightedSumBiasDerivative() {
    return 1;
}

double getDefaultCost(double neuronValue, double intendedValue) {
    double difference = neuronValue - intendedValue;

    return pow(difference, 2);
}
double getDefaultCostDerivative(double neuronValue, double intendedValue) {
    return neuronValue - intendedValue;
}