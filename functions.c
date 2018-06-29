#include <stdlib.h>
#include "functions.h"
#include "math.h"

double applySigmoid(double weightedSum) {
    double eToWSum = pow(M_E, weightedSum);
    return eToWSum / (eToWSum + 1);
}

double applySigmoidDerivative(double activationValue) {
    return activationValue * (1 - activationValue);
}

double applyReLU(double weightedSum) {
    return weightedSum < 0 ? 0 : weightedSum;
}

double applyReLUDerivative(double activationValue) {
    return activationValue == 0 ? 0 : 1;
}

double applyTanH(double weightedSum) {
    return 2 * applyReLU(2 * weightedSum) - 1;
}

double applyTanHDerivative(double activationValue) {
    return 1 - pow(activationValue, 2);
}

double getInitialXavierWeight(double previousLayerSize, double layerSize) {
    return sqrt(2 / previousLayerSize);
}

double getInitialRandomWeight(double previousLayerSize, double layerSize) {
    return ((double) rand() / RAND_MAX) * 0.01;
}

double getInitialBias(double previousLayerSize, double layerSize) {
    return 0;
}

double getCost(double neuronValue, double intendedValue) {
    double difference = neuronValue - intendedValue;

    return 0.5 * pow(difference, 2);
}

double getCostDerivative(double neuronValue, double intendedValue) {
    return neuronValue - intendedValue;
}