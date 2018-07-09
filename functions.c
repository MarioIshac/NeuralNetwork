#include <stdlib.h>
#include "functions.h"
#include "math.h"

double getSigmoid(double weightedSum) {
    double eToWSum = pow(M_E, weightedSum);
    return eToWSum / (eToWSum + 1);
}

double getSigmoidPrime(double activationValue) {
    return activationValue * (1 - activationValue);
}

double getReLU(double weightedSum) {
    return weightedSum <= 0 ? 0 : weightedSum;
}

double getReLUPrime(double activationValue) {
    return activationValue == 0 ? 0 : 1;
}

double getTanH(double weightedSum) {
    return 2 * getSigmoid(2 * weightedSum) - 1;
}

double getTanHPrime(double activationValue) {
    return 1 - pow(activationValue, 2);
}

double getInitialXavierWeight(double previousLayerSize, double layerSize) {
    return sqrt(2 / previousLayerSize);
}

double getInitialRandomWeight(double previousLayerSize, double layerSize) {
    return ((double) rand() / RAND_MAX);
}

double getInitialBias(double previousLayerSize, double layerSize) {
    return 0;
}

double getCost(double neuronValue, double intendedValue) {
    double difference = neuronValue - intendedValue;

    return 0.5 * pow(difference, 2);
}

double getCostPrime(double neuronValue, double intendedValue) {
    return neuronValue - intendedValue;
}

WeightInitializationFunction getRandomWeightGenerator(double min, double max) {
    double getRandomWeight(double startNeuronCount, double endNeuronCount) {
        float randomValue = rand();

        randomValue /= (max - min);
        randomValue += min;

        return randomValue;
    }

    return getRandomWeight;
}