//
// Created by mario on 6/20/18.
//

#include "functions.h"
#include "math.h"

double getActivation(double weightedSum) {
    double eToWSum = pow(M_E, weightedSum);
    return eToWSum / (eToWSum + 1);
}

double getWeightedSum(double* weights, double* biases, double* neuronValues, int length) {
    double weightedSum = 0.0;

    for (int index = 0; index < length; index++) {
        double weight = weights[index];
        double bias = biases[index];
        double neuronValue = neuronValues[index];

        double weightedActivation = weight * neuronValue + bias;
        weightedSum += weightedActivation;
    }

    return weightedSum;
}


double getActivationDerivative(double activationValue) {
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


/**
 * @return delta z[column, end_index] / delta b[column, end_index]. This is always
 * constant (1) since the bias is not multiplied by the neuron value when calculating weighted sum.
 */
double getWeightedSumBiasDerivative() {
    return 1;
}

double getCost(double neuronValue, double intendedValue) {
    double difference = neuronValue - intendedValue;

    return pow(difference, 2);
}
double getCostDerivative(double neuronValue, double intendedValue) {
    return neuronValue - intendedValue;
}

double getDefaultWeightValue(int numberOfLinks) {
    return sqrt(2 / numberOfLinks);
}