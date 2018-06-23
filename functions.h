//
// Created by mario on 6/20/18.
//

#ifndef NEURALNETWORK_FUNCTIONS_H
#define NEURALNETWORK_FUNCTIONS_H

double getActivation(double weightedSum);
double getWeightedSum(double* weights, double* biases, double* neuronValues, int length);

/**
 * Calculates the derivative of the sigmoid function at the weighted sum, which, when inputted into sigmoid,
 * gives activationValue.
 *
 * @param activationValue Since the derivative of a sigmoid does not depend on the input of a sigmoid,
 * but rather the output of the sigmoid, the value of the neuron is passed for efficiency.
 * @return activationValue * (1 - activationValue)
 */
double getActivationDerivative(double activationValue);
double getWeightedSumNeuronValueDerivative(double weight);
double getWeightedSumWeightDerivative(double neuronValue);
double getWeightedSumBiasDerivative();

double getCost(double neuronValue, double intendedValue);
double getCostDerivative(double neuronValue, double intendedValue);

double getDefaultWeightValue(int numberOfLinks);

#endif
