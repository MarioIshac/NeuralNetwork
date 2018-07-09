#ifndef NEURALNETWORK_FUNCTIONS_H
#define NEURALNETWORK_FUNCTIONS_H

#include "model.h"

double getSigmoid(double weightedSum);
double getSigmoidPrime(double activationValue);
double getReLU(double weightedSum);
double getReLUPrime(double activationVale);
double getTanH(double weightedSum);
double getTanHPrime(double activationValue);

/**
 * Calculates the derivative of the function that produces the weighted sum given a weight, neuron value and bias,
 * with respect to the neuron value.
 *
 * @param weight The weight that the function multiplies with the neuron value, which is w[endLayerIndex, endNeuronIndex,
 * startNeuronIndex]
 * @return Δ Z[endLayerIndex, endNeuronIndex] / Δ a[startLayerIndex, startNeuronIndex]
 */
double getWeightedSumNeuronValueDerivative(double weight);

/**
 * Calculates the derivative of the function that produces the weighted sum given a weight, neuron value and bias,
 * with respect to the weight.
 *
 * @param neuronValue The neuron value that the function multiplies with the weight, which is a[startLayerIndex,
 * startNeuronIndex]
 * @return Δ Z[endLayerIndex, endNeuronIndex] / Δ w[endLayerIndex, endNeuronIndex]
 */
double getWeightedSumWeightDerivative(double neuronValue);

/**
 * Calculates the derivative of the function that produces the weighted sum given a weight, neuron value and bias,
 * with respect to the bias.
 *
 * @return Δ Z[endLayerIndex, endNeuronIndex] / Δ b[endLayerIndex, endNeuronIndex]
 */
double getWeightedSumBiasDerivative();

/**
 * Calculates the cost of {@code neuronValue} relative to the target output.
 *
 * @param neuronValue The output produced by the neural network.
 * @param intendedValue The correct output that the neural network aims to produce.
 * @return The "cost"/effect of the network outputting {@code neuronValue} instead of {@code intendedValue}.
 */
double getCost(double neuronValue, double intendedValue);

/**
 * Calculates the derivative of the cost of {@code neuronValue} relative to the target output.
 *
 * @param neuronValue The output produced by the neural network.
 * @param intendedValue The correct output that the neural network aims to produce.
 * @return Δ "cost"/effect of the network outputting {@code neuronValue} instead of {@code intendedValue} /
 *         Δ neuronValue
 */
double getCostPrime(double neuronValue, double intendedValue);

/**
 * Calculates the initial value that a weight[endLayerIndex][endNeuronIndex][startNeuronIndex]
 * should have before the first epoch of the network.
 *
 * @param previousLayerSize Number of neurons in startLayerIndex.
 * @param layerSize Number on neurons in endLayerIndex.
 * @return
 */
double getInitialXavierWeight(double previousLayerSize, double layerSize);

double getInitialRandomWeight(double previousLayerSize, double layerSize);

/**
 * Calculates the initial value that a bias[endLayerIndex][endNeuronIndex]
 * should have before the first epoch of the network.
 *
 * @param previousLayerSize Number of neurons in startLayerIndex.
 * @param layerSize Number on neurons in endLayerIndex.
 * @return
 */
double getInitialBias(double previousLayerSize, double layerSize);

#endif
