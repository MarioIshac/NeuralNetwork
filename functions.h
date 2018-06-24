#ifndef NEURALNETWORK_FUNCTIONS_H
#define NEURALNETWORK_FUNCTIONS_H

/**
 * Activates the value given by the weighted sum function. This activated value will be put
 * into the neuron.
 *
 * @param weightedSum The result of the weighted sum function.
 * @return The activated neuron value.
 */
double getDefaultActivation(double weightedSum);

/**
 * Calculates the derivative of the sigmoid function at the weighted sum, which, when inputted into sigmoid,
 * gives activationValue.
 *
 * @param activationValue Since the derivative of a sigmoid does not depend on the input of a sigmoid,
 * but rather the output of the sigmoid, the value of the neuron is passed for efficiency.
 * @return activationValue * (1 - activationValue)
 */
double getDefaultActivationDerivative(double activationValue);

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
double getDefaultCost(double neuronValue, double intendedValue);

/**
 * Calculates the derivative of the cost of {@code neuronValue} relative to the target output.
 *
 * @param neuronValue The output produced by the neural network.
 * @param intendedValue The correct output that the neural network aims to produce.
 * @return Δ "cost"/effect of the network outputting {@code neuronValue} instead of {@code intendedValue} /
 *         Δ neuronValue
 */
double getDefaultCostDerivative(double neuronValue, double intendedValue);

/**
 * Calculates the initial value that a weight[endLayerIndex][endNeuronIndex][startNeuronIndex]
 * should have before the first epoch of the network.
 *
 * @param previousLayerSize Number of neurons in startLayerIndex.
 * @param layerSize Number on neurons in endLayerIndex.
 * @return
 */
double getDefaultInitialWeightValue(double previousLayerSize, double layerSize);

/**
 * Calculates the initial value that a bias[endLayerIndex][endNeuronIndex]
 * should have before the first epoch of the network.
 *
 * @param previousLayerSize Number of neurons in startLayerIndex.
 * @param layerSize Number on neurons in endLayerIndex.
 * @return
 */
double getDefaultInitialBiasValue(double previousLayerSize, double layerSize);

#endif
