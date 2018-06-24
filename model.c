#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include "model.h"
#include "functions.h"

#define GRADIENT_CHECKING true
#define PRINT_NEURON_VALUE false
#define PRINT_WEIGHT_UPDATE false
#define PRINT_BIAS_UPDATE false
#define PRINT_EPOCH_UPDATE true

/**
 * No weights are in the INPUT_LAYER. Thus, layer N is indexed as N - 1 in the weights
 * array of matrices.
 * @param layerIndex The layer index within the neural network.
 * @return The index within the weights storage of the matrix of weights that belong to layers[layerIndes].
 */
int offsetLayer(int layerIndex) {
    return layerIndex - 1;
}

/**
 * @param model
 * @param input The head to ann input array of size <code>model.neuronsPerLayer[INPUT_LAYER]</code> that has the inputs
 * of the model.
 */
void setInput(struct Model* model, double* inputHead) {
    model->values[INPUT_LAYER] = inputHead;
}

void propagateInputForward(struct Model* model, double* inputHead) {
    setInput(model, inputHead);

    for (int endLayerIndex = 1; endLayerIndex < NUMBER_OF_LAYERS; endLayerIndex++) {
        int offsetEndLayerIndex = offsetLayer(endLayerIndex);
        int startLayerIndex = endLayerIndex - 1;

        int endNeuronCount = model->neuronsPerLayer[endLayerIndex];
        int startNeuronCount = model->neuronsPerLayer[startLayerIndex];

        for (int endNeuronIndex = 0; endNeuronIndex < endNeuronCount; endNeuronIndex++) {
            double weightedSum = 0.0;
            double bias = model->biases[offsetEndLayerIndex][endNeuronIndex];

            for (int startNeuronIndex = 0; startNeuronIndex < startNeuronCount; startNeuronIndex++) {
                double weightOfLink = model->weights[offsetEndLayerIndex][endNeuronIndex][startNeuronIndex];
                double previousNeuronValue = model->values[startLayerIndex][startNeuronIndex];

                double weightedInfluence = weightOfLink * previousNeuronValue + bias;
                weightedSum += weightedInfluence;
            }

            double activatedNeuronValue = model->getActivation(weightedSum);

#if PRINT_NEURON_VALUE
            printf("Neuron[%i][%i] - Pre-Acivation: %lf, Post-Activation: %lf\n", layerIndex, neuronIndex, weightedSum, activatedNeuronValue);
#endif
            model->values[endLayerIndex][endNeuronIndex] = activatedNeuronValue;
        }
    }
}

#if GRADIENT_CHECKING
double getTotalCost(struct Model* model, const double targetOutputs[]) {
    int outputNeuronCount = model->neuronsPerLayer[OUTPUT_LAYER];

    double totalCost = 0.0;

    for (int outputNeuronIndex = 0; outputNeuronIndex < outputNeuronCount; outputNeuronIndex++) {
        double outputNeuronValue = model->values[OUTPUT_LAYER][outputNeuronIndex];
        double expectedOutputNeuronValue = targetOutputs[outputNeuronIndex];

        double cost = model->getCost(outputNeuronValue, expectedOutputNeuronValue);
        totalCost += cost;
    }

    return totalCost;
}

void initCheckParameterGradients(struct Model *model, double ***checkWeightGradients, double **checkBiasGradients) {
    for (int endLayerIndex = 1; endLayerIndex < NUMBER_OF_LAYERS; endLayerIndex++) {
        int offsetEndLayerIndex = offsetLayer(endLayerIndex);
        int startLayerIndex = endLayerIndex - 1;

        int endNeuronCount = model->neuronsPerLayer[endLayerIndex];
        int startNeuronCount = model->neuronsPerLayer[startLayerIndex];
        checkWeightGradients[offsetEndLayerIndex] = malloc(sizeof(double*) * endNeuronCount);
        checkBiasGradients[offsetEndLayerIndex] = malloc(sizeof(double) * endNeuronCount);

        for (int endNeuronIndex = 0; endNeuronIndex < endNeuronCount; endNeuronIndex++) {
            checkWeightGradients[offsetEndLayerIndex][endNeuronIndex] = malloc(sizeof(double) * startNeuronCount);
            checkBiasGradients[offsetEndLayerIndex][endNeuronIndex] = 0.0;

            for (int startNeuronIndex = 0; startNeuronIndex < startNeuronCount; startNeuronIndex++) {
                checkWeightGradients[offsetEndLayerIndex][endNeuronIndex][startNeuronIndex] = 0;
            }
        }
    }
}

/**
 * Updates the check weight and bias gradients (computed numerically as opposed to through back propagation).
 *
 * @param model
 * @param input
 * @param targetOutput
 * @param checkWeightGradients
 * @param checkBiasGradients
 */
void updateCheckParameterGradients(struct Model* model, double input[], const double targetOutput[],
                                   double** checkWeightGradient[], double* checkBiasGradients[]) {
    static float epsilon = 1e-6;

    double preChangeTotalCost;
    double postChangeTotalCost;
    double costDifference;

    for (int endLayerIndex = 1; endLayerIndex < NUMBER_OF_LAYERS; endLayerIndex++) {
        int offsetEndLayerIndex = offsetLayer(endLayerIndex);
        int startLayerIndex = endLayerIndex - 1;

        int endNeuronCount = model->neuronsPerLayer[endLayerIndex];
        int startNeuronCount = model->neuronsPerLayer[startLayerIndex];

        for (int endNeuronIndex = 0; endNeuronIndex < endNeuronCount; endNeuronIndex++) {
            propagateInputForward(model, input);
            preChangeTotalCost = getTotalCost(model, targetOutput);

            model->biases[offsetEndLayerIndex][endNeuronIndex] += epsilon;

            propagateInputForward(model, input);
            postChangeTotalCost = getTotalCost(model, targetOutput);

            costDifference = postChangeTotalCost - preChangeTotalCost;
            double checkBiasDeltaInfluence = costDifference / epsilon;

            checkBiasDeltaInfluence *= model->learningRate;

            checkBiasGradients[offsetEndLayerIndex][endNeuronIndex] += checkBiasDeltaInfluence;

            model->biases[offsetEndLayerIndex][endNeuronIndex] -= epsilon;

            for (int startNeuronIndex = 0; startNeuronIndex < startNeuronCount; startNeuronIndex++) {
                propagateInputForward(model, input);
                preChangeTotalCost = getTotalCost(model, targetOutput);

                model->weights[offsetEndLayerIndex][endNeuronIndex][startNeuronIndex] += epsilon;

                propagateInputForward(model, input);
                postChangeTotalCost = getTotalCost(model, targetOutput);

                // Undo change
                model->weights[offsetEndLayerIndex][endNeuronIndex][startNeuronIndex] -= epsilon;

                costDifference = postChangeTotalCost - preChangeTotalCost;
                double checkWeightDeltaInfluence = costDifference / epsilon;

                checkWeightDeltaInfluence *= model->learningRate;

                checkWeightGradient[offsetEndLayerIndex][endNeuronIndex][startNeuronIndex] += checkWeightDeltaInfluence;
            }
        }
    }
}

void printCheckParamterGradients(struct Model *model, double** checkWeightGradients[], double* checkBiasGradients[]) {
    for (int endLayerIndex = 1; endLayerIndex < NUMBER_OF_LAYERS; endLayerIndex++) {
        int offsetEndLayerIndex = offsetLayer(endLayerIndex);
        int startLayerIndex = endLayerIndex - 1;

        int endNeuronCount = model->neuronsPerLayer[endLayerIndex];
        int startNeuronCount = model->neuronsPerLayer[startLayerIndex];

        for (int endNeuronIndex = 0; endNeuronIndex < endNeuronCount; endNeuronIndex++) {
            double checkBiasDelta = checkBiasGradients[offsetEndLayerIndex][endNeuronIndex];

            printf("Check Δ Bias[%i][%i] = %lf\n", endLayerIndex, endNeuronIndex, checkBiasDelta);

            for (int startNeuronIndex = 0; startNeuronIndex < startNeuronCount; startNeuronIndex++) {
                double checkWeightDelta = checkWeightGradients[offsetEndLayerIndex][endNeuronIndex][startNeuronIndex];

                printf("Check Δ Weight[%i][%i][%i] = %lf\n", endLayerIndex, endNeuronIndex, startNeuronIndex, checkWeightDelta);
            }
        }
    }
}
#endif

/**
 * @param model The model which the parameter gradients will be based on.
 * @param layerIndex The layer index whose weight deltas are being calculated.
 * @param baseDelta The base delta, equal to change in the cost function over change in
 * the weighted sum of the neuron value.
 * @param weightGradients The weight gradient to fill.
 * @param biasGradients The bias gradient to fill.
 */
void updateParameterGradients(struct Model *model, const double* targetOutput, double** weightGradients[],
                              double* biasGradients[]) {
    int outputNeuronCount = model->neuronsPerLayer[OUTPUT_LAYER];

    // Entry indexed by [layerIndex][neuronIndex] gives
    // Δ C / Δ Z[layerIndex, neuronIndex]
    double* errors[NUMBER_OF_LAYERS];

    errors[OUTPUT_LAYER] = malloc(sizeof(double) * outputNeuronCount);

    // Fill errors of output layers
    for (int outputNeuronIndex = 0; outputNeuronIndex < outputNeuronCount; outputNeuronIndex++) {
        double outputNeuronValue = model->values[OUTPUT_LAYER][outputNeuronIndex];
        double targetOutputNeuronValue = targetOutput[outputNeuronIndex];

        // Δ C_outputNeuronIndex / Δ A[OUTPUT_LAYER][outputNeuronIndex]
        double firstErrorComponent = model->getCostDerivative(outputNeuronValue, targetOutputNeuronValue);
        // Δ A[OUTPUT_LAYER][outputNeuronIndex] / Δ Z[OUTPUT_LAYER][outputNeuronIndex]
        double secondErrorComponent = model->getActivation(outputNeuronValue);
        // Δ C_outputNeuronIndex / Δ Z[OUTPUT_LAYER][outputNeuronIndex]
        double error = firstErrorComponent * secondErrorComponent;

        errors[OUTPUT_LAYER][outputNeuronIndex] = error;
    }

    // Fill errors of non-output layers
    for (int endLayerIndex = OUTPUT_LAYER; endLayerIndex > INPUT_LAYER; endLayerIndex--) {
        int startLayerIndex = endLayerIndex - 1;
        int offsetEndLayerIndex = offsetLayer(endLayerIndex);

        int startNeuronsCount = model->neuronsPerLayer[startLayerIndex];
        int endNeuronsCount = model->neuronsPerLayer[endLayerIndex];

        errors[startLayerIndex] = malloc(sizeof(double) * startNeuronsCount);

        for (int startNeuronIndex = 0; startNeuronIndex < startNeuronsCount; startNeuronIndex++) {
            double error = 0.0;

            for (int endNeuronIndex = 0; endNeuronIndex < endNeuronsCount; endNeuronIndex++) {
                double nextError = errors[endLayerIndex][endNeuronIndex];
                double nextWeight = model->weights[offsetEndLayerIndex][endNeuronIndex][startNeuronIndex];

                double activationValue = model->values[startLayerIndex][startNeuronIndex];
                double activationValueDelta = model->getActivationChange(activationValue);

                double errorInfluence = nextWeight * nextError * activationValueDelta;
                error += errorInfluence;
            }

            // Take average of errors, not sum
            error /= endNeuronsCount;

            errors[startLayerIndex][startNeuronIndex] = error;
        }
    }

    // Update weights and biases of all layers based on errors
    for (int endLayerIndex = OUTPUT_LAYER; endLayerIndex > INPUT_LAYER; endLayerIndex--) {
        int offsetEndLaterIndex = offsetLayer(endLayerIndex);
        int startLayerIndex = endLayerIndex - 1;

        int endNeuronCount = model->neuronsPerLayer[endLayerIndex];
        int startNeuronCount = model->neuronsPerLayer[startLayerIndex];

        for (int endNeuronIndex = 0; endNeuronIndex < endNeuronCount; endNeuronIndex++) {
            for (int startNeuronIndex = 0; startNeuronIndex < startNeuronCount; startNeuronIndex++) {
                double errorOfEndNeuronOfWeight = errors[endLayerIndex][endNeuronIndex];

                double valueOfStartNeuron = model->values[startLayerIndex][startNeuronIndex];

                double biasGradientInfluence = errorOfEndNeuronOfWeight;
                double weightGradientInfluence = errorOfEndNeuronOfWeight * valueOfStartNeuron;

                biasGradientInfluence *= model->learningRate;
                weightGradientInfluence *= model->learningRate;

                weightGradients[offsetEndLaterIndex][endNeuronIndex][startNeuronIndex] += weightGradientInfluence;
                biasGradients[offsetEndLaterIndex][endNeuronIndex] += biasGradientInfluence;
            }
        }
    }
}

/**
 * Updates the weight and bias values within {@code model}, given the gradients of the cost function
 * with respect to the weights and biases.
 *
 * @param model
 * @param weightGradients
 * @param biasGradients
 */
void updateParameterValues(struct Model *model, double** weightGradients[], double* biasGradients[]) {
    for (int endLayerIndex = 1; endLayerIndex < NUMBER_OF_LAYERS; endLayerIndex++) {
        int offsetEndLayerIndex = offsetLayer(endLayerIndex);

        int endNeuronCount = model->neuronsPerLayer[endLayerIndex];
        int startNeuronCount = model->neuronsPerLayer[endLayerIndex - 1];

        for (int endNeuronIndex = 0; endNeuronIndex < endNeuronCount; endNeuronIndex++) {
            double biasDelta = biasGradients[offsetEndLayerIndex][endNeuronIndex];

            // update bias
            model->biases[offsetEndLayerIndex][endNeuronIndex] -= biasDelta;

#if PRINT_BIAS_UPDATE
            printf("Δ Bias[%i][%i] = %lf\n", endLayerIndex, endNeuronIndex, -biasDelta);
            printf("Bias[%i][%i] = %lf\n", endLayerIndex, endNeuronIndex, model->biases[offsetEndLayerIndex][endNeuronIndex]);
#endif

            for (int startNeuronIndex = 0; startNeuronIndex < startNeuronCount; startNeuronIndex++) {
                double weightDelta = weightGradients[offsetEndLayerIndex][endNeuronIndex][startNeuronIndex];

                // update weight
                model->weights[offsetEndLayerIndex][endNeuronIndex][startNeuronIndex] -= weightDelta;

#if PRINT_WEIGHT_UPDATE
                printf("Δ Weight[%i][%i][%i] = %lf\n", endLayerIndex, endNeuronIndex, startNeuronIndex, weightDelta);
                printf("Weight[%i][%i][%i] = %lf\n", endLayerIndex, endNeuronIndex, startNeuronIndex, model->weights[offsetEndLayerIndex][endNeuronIndex][startNeuronIndex]);
#endif
            }
        }
    }
}

static int epochIndex = 0;

/**
 * Allocates memory for the weight and bias gradients.
 *
 * @param model
 * @param weightGradients
 * @param biasGradients
 */
void initParameterGradients(struct Model* model, double** weightGradients[], double** biasGradients) {
    for (int layerIndex = 1; layerIndex < NUMBER_OF_LAYERS; layerIndex++) {
        int offsetLayerIndex = offsetLayer(layerIndex);

        int endNeuronCount = model->neuronsPerLayer[layerIndex];
        int startNeuronCount = model->neuronsPerLayer[layerIndex - 1];

        biasGradients[offsetLayerIndex] = malloc(sizeof(double) * endNeuronCount);
        weightGradients[offsetLayerIndex] = malloc(sizeof(double*) * endNeuronCount);

        for (int endNeuronIndex = 0; endNeuronIndex < endNeuronCount; endNeuronIndex++) {
            biasGradients[offsetLayerIndex][endNeuronIndex] = 0.0; // 1 1
            weightGradients[offsetLayerIndex][endNeuronIndex] = malloc(sizeof(double) * startNeuronCount);

            for (int startNeuronIndex = 0; startNeuronIndex < startNeuronCount; startNeuronIndex++)
                weightGradients[offsetLayerIndex][endNeuronIndex][startNeuronIndex] = 0.0;
        }
    }
}

/**
 * Feeds the input values of the entry into the input array given.
 *
 * @param input
 * @param entry
 * @param inputColumnIndices
 * @param inputColumnIndicesCount
 */
void initInput(double input[], const double entry[], const int inputColumnIndices[], int inputColumnIndicesCount) {
    for (int inputColumnIndex = 0; inputColumnIndex < inputColumnIndicesCount; inputColumnIndex++) {
        int inputColumn = inputColumnIndices[inputColumnIndex];
        input[inputColumnIndex] = entry[inputColumn];
    }
}

/**
 * Feeds the target output values of entry given into the target output array given.
 *
 * @param targetOutput
 * @param entry
 * @param outputColumnIndices
 * @param outputColumnIndicesCount
 */
void initTargetOutput(double targetOutput[], const double entry[], const int outputColumnIndices[], int outputColumnIndicesCount) {
    printf("Entry Input %lf Entry Output %lf\n", entry[0], entry[1]);

    for (int outputColumnIndex = 0; outputColumnIndex < outputColumnIndicesCount; outputColumnIndex++) {
        int outputColumn = outputColumnIndices[outputColumnIndex];
        targetOutput[outputColumnIndex] = entry[outputColumn];
    }
}

/**
 * Tests how well {@code model} fits {@code data}, placing the results into {@code predictedOutputs} and {@costs}.
 *
 * @param model
 * @param data
 * @param inputColumnIndices
 * @param outputColumnIndices
 * @param predictedOutputs
 * @param costs
 */
void test(struct Model* model, struct Data* data, int inputColumnIndices[], int outputColumnIndices[], double* predictedOutputs[], double costs[]) {
    int inputNeuronCount = model->neuronsPerLayer[INPUT_LAYER];
    int outputNeuronCount = model->neuronsPerLayer[OUTPUT_LAYER];

    for (int entryIndex = 0; entryIndex < data->numberOfEntries; entryIndex++) {
        double *entry = data->elements[entryIndex];

        double input[inputNeuronCount];
        double targetOutput[outputNeuronCount];

        initInput(input, entry, inputColumnIndices, inputNeuronCount);
        initTargetOutput(targetOutput, entry, outputColumnIndices, outputNeuronCount);

        // forward propagation
        propagateInputForward(model, input);
        double cost = 0.0;

        for (int outputIndex = 0; outputIndex < outputNeuronCount; outputIndex++) {
            double value = model->values[OUTPUT_LAYER][outputIndex];
            predictedOutputs[entryIndex][outputIndex] = value;

            double targetValue = targetOutput[outputIndex];
            cost += model->getCost(value, targetValue);
        }

        // Take average cost
        cost /= outputNeuronCount;

        costs[entryIndex] = cost;
    }
}

/**
 * Trains the model on the given data.
 *
 * @param model
 * @param data Container for the data the model will be trained on.
 * @param inputColumnIndices The indices of the columns within {@code data} that are the input columns.
 * @param outputColumnIndices The indices of the columns within {@code data} that are the output columns.
 */
void train(struct Model* model, struct Data* data, int inputColumnIndices[], int outputColumnIndices[]) {
    // [offsetLayerIndex][endNeuronIndex in layerIndex][startNeuronIndex in layerIndex - 1]
    double** weightGradients[NUMBER_OF_LAYERS - 1];
    // [offsetLayerIndex][endNeuronIndex]
    double* biasGradients[NUMBER_OF_LAYERS - 1];

    // Allocate the storage for the weight and bias deltas, in addition
    // to initializing them all weight and bias deltas with values of 0
    initParameterGradients(model, weightGradients, biasGradients);

#if GRADIENT_CHECKING
    // indexed same way as weightGradients and biasGradients
    double** checkWeightGradients[NUMBER_OF_LAYERS - 1];
    double* checkBiasGradients[NUMBER_OF_LAYERS - 1];

    initCheckParameterGradients(model, checkWeightGradients, checkBiasGradients);
#endif

    int inputNeuronCount = model->neuronsPerLayer[INPUT_LAYER];
    int outputNeuronCount = model->neuronsPerLayer[OUTPUT_LAYER];
    epochIndex++;

    // Feed each input into model
    for (int entryIndex = 0; entryIndex < data->numberOfEntries; entryIndex++) {
        double* entry = data->elements[entryIndex];

        double input[inputNeuronCount];
        double targetOutput[outputNeuronCount];

        // Feed values of entry into input and targetOutput given indices of input and output columns
        initInput(input, entry, inputColumnIndices, inputNeuronCount);
        initTargetOutput(targetOutput, entry, outputColumnIndices, outputNeuronCount);

        // forward propagation
        propagateInputForward(model, input);

#if PRINT_EPOCH_UPDATE
        double cost = 0.0;

        for (int outputIndex = 0; outputIndex < outputNeuronCount; outputIndex++) {
            double value = model->values[OUTPUT_LAYER][outputIndex];
            double targetValue = targetOutput[outputIndex];
            cost += model->getCost(value, targetValue);
        }

        printf("Epoch %i, Entry %i, Total Cost %lf, \t\tCost %lf\n", epochIndex, entryIndex, cost, cost / outputNeuronCount);
#endif

        // update weight and bias gradients based on this entry, part of the batch
        updateParameterGradients(model, targetOutput, weightGradients, biasGradients);

#if GRADIENT_CHECKING
        updateCheckParameterGradients(model, input, targetOutput, checkWeightGradients, checkBiasGradients);
#endif
    }

    // now that
    updateParameterValues(model, checkWeightGradients, checkBiasGradients);

#if GRADIENT_CHECKING
    printCheckParamterGradients(model, checkWeightGradients, checkBiasGradients);
#endif

    // free the memory taken by weight and bias gradients
    for (int layerIndex = 1; layerIndex < NUMBER_OF_LAYERS; layerIndex++) {
        int offsetLayerIndex = offsetLayer(layerIndex);

        int neuronCount = model->neuronsPerLayer[layerIndex];

        free(biasGradients[offsetLayerIndex]);

        for (int neuronIndex = 0; neuronIndex < neuronCount; neuronIndex++)
            free(weightGradients[offsetLayerIndex][neuronIndex]);
    }
}

/**
 * Allocates the memory for the parameters (weights and biases) of the model, in addition to initializing
 * them to their default values.
 *
 * @param model
 */
void initParameters(struct Model* model) {
    // initialize weights with arbitrary
    for (int endLayerIndex = 1; endLayerIndex < NUMBER_OF_LAYERS; endLayerIndex++) {
        int offsetEndLayerIndex = offsetLayer(endLayerIndex);

        int endNeuronCount = model->neuronsPerLayer[endLayerIndex];
        int startNeuronCount = model->neuronsPerLayer[endLayerIndex - 1];

        model->weights[offsetEndLayerIndex] = malloc(sizeof(double*) * endNeuronCount);

        for (int endNeuronIndex = 0; endNeuronIndex < endNeuronCount; endNeuronIndex++) {
            model->weights[offsetEndLayerIndex][endNeuronIndex] = malloc(sizeof(double) * startNeuronCount);
            model->biases[offsetEndLayerIndex] = malloc(sizeof(double) * endNeuronCount);

            for (int startNeuronIndex = 0; startNeuronIndex < startNeuronCount; startNeuronIndex++) {
                model->weights[offsetEndLayerIndex][endNeuronIndex][startNeuronIndex] = model->getInitialWeightValue(startNeuronCount, endNeuronCount);
                model->biases[offsetEndLayerIndex][endNeuronIndex] = model->getInitialBiasValue(startNeuronCount, endNeuronCount);
            }
        }
    }
}

/**
 * Allocayes the memory for the values of the model.
 *
 * @param model
 */
void initValues(struct Model* model) {
    for (int layerIndex = 0; layerIndex < NUMBER_OF_LAYERS; layerIndex++) {
        int neuronsInLayer = model->neuronsPerLayer[layerIndex];
        model->values[layerIndex] = malloc(sizeof(double) * neuronsInLayer);
    }
}