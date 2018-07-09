#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include "model.h"
#include "functions.h"

#define GRADIENT_CHECKING false
#define PRINT_WEIGHT_UPDATE false
#define PRINT_EPOCH_UPDATE false

/**
 * @param model
 * @param input The head to ann input array of size <code>model.neuronsPerLayer[INPUT_LAYER]</code> that has the inputs
 * of the model.
 */
void setInput(struct Model* model, double input[]) {
    model->values[INPUT_LAYER] = input;
}

void propagateInputForward(struct Model* model, double input[]) {
    setInput(model, input);

    for (int endLayerIndex = INPUT_LAYER + 1; endLayerIndex < model->numberOfLayers; endLayerIndex++) {
        int startLayerIndex = endLayerIndex - 1;

        int endNeuronCount = model->neuronsPerLayer[endLayerIndex];
        int startNeuronCount = model->neuronsPerLayer[startLayerIndex];

        for (int endNeuronIndex = 0; endNeuronIndex < endNeuronCount; endNeuronIndex++) {
            // Add bias at beginning, same as adding at end
            double weightedSum = model->biases[endLayerIndex][endNeuronIndex];

            for (int startNeuronIndex = 0; startNeuronIndex < startNeuronCount; startNeuronIndex++) {
                double weight = model->weights[endLayerIndex][endNeuronIndex][startNeuronIndex];
                double startNeuronValue = model->values[startLayerIndex][startNeuronIndex];

                double weightedInfluence = weight * startNeuronValue;
                weightedSum += weightedInfluence;
            }

            double activatedNeuronValue = model->getActivation(weightedSum);

            model->values[endLayerIndex][endNeuronIndex] = activatedNeuronValue;
        }
    }
}

#if GRADIENT_CHECKING
double getTotalCost(struct Model* model, const double targetOutputs[]) {
    const int outputLayerIndex = model->numberOfLayers - 1;

    int outputNeuronCount = model->neuronsPerLayer[outputLayerIndex];

    double totalCost = 0.0;

    for (int outputNeuronIndex = 0; outputNeuronIndex < outputNeuronCount; outputNeuronIndex++) {
        double outputNeuronValue = model->values[outputLayerIndex][outputNeuronIndex];
        double expectedOutputNeuronValue = targetOutputs[outputNeuronIndex];

        double cost = model->getCost(outputNeuronValue, expectedOutputNeuronValue);
        totalCost += cost;
    }

    return totalCost;
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
    const static float epsilon = 1e-5;

    double preChangeTotalCost = getTotalCost(model, targetOutput);

    for (int endLayerIndex = INPUT_LAYER + 1; endLayerIndex < model->numberOfLayers; endLayerIndex++) {
        int startLayerIndex = endLayerIndex - 1;

        int endNeuronCount = model->neuronsPerLayer[endLayerIndex];
        int startNeuronCount = model->neuronsPerLayer[startLayerIndex];

        for (int endNeuronIndex = 0; endNeuronIndex < endNeuronCount; endNeuronIndex++) {
            model->biases[endLayerIndex][endNeuronIndex] += epsilon;
            propagateInputForward(model, input);
            double postChangeTotalCostFromBias = getTotalCost(model, targetOutput);
            model->biases[endLayerIndex][endNeuronIndex] -= epsilon;

            double costDifferenceDueToBias = postChangeTotalCostFromBias - preChangeTotalCost;
            double checkBiasDeltaInfluence = costDifferenceDueToBias / epsilon;

            checkBiasGradients[endLayerIndex][endNeuronIndex] += checkBiasDeltaInfluence;

            for (int startNeuronIndex = 0; startNeuronIndex < startNeuronCount; startNeuronIndex++) {

                model->weights[endLayerIndex][endNeuronIndex][startNeuronIndex] += epsilon;
                propagateInputForward(model, input);
                double postChangeTotalCostFromWeight = getTotalCost(model, targetOutput);
                model->weights[endLayerIndex][endNeuronIndex][startNeuronIndex] -= epsilon;

                double costDifferenceDueToWeight = postChangeTotalCostFromWeight - preChangeTotalCost;
                double checkWeightDeltaInfluence = costDifferenceDueToWeight / epsilon;

                checkWeightGradient[endLayerIndex][endNeuronIndex][startNeuronIndex] += checkWeightDeltaInfluence;
            }
        }
    }
}

void updateParameterDifferences(struct Model *model, double ***weightGradients, double **biasGradients,
                                double ***checkWeightGradients, double **checkBiasGradients,
                                double ***weightDifferences, double **biasDifferences) {
    for (int endLayerIndex = INPUT_LAYER + 1; endLayerIndex < model->numberOfLayers; endLayerIndex++) {
        int endNeuronCount = model->neuronsPerLayer[endLayerIndex];

        int startLayerIndex = endLayerIndex - 1;
        int startNeuronCount = model->neuronsPerLayer[startLayerIndex];

        for (int endNeuronIndex = 0; endNeuronIndex < endNeuronCount; endNeuronIndex++) {
            double backPropagationBias = biasGradients[endLayerIndex][endNeuronIndex];
            double checkBias = checkBiasGradients[endLayerIndex][endNeuronIndex];

            double biasDifference = backPropagationBias - checkBias;

            biasDifferences[endLayerIndex][endNeuronIndex] = biasDifference;

            for (int startNeuronIndex = 0; startNeuronIndex < startNeuronCount; startNeuronIndex++) {
                double backPropagationWeight = weightGradients[endLayerIndex][endNeuronIndex][startNeuronIndex];
                double checkWeight = checkWeightGradients[endLayerIndex][endNeuronIndex][startNeuronIndex];

                double weightDifference = backPropagationWeight - checkWeight;

                weightDifferences[endLayerIndex][endNeuronIndex][startNeuronIndex] = weightDifference;
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
void updateParameterGradients(struct Model* model, const double* targetOutput, double** weightGradients[],
                              double* biasGradients[]) {
    const int outputLayerIndex = model->numberOfLayers - 1;

    int outputNeuronCount = model->neuronsPerLayer[outputLayerIndex];

    // Entry indexed by [layerIndex][neuronIndex] gives
    // Δ C / Δ Z[layerIndex, neuronIndex]
    double* errors[model->numberOfLayers];

    errors[outputLayerIndex] = malloc(sizeof(double) * outputNeuronCount);

    // Fill errors of output layers
    for (int outputNeuronIndex = 0; outputNeuronIndex < outputNeuronCount; outputNeuronIndex++) {
        double outputNeuronValue = model->values[outputLayerIndex][outputNeuronIndex];
        double targetOutputNeuronValue = targetOutput[outputNeuronIndex];

        // Δ C_outputNeuronIndex / Δ A[OUTPUT_LAYER][outputNeuronIndex]
        double firstErrorComponent = model->getCostPrime(outputNeuronValue, targetOutputNeuronValue);
        // Δ A[OUTPUT_LAYER][outputNeuronIndex] / Δ Z[OUTPUT_LAYER][outputNeuronIndex]
        double secondErrorComponent = model->getActivationPrime(outputNeuronValue);
        // Δ C_outputNeuronIndex / Δ Z[OUTPUT_LAYER][outputNeuronIndex]
        double error = firstErrorComponent * secondErrorComponent;

        errors[outputLayerIndex][outputNeuronIndex] = error;
    }

    // Fill errors of non-output layers
    for (int endLayerIndex = outputLayerIndex; endLayerIndex > INPUT_LAYER; endLayerIndex--) {
        int startLayerIndex = endLayerIndex - 1;

        int startNeuronsCount = model->neuronsPerLayer[startLayerIndex];
        int endNeuronsCount = model->neuronsPerLayer[endLayerIndex];

        errors[startLayerIndex] = malloc(sizeof(double) * startNeuronsCount);

        for (int startNeuronIndex = 0; startNeuronIndex < startNeuronsCount; startNeuronIndex++) {
            double error = 0.0;

            for (int endNeuronIndex = 0; endNeuronIndex < endNeuronsCount; endNeuronIndex++) {
                double nextError = errors[endLayerIndex][endNeuronIndex];
                double nextWeight = model->weights[endLayerIndex][endNeuronIndex][startNeuronIndex];

                double activationValue = model->values[startLayerIndex][startNeuronIndex];

                // Δ A[startLayerIndex][startNeuronIndex] / Δ Z[startLayerIndex][startNeuronIndex]
                double thirdComponent = model->getActivationPrime(activationValue);

                double errorInfluence = nextWeight * nextError * thirdComponent;
                error += errorInfluence;
            }

            errors[startLayerIndex][startNeuronIndex] = error;
        }
    }

    // Update weights and biases of all layers based on errors
    for (int endLayerIndex = outputLayerIndex; endLayerIndex > INPUT_LAYER; endLayerIndex--) {
        int startLayerIndex = endLayerIndex - 1;

        int endNeuronCount = model->neuronsPerLayer[endLayerIndex];
        int startNeuronCount = model->neuronsPerLayer[startLayerIndex];

        for (int endNeuronIndex = 0; endNeuronIndex < endNeuronCount; endNeuronIndex++) {
            double endNeuronError = errors[endLayerIndex][endNeuronIndex];

            double biasGradientInfluence = endNeuronError;
            //printf("BiasGC[%i][%i] = %lf\n", endLayerIndex, endNeuronIndex, biasGradientInfluence);
            biasGradients[endLayerIndex][endNeuronIndex] += biasGradientInfluence;

            for (int startNeuronIndex = 0; startNeuronIndex < startNeuronCount; startNeuronIndex++) {
                double startNeuronValue = model->values[startLayerIndex][startNeuronIndex];

                double weightGradientInfluence = endNeuronError * startNeuronValue;
                //printf("WeightGC[%i][%i][%i] = %lf\n", endLayerIndex, endNeuronIndex, startNeuronIndex, weightGradientInfluence);
                weightGradients[endLayerIndex][endNeuronIndex][startNeuronIndex] += weightGradientInfluence;
            }
        }
    }

    for (int endLayerIndex = INPUT_LAYER; endLayerIndex < model->numberOfLayers; endLayerIndex++) {
        double* layerErrors = errors[endLayerIndex];
        free(layerErrors);
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
void updateParameterValues(struct Model* model, double*** weightGradients, double** biasGradients, int batchSize) {
    for (int endLayerIndex = 1; endLayerIndex < model->numberOfLayers; endLayerIndex++) {
        int endNeuronCount = model->neuronsPerLayer[endLayerIndex];

        int startLayerIndex = endLayerIndex - 1;
        int startNeuronCount = model->neuronsPerLayer[startLayerIndex];

        for (int endNeuronIndex = 0; endNeuronIndex < endNeuronCount; endNeuronIndex++) {
            double biasDelta = biasGradients[endLayerIndex][endNeuronIndex];

            //printf("-T BiasGC[%i][%i] = %lf\n", endLayerIndex, endNeuronIndex, -biasDelta);

            biasDelta /= batchSize;
            biasDelta *= model->learningRate;

            // update bias
            model->biases[endLayerIndex][endNeuronIndex] -= biasDelta;

            //printf("-Δ Bias[%i][%i] = %lf\n", endLayerIndex, endNeuronIndex, -biasDelta);
            //printf("Bias[%i][%i] = %lf\n", endLayerIndex, endNeuronIndex, model->biases[endLayerIndex][endNeuronIndex]);

            for (int startNeuronIndex = 0; startNeuronIndex < startNeuronCount; startNeuronIndex++) {
                double weightDelta = weightGradients[endLayerIndex][endNeuronIndex][startNeuronIndex];
                //printf("-T WeightGC[%i][%i][%i] = %lf\n", endLayerIndex, endNeuronIndex, startNeuronIndex, -weightDelta);

                weightDelta /= batchSize;
                weightDelta *= model->learningRate;

                // update weight
                model->weights[endLayerIndex][endNeuronIndex][startNeuronIndex] -= weightDelta;

#if PRINT_WEIGHT_UPDATE
                printf("-Δ Weight[%i][%i][%i] = %lf\n", endLayerIndex, endNeuronIndex, startNeuronIndex, -weightDelta);
                printf("Weight[%i][%i][%i] = %lf\n", endLayerIndex, endNeuronIndex, startNeuronIndex, model->weights[endLayerIndex][endNeuronIndex][startNeuronIndex]);
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
void initializeGradients(struct Model* model, double*** weightGradients, double** biasGradients) {
    for (int endLayerIndex = INPUT_LAYER + 1; endLayerIndex < model->numberOfLayers; endLayerIndex++) {
        int endNeuronCount = model->neuronsPerLayer[endLayerIndex];

        int startLayerIndex = endLayerIndex - 1;
        int startNeuronCount = model->neuronsPerLayer[startLayerIndex];

        biasGradients[endLayerIndex] = malloc(sizeof(double) * endNeuronCount);
        weightGradients[endLayerIndex] = malloc(sizeof(double*) * endNeuronCount);

        for (int endNeuronIndex = 0; endNeuronIndex < endNeuronCount; endNeuronIndex++) {
            biasGradients[endLayerIndex][endNeuronIndex] = 0.0;
            weightGradients[endLayerIndex][endNeuronIndex] = malloc(sizeof(double) * startNeuronCount);

            for (int startNeuronIndex = 0; startNeuronIndex < startNeuronCount; startNeuronIndex++)
                weightGradients[endLayerIndex][endNeuronIndex][startNeuronIndex] = 0.0;
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
void test(struct Model* model, struct Data* data, int inputColumnIndices[], int outputColumnIndices[], double** predictedOutputs, double costs[]) {
    const int outputLayerIndex = model->numberOfLayers - 1;

    int inputNeuronCount = model->neuronsPerLayer[INPUT_LAYER];
    int outputNeuronCount = model->neuronsPerLayer[outputLayerIndex];

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
            double value = model->values[outputLayerIndex][outputIndex];
            predictedOutputs[entryIndex][outputIndex] = value;

            double targetValue = targetOutput[outputIndex];
            cost += model->getCost(value, targetValue);
        }

        // Take average cost
        cost /= outputNeuronCount;

        costs[entryIndex] = cost;
    }
}

void freeGradients(struct Model* model, double*** weightGradients, double** biasGradients) {
    for (int endLayerIndex = INPUT_LAYER + 1; endLayerIndex < model->numberOfLayers; endLayerIndex++) {
        double* layerBiases = biasGradients[endLayerIndex];

        free(layerBiases);

        int endNeuronCount = model->neuronsPerLayer[endLayerIndex];

        for (int neuronIndex = 0; neuronIndex < endNeuronCount; neuronIndex++) {
            double* neuronWeights = weightGradients[endLayerIndex][neuronIndex];
            free(neuronWeights);
        }

        double** layerWeights = weightGradients[endLayerIndex];

        free(layerWeights);
    }
}

/**
 * Trains the model on the given data.
 *
 * @param model
 * @param data Container for the data the model will be trained on.
 * @param inputColumnIndices The indices of the columns within {@code data} that are the input columns.
 * @param targetOutputColumnIndices The indices of the columns within {@code data} that are the output columns.
 */
void train(struct Model* model, struct Data* data, int inputColumnIndices[], int targetOutputColumnIndices[]) {
    const int outputLayerIndex = model->numberOfLayers - 1;

    const int inputNeuronCount = model->neuronsPerLayer[INPUT_LAYER];
    const int outputNeuronCount = model->neuronsPerLayer[outputLayerIndex];

    // For both weightGradients and biasGradients, index 0 is not occupied.
    // [endLayerIndex][endNeuronIndex in layerIndex][startNeuronIndex in layerIndex - 1]
    double** weightGradients[model->numberOfLayers];
    // [endLayerIndex][endNeuronIndex]
    double* biasGradients[model->numberOfLayers];

    // Allocate the storage for the weight and bias deltas, in addition
    // to initializing them all weight and bias deltas with values of 0
    initializeGradients(model, weightGradients, biasGradients);

#if GRADIENT_CHECKING
    // indexed same way as weightGradients and biasGradients
    double** checkWeightGradients[model->numberOfLayers];
    double* checkBiasGradients[model->numberOfLayers];

    initializeGradients(model, checkWeightGradients, checkBiasGradients);

    double** weightGradientsDifference[model->numberOfLayers];
    double* biasGradientsDifference[model->numberOfLayers];

    initializeGradients(model, weightGradientsDifference, biasGradientsDifference);
#endif

    epochIndex++;

    // Feed each input into model
    for (int entryIndex = 0; entryIndex < data->numberOfEntries; entryIndex++) {
        double* entry = data->elements[entryIndex];

        double input[inputNeuronCount];
        double targetOutput[outputNeuronCount];

        // Feed values of entry into input and targetOutput given indices of input and output columns
        initInput(input, entry, inputColumnIndices, inputNeuronCount);
        initTargetOutput(targetOutput, entry, targetOutputColumnIndices, outputNeuronCount);

        // forward propagation
        propagateInputForward(model, input);

#if PRINT_EPOCH_UPDATE
        double cost = 0.0;

        for (int outputNeuronIndex = 0; outputNeuronIndex < outputNeuronCount; outputNeuronIndex++) {
            double value = model->values[outputLayerIndex][outputNeuronIndex];
            double targetValue = targetOutput[outputNeuronIndex];
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

#if GRADIENT_CHECKING
    updateParameterDifferences(model, weightGradients, biasGradients,
                               checkWeightGradients, checkBiasGradients,
                               weightGradientsDifference, biasGradientsDifference);

    for (int endLayerIndex = INPUT_LAYER + 1; endLayerIndex < model->numberOfLayers; endLayerIndex++) {
        int endNeuronCount = model->neuronsPerLayer[endLayerIndex];

        int startLayerIndex = endLayerIndex - 1;
        int startNeuronCount = model->neuronsPerLayer[startLayerIndex];

        for (int endNeuronIndex = 0; endNeuronIndex < endNeuronCount; endNeuronIndex++) {
            double biasDifference = biasGradientsDifference[endLayerIndex][endNeuronIndex];
            printf("BiasDifference[%i][%i] = %lf\n", endLayerIndex, endNeuronIndex, biasDifference);

            for (int startNeuronIndex = 0; startNeuronIndex < startNeuronCount; startNeuronIndex++) {
                double weightDifference = weightGradientsDifference[endLayerIndex][endNeuronIndex][startNeuronIndex];
                printf("WeightDifference[%i][%i][%i] = %lf\n", endLayerIndex, endNeuronIndex, startNeuronIndex, weightDifference);
            }
        }
    }
#endif

    updateParameterValues(model, weightGradients, biasGradients, data->numberOfEntries);

#if GRADIENT_CHECKING
    //printCheckParameterGradients(model, checkWeightGradients, checkBiasGradients);
#endif

    freeGradients(model, weightGradients, biasGradients);

#if GRADIENT_CHECKING
    freeGradients(model, checkWeightGradients, checkBiasGradients);
    freeGradients(model, weightGradientsDifference, biasGradientsDifference);
#endif
}

/**
 * Allocates the memory for the parameters (weights and biases) of the model, in addition to initializing
 * them to their default values.
 *
 * @param model
 */
void initParameters(struct Model* model) {
    model->weights = malloc(sizeof(double**) * model->numberOfLayers);
    model->biases = malloc(sizeof(double*) * model->numberOfLayers);

    for (int endLayerIndex = INPUT_LAYER + 1; endLayerIndex < model->numberOfLayers; endLayerIndex++) {
        int endNeuronCount = model->neuronsPerLayer[endLayerIndex];

        int startLayerIndex = endLayerIndex - 1;
        int startNeuronCount = model->neuronsPerLayer[startLayerIndex];

        model->weights[endLayerIndex] = malloc(sizeof(double*) * endNeuronCount);
        model->biases[endLayerIndex] = malloc(sizeof(double) * endNeuronCount);

        for (int endNeuronIndex = 0; endNeuronIndex < endNeuronCount; endNeuronIndex++) {

            model->weights[endLayerIndex][endNeuronIndex] = malloc(sizeof(double) * startNeuronCount);

            for (int startNeuronIndex = 0; startNeuronIndex < startNeuronCount; startNeuronIndex++) {
                model->weights[endLayerIndex][endNeuronIndex][startNeuronIndex] = model->getInitialWeightValue(startNeuronCount, endNeuronCount);
                model->biases[endLayerIndex][endNeuronIndex] = model->getInitialBiasValue(startNeuronCount, endNeuronCount);
            }
        }
    }
}

void freeParameters(struct Model* model) {
    for (int endLayerIndex = INPUT_LAYER + 1; endLayerIndex < model->numberOfLayers; endLayerIndex++) {
        int endNeuronCount = model->neuronsPerLayer[endLayerIndex];

        double* layerBiases = model->biases[endLayerIndex];
        free(layerBiases);

        for (int endNeuronIndex = 0; endNeuronIndex < endNeuronCount; endNeuronIndex++) {
            double* neuronWeights = model->weights[endLayerIndex][endNeuronIndex];
            free(neuronWeights);
        }

        double** layerWeights = model->weights[endLayerIndex];
        free(layerWeights);
    }

    free(model->biases);
    free(model->weights);
}

void initValues(struct Model* model) {
    model->values = malloc(sizeof(double*) * model->numberOfLayers);

    for (int layerIndex = 0; layerIndex < model->numberOfLayers; layerIndex++) {
        int neuronsInLayer = model->neuronsPerLayer[layerIndex];
        model->values[layerIndex] = malloc(sizeof(double) * neuronsInLayer);
    }
}

void freeValues(struct Model* model) {
    for (int layerIndex = 0; layerIndex < model->numberOfLayers; layerIndex++) {
        double* layerValues = model->values[layerIndex];
        free(layerValues);
    }

    free(model->values);
}

void initializeModel(struct Model* model, int neuronsPerLayer[], int numberOfLayers, double learningRate, ActivationFunctionType activationFunctionType) {
    switch (activationFunctionType) {
        case SIGMOID: {
            model->getActivation = getSigmoid;
            model->getActivationPrime = getSigmoidPrime;
            break;
        }

        case RELU: {
            model->getActivation = getReLU;
            model->getActivationPrime = getReLUPrime;
            break;
        }

        case TANH: {
            model->getActivation = getTanH;
            model->getActivationPrime = getTanH;
            break;
        }
    }

    model->getCost = getCost;
    model->getCostPrime = getCostPrime;
    model->learningRate = learningRate;
    model->numberOfLayers = numberOfLayers;
    model->neuronsPerLayer = neuronsPerLayer;
    model->getInitialBiasValue = getInitialBias;
    model->getInitialWeightValue = getInitialRandomWeight;

    initParameters(model);
    initValues(model);
}