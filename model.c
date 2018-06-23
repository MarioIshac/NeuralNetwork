//
// Created by mario on 6/19/18.
//

#include <stdlib.h>
#include <stdio.h>
#include "model.h"
#include "functions.h"

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

void getOutput(struct Model* model, double outputNeuronValues[]) {
    int outputNeuronCount = model->neuronsPerLayer[OUTPUT_LAYER];

    for (int outputNeuronIndex = 0; outputNeuronIndex < outputNeuronCount; outputNeuronIndex++) {
        double outputNeuronValue = model->values[OUTPUT_LAYER][outputNeuronIndex];
        outputNeuronValues[outputNeuronIndex] = outputNeuronValue ;
    }
}

/**
 * @param model
 * @param layerIndex The layer index whose weight deltas are being calculated.
 * @param baseDelta The base delta, equal to change in the cost function over change in
 * the weighted sum of the neuron value.
 * @param weightGradients The weight gradient to fill.
 * @param biasGradients The bias gradient to fill.
 */
void updateParameterDeltas(struct Model *model, const double *targetOutput, double ***weightGradients, double **biasGradients) {
    int outputNeuronCount = model->neuronsPerLayer[OUTPUT_LAYER];

    // Entry indexed by [outputNeuronIndex][layerIndex][neuronIndex] gives
    // Δ C[outputNeuronIndex] / Δ Z[layerIndex, neuronIndex]
    double* errors[outputNeuronCount][NUMBER_OF_LAYERS];

    // Fill errors of output layers
    for (int outputNeuronIndex = 0; outputNeuronIndex < outputNeuronCount; outputNeuronIndex++) {
        errors[outputNeuronIndex][OUTPUT_LAYER] = malloc(sizeof(double) * outputNeuronCount);

        double outputNeuronValue = model->values[OUTPUT_LAYER][outputNeuronIndex];
        double targetOutputNeuronValue = targetOutput[outputNeuronIndex];

        // Δ C[outputNeuronIndex] / Δ A[OUTPUT_LAYER][outputNeuronIndex]
        double firstErrorComponent = getCostDerivative(outputNeuronValue, targetOutputNeuronValue);
        // Δ A[OUTPUT_LAYER][outputNeuronIndex] / Δ Z[OUTPUT_LAYER][outputNeuronIndex]
        double secondErrorComponent = getActivationDerivative(outputNeuronValue);
        // Δ C[outputNeuronIndex] / Δ Z[OUTPUT_LAYER][outputNeuronIndex]
        double error = firstErrorComponent * secondErrorComponent;

        errors[outputNeuronIndex][OUTPUT_LAYER][outputNeuronIndex] = error;
    }

    // Fill errors of non-output layers
    for (int startLayerIndex = OUTPUT_LAYER - 1; startLayerIndex > INPUT_LAYER; startLayerIndex--) {
        int endLayerIndex = startLayerIndex + 1;
        int offsetEndLayerIndex = offsetLayer(endLayerIndex);

        int startNeuronsCount = model->neuronsPerLayer[startLayerIndex];
        int endNeuronsCount = model->neuronsPerLayer[endLayerIndex];

        for (int outputNeuronIndex = 0; outputNeuronIndex < outputNeuronCount; outputNeuronIndex++) {
            errors[outputNeuronIndex][startLayerIndex] = malloc(sizeof(double) * startNeuronsCount);

            for (int startNeuronIndex = 0; startNeuronIndex < startNeuronsCount; startNeuronIndex++) {
                double error = 0.0;

                for (int endNeuronIndex = 0; endNeuronIndex < endNeuronsCount; endNeuronIndex++) {
                    double nextError = errors[outputNeuronIndex][endLayerIndex][endNeuronIndex];
                    double nextWeight = model->weights[offsetEndLayerIndex][endNeuronIndex][startNeuronIndex];

                    double value = model->values[startLayerIndex][startNeuronIndex];
                    double deltaValue = getActivationDerivative(value);

                    double errorInfluence = nextWeight * nextError * deltaValue;
                    error += errorInfluence;
                }

                errors[outputNeuronIndex][startLayerIndex][startNeuronIndex] = error;
            }
        }
    }

    for (int endLayerIndex = OUTPUT_LAYER; endLayerIndex > INPUT_LAYER; endLayerIndex--) {
        int offsetEndLaterIndex = offsetLayer(endLayerIndex);
        int startLayerIndex = endLayerIndex - 1;

        int endNeuronCount = model->neuronsPerLayer[endLayerIndex];
        int startNeuronCount = model->neuronsPerLayer[startLayerIndex];

        for (int outputNeuronIndex = 0; outputNeuronIndex < outputNeuronCount; outputNeuronIndex++) {
            for (int endNeuronIndex = 0; endNeuronIndex < endNeuronCount; endNeuronIndex++) {
                for (int startNeuronIndex = 0; startNeuronIndex < startNeuronCount; startNeuronIndex++) {
                    double errorOfEndNeuronOfWeight = errors[outputNeuronIndex][endLayerIndex][endNeuronIndex];

                    double valueOfStartNeuron = model->values[startLayerIndex][startNeuronIndex];

                    double biasGradientInfluence = errorOfEndNeuronOfWeight;
                    double weightGradientInfluence = errorOfEndNeuronOfWeight * valueOfStartNeuron;

                    weightGradients[offsetEndLaterIndex][endNeuronIndex][startNeuronIndex] += weightGradientInfluence;
                    biasGradients[offsetEndLaterIndex][endNeuronIndex] += biasGradientInfluence;
                }
            }
        }
    }
}

void updateWeightsAndBiases(struct Model* model, double** weightDeltas[], double* biasDeltas[]) {
    for (int endLayerIndex = 1; endLayerIndex < NUMBER_OF_LAYERS; endLayerIndex++) {
        int offsetEndLayerIndex = offsetLayer(endLayerIndex);

        int endNeuronCount = model->neuronsPerLayer[endLayerIndex];
        int startNeuronCount = model->neuronsPerLayer[endLayerIndex - 1];

        for (int endNeuronIndex = 0; endNeuronIndex < endNeuronCount; endNeuronIndex++) {
            double biasDelta = biasDeltas[offsetEndLayerIndex][endNeuronIndex];

            // update bias
            model->biases[offsetEndLayerIndex][endNeuronIndex] -= biasDelta;
            //printf("Δ Bias[%i][%i] = %lf\n", endLayerIndex, endNeuronIndex, -biasDelta);

            for (int startNeuronIndex = 0; startNeuronIndex < startNeuronCount; startNeuronIndex++) {
                double weightDelta = weightDeltas[offsetEndLayerIndex][endNeuronIndex][startNeuronIndex];

                // update weight
                model->weights[offsetEndLayerIndex][endNeuronIndex][startNeuronIndex] -= weightDelta;
                //printf("Δ Weight[%i][%i][%i] = %lf\n", endLayerIndex, endNeuronIndex, startNeuronIndex, weightDelta);

            }
        }
    }
}

/**
 * @see fillGradientRecursively
 */
void updateParameterDeltasPerOutputNeuron(struct Model *model, const double *targetOutput, double ***weightGradients, double **biasGradients) {
    int outputNeuronCount = model->neuronsPerLayer[OUTPUT_LAYER];

    for (int outputNeuronIndex = 0; outputNeuronIndex < outputNeuronCount; outputNeuronIndex++)
        updateParameterDeltas(model, targetOutput, weightGradients, biasGradients);
}

void propagateInputForward(struct Model* model, double* inputHead) {
    setInput(model, inputHead);

    for (int layerIndex = 1; layerIndex < NUMBER_OF_LAYERS; layerIndex++) {
        int neuronsInLayer = model->neuronsPerLayer[layerIndex];
        int neuronsInPreviousLayer = model->neuronsPerLayer[layerIndex - 1];

        for (int neuronIndex = 0; neuronIndex < neuronsInLayer; neuronIndex++) {
            double weightedSum = 0.0;
            double bias = model->biases[offsetLayer(layerIndex)][neuronIndex];

            for (int previousNeuronIndex = 0; previousNeuronIndex < neuronsInPreviousLayer; previousNeuronIndex++) {
                double weightOfLink = model->weights[offsetLayer(layerIndex)][neuronIndex][previousNeuronIndex];
                double previousNeuronValue = model->values[layerIndex - 1][previousNeuronIndex];

                double weightedInfluence = weightOfLink * previousNeuronValue + bias;
                weightedSum += weightedInfluence;
            }

            double activatedNeuronValue = getActivation(weightedSum);

            //printf("Neuron[%i][%i] - z: %lf, a: %lf\n", layerIndex, neuronIndex, weightedSum, activatedNeuronValue);
            model->values[layerIndex][neuronIndex] = activatedNeuronValue;
        }
    }
}

static int i = 0;

/**
 * Passes the input into the model,
 *
 * @param model
 * @param input An input array of size <code>model.neuronsPerLayer[INPUT_LAYER]</code> that has the inputs
 * of the model.
 */
void train(struct Model* model, double** input, double** targetOutputs, int inputSize) {
    // [offsetLayerIndex][endNeuronIndex in layerIndex][startNeuronIndex in layerIndex - 1]
    double **weightGradients[NUMBER_OF_LAYERS - 1];
    // [offsetLayerIndex][endNeuronIndex]
    double *biasGradients[NUMBER_OF_LAYERS - 1];

    // Allocate the storage for the weight and bias deltas, in addition
    // to initializing them all weight and bias deltas with values of 0
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
    i++;

    // Feed each input into model
    for (int inputIndex = 0; inputIndex < inputSize; inputIndex++) {
        double* targetOutput = targetOutputs[inputIndex];

        double* inputHead = input[inputIndex];

        //printf("Inputs are ");
        for (int a = 0; a < model->neuronsPerLayer[INPUT_LAYER]; a++) {
            //printf(" %lf ", inputHead[a]);
        }
        //printf(".\n");

        // forward propagation
        propagateInputForward(model, inputHead);
        double cost = 0.0;

        for (int outputIndex = 0; outputIndex < model->neuronsPerLayer[OUTPUT_LAYER]; outputIndex++) {
            double value = model->values[OUTPUT_LAYER][outputIndex];
            cost += getCost(value, targetOutput[outputIndex]);

        }
        printf("Epoch %i, Feature Vector Index %i: %lf\n", i, inputIndex, cost);


        updateParameterDeltasPerOutputNeuron(model, targetOutput, weightGradients, biasGradients);

        for (int outputNeuronIndex = 0; outputNeuronIndex < model->neuronsPerLayer[OUTPUT_LAYER]; outputNeuronIndex++) {}
            //printf("Output Neuron Value at Index %i is %lf with target value %lf.\n", outputNeuronIndex, model->values[OUTPUT_LAYER][outputNeuronIndex], targetOutput[outputNeuronIndex]);
    }

    updateWeightsAndBiases(model, weightGradients, biasGradients);
    //printf("\n");

    // free the memory taken by weight and bias gradients
    for (int layerIndex = 1; layerIndex < NUMBER_OF_LAYERS; layerIndex++) {
        int offsetLayerIndex = offsetLayer(layerIndex);

        int neuronCount = model->neuronsPerLayer[layerIndex];

        free(biasGradients[offsetLayerIndex]);

        for (int neuronIndex = 0; neuronIndex < neuronCount; neuronIndex++)
            free(weightGradients[offsetLayerIndex][neuronIndex]);
    }
}



/*
 * for layer in layersn
 *      for weights in layer
 *          for layer in rev(layers)
 *              change in cost / change in neuron value
 *              change in neuron value / change in weighted sum
 *              change in weighted sum / change in weight
 *              change in weight / change in neuron value
 *              change in neuron value / change in weighted sum
 *              change in weighted sum / change in weight
 *              change in weight ...
 *
 *
 *
 *
 *
 */