#include <stdio.h>
#include <stdlib.h>
#include "model.h"
#include "functions.h"

/**
 * @param angle The angle in degrees that the car is turning at, relative to the direction it is going at.
 * @return The factor to be applied to the car's speed when turning at this angle. In other words, the speed drop
 * off is (1 - this factor).
 */
double getSpeedFactor(double angle) {
    return 1 - angle / 180;
}

int main() {
    struct Model model = {
        .getSpeedFactor = getSpeedFactor,
        .neuronsPerLayer = {1, 3, 1},
    };

    int numberOfInputs = model.neuronsPerLayer[INPUT_LAYER];
    int numberOfOutputs = model.neuronsPerLayer[OUTPUT_LAYER];

    double **inputs = malloc(sizeof(double*) * NUMBER_OF_TRAINING_FEATURE_VECTORS);

    for (int inputIndex = 0; inputIndex < NUMBER_OF_TRAINING_FEATURE_VECTORS; inputIndex++)
        inputs[inputIndex] = malloc(sizeof(double) * numberOfInputs);

    inputs[0][0] = 1;
    inputs[1][0] = 4;

    double **targetOutputs = malloc(sizeof(double*) * NUMBER_OF_TRAINING_FEATURE_VECTORS);

    for (int targetOutputIndex = 0; targetOutputIndex < NUMBER_OF_TRAINING_FEATURE_VECTORS; targetOutputIndex++)
        targetOutputs[targetOutputIndex] = malloc(sizeof(double) * numberOfOutputs);

    targetOutputs[0][0] = true;
    targetOutputs[1][0] = false;

    for (int layerIndex = 0; layerIndex < NUMBER_OF_LAYERS; layerIndex++) {
        int neuronsInLayer = model.neuronsPerLayer[layerIndex];
        model.values[layerIndex] = malloc(sizeof(double) * neuronsInLayer);
    }

    // initialize weights with arbitrary
    for (int layerIndex = 1; layerIndex < NUMBER_OF_LAYERS; layerIndex++) {
        int offsetLayerIndex = offsetLayer(layerIndex);

        int endNeuronCount = model.neuronsPerLayer[layerIndex];
        int startNeuronCount = model.neuronsPerLayer[layerIndex - 1];

        model.weights[offsetLayerIndex] = malloc(sizeof(double*) * endNeuronCount);

        for (int endNeuronIndex = 0; endNeuronIndex < endNeuronCount; endNeuronIndex++) {

            model.weights[offsetLayerIndex][endNeuronIndex] = malloc(sizeof(double) * startNeuronCount);
            model.biases[offsetLayerIndex] = malloc(sizeof(double) * endNeuronCount);

            for (int startNeuronIndex = 0; startNeuronIndex < startNeuronCount; startNeuronIndex++) {
                model.weights[offsetLayerIndex][endNeuronIndex][startNeuronIndex] = getDefaultWeightValue(startNeuronCount);
                model.biases[offsetLayerIndex][endNeuronIndex] = 0.0;
            }
        }
    }

    int j = 0;

    while (j++ < 1000) {
        ////printf("Epoch %i", j);
        train(&model, inputs, targetOutputs, NUMBER_OF_TRAINING_FEATURE_VECTORS);
    }

    exit(0);
}

