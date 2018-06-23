//
// Created by mario on 6/19/18.
//

#ifndef NEURALNETWORK_MODEL_H
#define NEURALNETWORK_MODEL_H

#include <stdbool.h>

#define NUMBER_OF_LAYERS 3
#define INPUT_LAYER 0
#define OUTPUT_LAYER (NUMBER_OF_LAYERS - 1)
#define NUMBER_OF_TRAINING_FEATURE_VECTORS 2

/**
 * Represents a model to be used in regards to training the vehicle to move efficiently.
 * Implemented as a neural network.
 */
struct Model {
    /* Weight of link between two neurons is received with indices
     * [layer index of end neuron][index of end neuron within its layer][index of start neuron within its layer] */
    double** weights[NUMBER_OF_LAYERS - 1];
    double* biases[NUMBER_OF_LAYERS - 1];
    double* values[NUMBER_OF_LAYERS];

    int neuronsPerLayer[NUMBER_OF_LAYERS];

    /**
     * @param angle The angle in degrees that the car is turning at, relative to the direction it is going at.
     * @return The factor to be applied to the car's speed when turning at this angle. In other words, the speed drop
     * off is (1 - this factor).
     */
    double (*getSpeedFactor)(double angle);
};

void train(struct Model* model, double* input[], double* targetOutputs[], int inputSize);
int offsetLayer(int layerIndex);
int unoffsetLayer(int layerIndex);

#endif
