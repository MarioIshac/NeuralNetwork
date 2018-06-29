#ifndef NEURALNETWORK_MODEL_H
#define NEURALNETWORK_MODEL_H

#include <stdbool.h>
#include "data.h"

#define NUMBER_OF_LAYERS 3
#define INPUT_LAYER 0
#define OUTPUT_LAYER (NUMBER_OF_LAYERS - 1)
#define NUMBER_OF_TRAINING_FEATURE_VECTORS 2

typedef double (*CostFunction)(double, double);
typedef double (*ActivationFunction)(double);
typedef double (*WeightInitializationFunction)(double, double);
typedef double (*BiasInitializingFunction)(double, double);


/**
 * Represents a model to be used in regards to training the vehicle to move efficiently.
 * Implemented as a neural network.
 */
struct Model {
    /* Weight of link between two neurons is received with indices
     * [layer index of end neuron][index of end neuron within its layer][index of start neuron within its layer] */
    double** weights[NUMBER_OF_LAYERS];
    double* biases[NUMBER_OF_LAYERS];
    double* values[NUMBER_OF_LAYERS];

    int neuronsPerLayer[NUMBER_OF_LAYERS];
    double learningRate;

    /**
     * the function used to activate the neuron. The value returned by this function is put into the neuron.
     */
    ActivationFunction getActivation;

    /**
     * the derivative of the function used to activate the neuron.
     */
    ActivationFunction getActivationDerivative;

    /**
     * the function uses to calculate the cost, given an output neuron's value and its target value
     */
    CostFunction getCost;

    /**
     * the function used to calculate the derivative of the cost with respect to the output neuron's value, given the
     * output neuron's value and its target value
     */
    CostFunction getCostDerivative;

    WeightInitializationFunction getInitialWeightValue;
    BiasInitializingFunction getInitialBiasValue;
};

void train(struct Model* model, struct Data* data, int inputColumnIndices[], int outputColumnIndices[]);
void test(struct Model* model, struct Data* data, int inputColumnIndices[], int outputColumnIndices[], double* predictedOutputs[], double costs[]);
void compute(struct Model* model, struct Data* data, int inputColumnIndices[], double cost[]);
void initParameters(struct Model* model);
void initValues(struct Model* model);

void initInput(double input[], const double entry[], const int inputColumnIndices[], int inputColumnIndicesCount);
void initTargetOutput(double targetOutput[], const double entry[], const int targetOutputIndices[], int targetOutputIndicesCount);

#endif
