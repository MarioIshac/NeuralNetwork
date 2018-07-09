#include <stdio.h>
#include <stdlib.h>
#include <zconf.h>
#include <time.h>
#include "model.h"
#include "functions.h"
#include "data.h"

#define EPOCH_COUNT 100000

#define PRINT_TEST_RESULTS true

int main() {
    time_t currentTime;
    time(&currentTime);
    srand(currentTime);

    struct Model model = {
        .neuronsPerLayer = {2, 2, 1},
        .learningRate = 1,

        // Default values
        .getActivation = getSigmoid,
        .getActivationDerivative = getSigmoidPrime,
        .getCost = getCost,
        .getCostDerivative = getCostPrime,
        .getInitialWeightValue = getInitialRandomWeight,
        .getInitialBiasValue = getInitialBias,
    };

    int numberOfInputs = model.neuronsPerLayer[INPUT_LAYER];
    int numberOfOutputs = model.neuronsPerLayer[OUTPUT_LAYER];

    // Change working directory so data can be referenced relative to parent data folder
    chdir("..");

    struct Data trainData;
    fill(&trainData, "data/xor/train.csv");

    struct Data testData;
    fill(&testData, "data/xor/test.csv");

    int inputColumnIndices[numberOfInputs];
    int outputColumnIndices[numberOfOutputs];

    inputColumnIndices[0] = 0;
    inputColumnIndices[1] = 1;
    outputColumnIndices[0] = 2;

    initValues(&model);
    initParameters(&model);

    for (int epochIndex = 0; epochIndex < EPOCH_COUNT; epochIndex++)
        train(&model, &trainData, inputColumnIndices, outputColumnIndices);

    // Testing
    double* predictedOutputs[testData.numberOfEntries];
    for (int predictedOutputIndex = 0; predictedOutputIndex < testData.numberOfEntries; predictedOutputIndex++)
        predictedOutputs[predictedOutputIndex] = malloc(sizeof(double) * numberOfOutputs);

    double costs[testData.numberOfEntries];

    test(&model, &testData, inputColumnIndices, outputColumnIndices, predictedOutputs, costs);

    for (int entryIndex = 0; entryIndex < testData.numberOfEntries; entryIndex++) {
        double* entry = testData.elements[entryIndex];

        double inputs[numberOfInputs];
        double targetOutputs[numberOfOutputs];

        initInput(inputs, entry, inputColumnIndices, numberOfInputs);
        initTargetOutput(targetOutputs, entry, outputColumnIndices, numberOfOutputs);

#if PRINT_TEST_RESULTS
        printf("Inputs =");

        for (int inputIndex = 0; inputIndex < numberOfInputs; inputIndex++) {
            double input = inputs[inputIndex];
            printf(" %lf", input);
        }

        printf(", Target Outputs =");

        for (int outputIndex = 0; outputIndex < numberOfOutputs; outputIndex++) {
            double targetOutput = targetOutputs[outputIndex];
            printf(" %lf", targetOutput);
        }

        printf(", Predicted Outputs =");

        for (int outputIndex = 0; outputIndex < numberOfOutputs; outputIndex++) {
            double predictedOutput = predictedOutputs[entryIndex][outputIndex];
            printf(" %lf", predictedOutput);
        }

        printf(".\n");
#endif
    }

    for (int predictedOutputIndex = 0; predictedOutputIndex < testData.numberOfEntries; predictedOutputIndex++) {
        double* predictedOutput = predictedOutputs[predictedOutputIndex];

        free(predictedOutput);
    }

    freeColumnNames(&trainData);
    freeElements(&trainData);

    freeColumnNames(&testData);
    freeElements(&testData);

    freeValues(&model);
    freeParameters(&model);

    return 0;
}

