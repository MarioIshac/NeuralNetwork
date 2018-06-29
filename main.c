#include <stdio.h>
#include <stdlib.h>
#include <zconf.h>
#include <time.h>
#include "model.h"
#include "functions.h"
#include "data.h"

#define EPOCH_COUNT 20000
#define NUMBER_OF_COLUMNS 3
#define TRAIN_ENTRIES_SIZE 4
#define TEST_ENTRIES_SIZE 4

#define PRINT_TEST_RESULTS true

int main() {
    time_t currentTime;
    time(&currentTime);
    srand(currentTime);

    struct Model model = {
            .neuronsPerLayer = {2, 2, 1},
            .learningRate = 0.02,

            // Default values
            .getActivation = applySigmoid,
            .getActivationDerivative = applySigmoidDerivative,
            .getCost = getCost,
            .getCostDerivative = getCostDerivative,
            .getInitialWeightValue = getInitialRandomWeight,
            .getInitialBiasValue = getInitialBias,
    };

    int numberOfInputs = model.neuronsPerLayer[INPUT_LAYER];
    int numberOfOutputs = model.neuronsPerLayer[OUTPUT_LAYER];

    // Change working directory so data can be referenced relative to parent data folder
    chdir("..");

    struct Data trainData;
    fill(&trainData, "data/xor/train.csv", NUMBER_OF_COLUMNS, TRAIN_ENTRIES_SIZE);

    struct Data testData;
    fill(&testData, "data/xor/test.csv", NUMBER_OF_COLUMNS, TEST_ENTRIES_SIZE);

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
    double* predictedOutputs[TEST_ENTRIES_SIZE];
    for (int predictedOutputIndex = 0; predictedOutputIndex < TEST_ENTRIES_SIZE; predictedOutputIndex++)
        predictedOutputs[predictedOutputIndex] = malloc(sizeof(double) * numberOfOutputs);

    double costs[TEST_ENTRIES_SIZE];

    test(&model, &testData, inputColumnIndices, outputColumnIndices, predictedOutputs, costs);

    for (int entryIndex = 0; entryIndex < TEST_ENTRIES_SIZE; entryIndex++) {
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
    exit(0);
}

