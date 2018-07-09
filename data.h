#ifndef NEURALNETWORK_DATA_H
#define NEURALNETWORK_DATA_H

#include <stdbool.h>

#define MAX_STRING_LENGTH 20

struct Data {
    char** columnNames;
    double** elements;

    int numberOfColumns;
    int numberOfEntries;
};

void fill(struct Data* data, char fileName[]);
void freeColumnNames(struct Data* data);
void freeElements(struct Data* data);

#endif
