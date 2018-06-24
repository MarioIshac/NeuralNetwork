#include <stdio.h>
#include <memory.h>
#include <stdlib.h>
#include "data.h"

void fill(struct Data* data, char fileName[], int numberOfColumns, int numberOfEntries) {
    data->numberOfColumns = numberOfColumns;
    data->numberOfEntries = numberOfEntries;

    FILE *file = fopen(fileName, "r");

    fseek(file, 0L, SEEK_END);
    long fileLength = ftell(file);
    fseek(file, 0L, SEEK_SET);

    char fileContent[fileLength + 1];

    fread(fileContent, fileLength, 1, file);
    fclose(file);

    data->columnNames = malloc(sizeof(char*) * numberOfColumns);

    char* saveForColumn;
    char* saveForLine;

    char* columnLine = strtok_r(fileContent, "\n", &saveForLine);

    char* columnName = strtok_r(columnLine, ",", &saveForColumn);
    for (int columnIndex = 0; columnIndex < numberOfColumns; columnIndex++) {
        data->columnNames[columnIndex] = columnName;

        columnName = strtok_r(NULL, ",", &saveForColumn);
    }

    data->elements = malloc(sizeof(double*) * numberOfEntries);

    char* line = strtok_r(NULL, "\n", &saveForLine);
    for (int entryIndex = 0; entryIndex < numberOfEntries; entryIndex++) {
        data->elements[entryIndex] = malloc(sizeof(double) * numberOfColumns);

        char* saveForValue;

        char* elementValueStr = strtok_r(line, ",", &saveForValue);
        double elementValue;

        for (int columnIndex = 0; columnIndex < numberOfColumns; columnIndex++) {
            sscanf(elementValueStr, "%lf", &elementValue);

            data->elements[entryIndex][columnIndex] = elementValue;

            elementValueStr = strtok_r(NULL, ",", &saveForValue);
            printf("Element[%i][%i] = %lf\n", entryIndex, columnIndex, data->elements[entryIndex][columnIndex]);
        }

        line = strtok_r(NULL, "\n", &saveForLine);
    }
}