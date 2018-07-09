#include <stdio.h>
#include <memory.h>
#include <stdlib.h>
#include "data.h"

void fill(struct Data* data, char fileName[]) {
    /* Two declarations because character comparison required another character, while string splitting
     * requires string */
    static const char SEPARATOR_CHAR  = ',';
    static const char SEPARATOR_STR[] = ",";

    FILE *file = fopen(fileName, "r");

    fseek(file, 0L, SEEK_END);
    size_t fileLength = (size_t) ftell(file);
    fseek(file, 0L, SEEK_SET);

    char fileContent[fileLength + 1];

    fread(fileContent, fileLength, 1, file);
    fclose(file);

    data->numberOfEntries = 0;

    /* Start off with one column, not zero, since one comma = two columns, n commas = n + 1 columns.
     * This does assume that there is atleast one column */
    data->numberOfColumns = 1;

    for (int characterIndex = 0; characterIndex < fileLength; characterIndex++) {
        char character = fileContent[characterIndex];

        // If there have been no detected entries, that means we are on the first line of the file
        bool onColumnLine = data->numberOfEntries == 0;

        // Each separator implies a column that exists past it
        if (onColumnLine && character == SEPARATOR_CHAR)
            data->numberOfColumns++;

        // Each newline implies an entry that exists past it
        if (character == '\n')
            data->numberOfEntries++;
    }

    data->columnNames = malloc(sizeof(char*) * data->numberOfColumns);

    // Used for strtok_r saves
    char* saveForColumn;
    char* saveForLine;

    char* columnLine = strtok_r(fileContent, "\n", &saveForLine);

    char* columnName = strtok_r(columnLine, SEPARATOR_STR, &saveForColumn);
    for (int columnIndex = 0; columnIndex < data->numberOfColumns; columnIndex++) {
        data->columnNames[columnIndex] = columnName;

        columnName = strtok_r(NULL, ",", &saveForColumn);
    }

    data->elements = malloc(sizeof(double*) * data->numberOfEntries);

    char* line = strtok_r(NULL, "\n", &saveForLine);
    for (int entryIndex = 0; entryIndex < data->numberOfEntries; entryIndex++) {
        data->elements[entryIndex] = malloc(sizeof(double) * data->numberOfColumns);

        char* saveForValue;

        char* elementValueStr = strtok_r(line, SEPARATOR_STR, &saveForValue);
        double elementValue;

        for (int columnIndex = 0; columnIndex < data->numberOfColumns; columnIndex++) {
            sscanf(elementValueStr, "%lf", &elementValue);

            data->elements[entryIndex][columnIndex] = elementValue;

            elementValueStr = strtok_r(NULL, ",", &saveForValue);
        }

        line = strtok_r(NULL, "\n", &saveForLine);
    }
}

void freeData(struct Data* data) {
    freeColumnNames(data);
    freeElements(data);
}

void freeColumnNames(struct Data* data) {
    free(data->columnNames);
}

void freeElements(struct Data* data) {
    for (int entryIndex = 0; entryIndex < data->numberOfEntries; entryIndex++) {
        double* entry = data->elements[entryIndex];

        free(entry);
    }

    free(data->elements);
}