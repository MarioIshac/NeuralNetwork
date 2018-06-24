cmake_minimum_required(VERSION 3.10)
project(NeuralNetwork C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "-pedantic -Wall -Wshadow -Wpointer-arith -Wcast-qual")

add_executable(NeuralNetwork main.c model.c model.h functions.c functions.h data.c data.h)

target_link_libraries(NeuralNetwork m)