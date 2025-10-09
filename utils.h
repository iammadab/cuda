#ifndef UTILS_H
#define UTILS_H
#include <stddef.h>

void print_arr(float* Arr, size_t N);
float* rand_init(size_t N);
void check_err(cudaError_t resp);

#endif
