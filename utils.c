#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

void print_arr(float *Arr, size_t N) {
  for (size_t i = 0; i < N; ++i) 
    printf("%f\n", Arr[i]);
  printf("\n");
}

float* rand_init(size_t N) {
  float *Arr = (float*) malloc(N * sizeof(float));
  for (size_t i = 0; i < N; ++i)
    Arr[i] = (float) rand() / RAND_MAX;
  return Arr;
}
