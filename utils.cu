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

// TODO: change this to a macro, so __FILE__ and __LINE__ are useful
void check_err(cudaError_t resp) {
  if (resp != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(resp), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
}
