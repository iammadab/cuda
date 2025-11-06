#define LINALG_IMPLEMENTATION
#include "../../include/linalg.h"

#define CONV_IMPLEMENTATION
#include "../../include/conv.h"

int N = 5;
int FILTER_SIZE = 1;
int FILTER_WIDTH = 2 * FILTER_SIZE + 1;

int main() {
  // allocate memory for the input
  int size = N * N;
  float *A = rand_init(size);
  print_arr_2d(A, N, size);

  // allocate memory for the filter
  int size_f = FILTER_WIDTH * FILTER_WIDTH; 
  float *F = (float*) malloc(size_f * sizeof(float));

  // scale each element by a factor
  F[FILTER_SIZE * FILTER_WIDTH + FILTER_SIZE] = 2.0;
  print_arr_2d(F, FILTER_WIDTH, size_f);

  // allocate the output
  float *C = (float*) malloc(size * sizeof(float));
  conv_cpu(A, C, F, FILTER_SIZE, N);

  print_arr_2d(C, N, size);
}

