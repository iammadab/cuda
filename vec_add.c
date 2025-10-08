#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "utils.h"

void vec_add_cpu(float *A_h, float *B_h, float *C_h, int n) {
  for (int i = 0; i < n; ++i)
    C_h[i] = A_h[i] + B_h[i];
}

void vec_add(float *A, float* B, float* C, int n) {
  int size = n * sizeof(float);
  float *d_A, *d_B, *d_C;

  // Part 1: Allocate device memory for A, B and C
  // Copy A and to device memory
  
  // Part 2: call kernel - to launch a grid of threads
  // to perform the actual vector addition

  // Part 3: Copy C from this device memory
  // Free device vectors
}

int main() {
  // set the seed for the random number generator
  srand(time(NULL));

  // allocate memory for the vectors
  int N = 10;

  float *A_h = rand_init(N);
  float *B_h = rand_init(N);
  float *C_h = (float*) malloc(N * sizeof(float));


  print_arr(A_h, N);
  print_arr(B_h, N);
  print_arr(C_h, N);

  vec_add_cpu(A_h, B_h, C_h, N);

  print_arr(A_h, N);
  print_arr(B_h, N);
  print_arr(C_h, N);
  
  // free allocated memory
  free(A_h);
  free(B_h);
  free(C_h);
}


