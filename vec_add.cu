#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "utils.h"

void vec_add_cpu(float *A_h, float *B_h, float *C_h, int n) {
  for (int i = 0; i < n; ++i)
    C_h[i] = A_h[i] + B_h[i];
}

__global__
void vec_add_kernel(float *A, float *B, float *C, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

void vec_add(float *A_h, float* B_h, float* C_h, int n) {
  int size = n * sizeof(float);
  float *A_d, *B_d, *C_d;

  // Part 1: Allocate device memory for A, B and C
  check_err(cudaMalloc(&A_d, size));
  check_err(cudaMalloc(&B_d, size));
  check_err(cudaMalloc(&C_d, size));
  
  // Copy A and B to device memory
  // (dest, source, size, direction)
  check_err(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
  check_err(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));
  
  // Part 2: call kernel - to launch a grid of threads
  // to perform the actual vector addition
  vec_add_kernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, size);

  // Part 3: Copy C from this device memory
  check_err(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

  // Free device vectors
  check_err(cudaFree(A_d));
  check_err(cudaFree(B_d));
  check_err(cudaFree(C_d));
}

int main() {
  // set the seed for the random number generator
  srand(time(NULL));

  // allocate memory for the vectors
  int N = 10;

  float *A_h = rand_init(N);
  float *B_h = rand_init(N);
  float *C_h = (float*) malloc(N * sizeof(float));

  vec_add(A_h, B_h, C_h, N);

  print_arr(A_h, N);
  print_arr(B_h, N);
  print_arr(C_h, N);

  // free allocated memory
  free(A_h);
  free(B_h);
  free(C_h);
}
