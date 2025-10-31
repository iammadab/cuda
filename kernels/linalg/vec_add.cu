#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define UTILS_IMPLEMENTATION
#include "../../include/utils.h"

#define LINALG_IMPLEMENTATION
#include "../../include/linalg.h"

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
  CHECK_ERR(cudaMalloc(&A_d, size));
  CHECK_ERR(cudaMalloc(&B_d, size));
  CHECK_ERR(cudaMalloc(&C_d, size));
  
  // Copy A and B to device memory
  // (dest, source, size, direction)
  CHECK_ERR(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
  CHECK_ERR(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));

  // create cuda events for timing
  cudaEvent_t start, stop;
  CHECK_ERR(cudaEventCreate(&start));
  CHECK_ERR(cudaEventCreate(&stop));

  int grid_size = ceil(n/256.0);
  int block_size = 256;

  // warm up
  for (int i = 0; i < 10; i++) {
    vec_add_kernel<<<grid_size, block_size>>>(A_d, B_d, C_d, size);
  }
  cudaDeviceSynchronize();

  int runs = 100;
  float total_ms = 0;

  for (int i = 0; i < runs; i++) {
    // Record the start event
    CHECK_ERR(cudaEventRecord(start));
    
    // Part 2: call kernel - to launch a grid of threads
    // to perform the actual vector addition
    vec_add_kernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, size);

    // Record the stop event
    CHECK_ERR(cudaEventRecord(stop));
    CHECK_ERR(cudaEventSynchronize(stop));

    // calculate the elapsed time
    float ms = 0;
    CHECK_ERR(cudaEventElapsedTime(&ms, start, stop));
    total_ms += ms;
  }

  printf("Average kernel execution time: %f ms \n", total_ms / runs);

  // Part 3: Copy C from this device memory
  CHECK_ERR(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

  // Free device vectors
  CHECK_ERR(cudaFree(A_d));
  CHECK_ERR(cudaFree(B_d));
  CHECK_ERR(cudaFree(C_d));
}

int main() {
  // set the seed for the random number generator
  srand(time(NULL));

  // allocate memory for the vectors
  int N = 1000000;

  float *A_h = rand_init(N);
  float *B_h = rand_init(N);
  float *C_h = (float*) malloc(N * sizeof(float));

  vec_add(A_h, B_h, C_h, N);

  // free allocated memory
  free(A_h);
  free(B_h);
  free(C_h);
}
