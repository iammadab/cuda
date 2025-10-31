// Declarations
#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>

void print_arr(float* Arr, size_t N);
float* rand_init(size_t N);
void matmul_cpu(float *A, float *B, float *C, int M, int N, int K);
float* transpose_arr(float *arr, int M, int N);

#endif


// Implementations
#ifdef UTILS_IMPLEMENTATION
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

#define CHECK_ERR(resp) do { \
    cudaError_t _err = (resp); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "%s in %s at line %d\n", \
                cudaGetErrorString(_err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

void matmul_cpu(float *A, float *B, float *C, int M, int N, int K) {
  for (int r = 0; r < M; ++r) {
    for (int c = 0; c < N; ++c) {
      float sum = 0;
      for (int i = 0; i < K; ++i) {
        sum += A[r * K + i] * B[c + i * N];
      }
      C[r * N + c] = sum;
    }
  }
}

// given some matrix A of dim (M, N)
/// returns a pointer to a new matrix (N, M)
/// with the data transposed
float* transpose_arr(float *arr, int M, int N) {
  float *result = (float*) malloc(M * N * sizeof(float));
  for (int r = 0; r < M; ++r) {
    for (int c = 0; c < N; ++c) {
      // (r, c) -> (c, r)
      result[c * M + r] = arr[r * N + c];
    }
  }
  return result;
}

#endif
