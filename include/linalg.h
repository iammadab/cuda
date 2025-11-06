// Declaration 
#ifndef MATMUL_H
#define MATMUL_H

#define EPSILON 1e-4f

#include <stddef.h>

void print_arr(float* Arr, size_t N);
void print_arr_2d(float *Arr, size_t width, size_t N);
float* rand_init(size_t N);
void matmul_cpu(float *A, float *B, float *C, int M, int N, int K);
float* transpose_arr(float *arr, int M, int N);
void compare_arr(float *A, float *B, int size, float eps);

#endif



// Implementations
#ifdef LINALG_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>

void print_arr(float *Arr, size_t N) {
  for (size_t i = 0; i < N; ++i) 
    printf("%f\n", Arr[i]);
  printf("\n");
}

void print_arr_2d(float *Arr, size_t width, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    if (i != 0 && i % width == 0) {
      printf("\n");
    }
    printf("%.3f\t", Arr[i]);
  }
  printf("\n");
}

float* rand_init(size_t N) {
  float *Arr = (float*) malloc(N * sizeof(float));
  for (size_t i = 0; i < N; ++i)
    Arr[i] = (float) rand() / RAND_MAX;
  return Arr;
}

void matmul_cpu(float *A, float *B, float *C, int M, int N, int K) {
  #ifndef NOCPU
    for (int r = 0; r < M; ++r) {
      for (int c = 0; c < N; ++c) {
        float sum = 0;
        for (int i = 0; i < K; ++i) {
          sum += A[r * K + i] * B[c + i * N];
        }
        C[r * N + c] = sum;
      }
    }
  #else 
    printf("warning: not computing CPU matmul\n");
  #endif
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

void compare_arr(float *A, float *B, int size, float eps) {
  #ifndef NOCPU
    for (int i = 0; i < size; ++i) {
      if (fabsf(A[i] - B[i]) > eps) {
        fprintf(stderr, "result mismatch\n");
        exit(1);
      }
    }
  #else
    printf("warning: not comparing cpu output with kernel output\n");
  #endif
}

#endif
