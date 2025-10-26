#define UTILS_IMPLEMENTATION
#include "utils.h"

int M = 100;
int K = 100;
int N = 100;

// MATMUL KERNEL
// C = A x B
//
// Dimensions (row, col)
// A = (M, K)
// B = (K, N)
// C = (M, N)

__global__ void matmul_kernel_tiled(float *A, float *B, float *C, int M, int N, int k) {}

int main() {
  // allocate memory on the cpu and the gpu
  // do warmup
  // run kernel + timing
  // then do correctness check

  int size_a = M * K;
  int size_b = K * N;
  int size_c = M * N;

  // allocate memory on host
  float *A_h = rand_init(size_a);
  float *B_h = rand_init(size_b);
  float *C_h = (float *) malloc(size_c * sizeof(float));
  float *C_h_cpu_result = (float *) malloc(size_c * sizeof(float));

  // allocate memory on device
  float *A_d, *B_d, *C_d;
  check_err(cudaMalloc(&A_d, size_a * sizeof(float)));
  check_err(cudaMalloc(&B_d, size_b * sizeof(float)));
  check_err(cudaMalloc(&C_d, size_c * sizeof(float)));

  // compute expected answer on the cpu
  matmul_cpu(A_h, B_h, C_h_cpu_result, M, N, K);

  return 0;
}
