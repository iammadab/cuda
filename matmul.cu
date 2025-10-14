#define UTILS_IMPLEMENTATION
#include "utils.h"

#ifndef M
#define M 10
#define K 10
#define N 10
#endif

// MATMUL KERNEL
// C = A x B
//
// Dimensions (row, col)
// A = (M, K)
// B = (K, N)
// C = (M, N)

int main() {
  int size_a = M * N;
  int size_b = K * N;
  int size_c = M * N;

  // allocate memory on host
  float *A_h = rand_init(size_a);
  float *B_h = rand_init(size_b);
  float *C_h = malloc(size_c * sizeof(float));

  // allocate memory on device
  float *A_d, *B_d, *C_d;
  check_err(cudaMalloc(&A_d, size_a * sizeof(float)));
  check_err(cudaMalloc(&B_d, size_b * sizeof(float)));
  check_err(cudaMalloc(&C_d, size_c * sizeof(float)));

  // move the A and B matrice to device
  check_err(cudaMemcpy(A_d, A_h, size_a * sizeof(float), cudaMemcpyHostToDevice));
  check_err(cudaMemcpy(B_d, B_h, size_b * sizeof(float), cudaMemcpyHostToDevice));

  // launch the kernel
  dim3 block(16, 16);
  dim3 grid(ceil(N / 16.0), ceil(M / 16.0));
  matmul_kernel<<<grid, block>>>(A_d, B_d, C_d, M, N);

  // TODO: implement CPU matmul for correctness testing

  // copy result to host
  check_err(cudaMemcpy(C_h, C_d, size_c * sizeof(float), cudaMemcpyDeviceToHost));

  // TODO: compare against the CPUs matmul computation

  return 0;
}
