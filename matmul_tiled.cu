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

// TODO: plot improvement with different tile widths

#define TILE_WIDTH 16

// TODO: handle irregular sized tile_widths

__global__ void matmul_kernel_tiled(float *A, float *B, float *C, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

  float sum = 0;

  for (int phase = 0; phase < K / TILE_WIDTH; ++phase) {
    Ads[row][col] = A[row * K + (phase * TILE_WIDTH + threadIdx.x)];
    Bds[row][col] = B[(phase * TILE_WIDTH + threadIdx.y) * K + col];
    __syncthreads();

    for (int i = 0; i < TILE_WIDTH; ++i) {
      sum += Ads[row][i] * Bds[i][col];
    }
    __syncthreads();
  }

  C[row * K + col] = sum;
}

int main() {
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

  // copy data to host
  check_err(cudaMemcpy(A_d, A_h, size_a * sizeof(float), cudaMemcpyDeviceToHost));
  check_err(cudaMemcpy(B_d, B_h, size_b * sizeof(float), cudaMemcpyDeviceToHost));

  // compute expected answer on the cpu
  matmul_cpu(A_h, B_h, C_h_cpu_result, M, N, K);

  // kernel parameters
  dim3 block(TILE_WIDTH, TILE_WIDTH);
  dim3 grid(ceil(N / (float) TILE_WIDTH), ceil(M / (float) TILE_WIDTH));

  // TODO: use the tile_width as the block size
  // TODO: compute the GRID size accordingly

  return 0;
}
