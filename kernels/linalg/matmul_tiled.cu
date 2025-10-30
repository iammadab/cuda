#define UTILS_IMPLEMENTATION
#include "utils.h"

int M = 320;
int K = 320;
int N = 320;

// MATMUL KERNEL
// C = A x B
//
// Dimensions (row, col)
// A = (M, K)
// B = (K, N)
// C = (M, N)

// TODO: plot improvement with different tile widths

#define TILE_WIDTH 16

__global__ void matmul_kernel_tiled(float *A, float *B, float *C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

  float sum = 0;

  for (int phase = 0; phase < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++phase) {
    int ph_row = phase * TILE_WIDTH + threadIdx.y;
    int ph_col = phase * TILE_WIDTH + threadIdx.x;

    if (row < M && ph_col < K)
      Ads[threadIdx.y][threadIdx.x] = A[row * K + ph_col];
    else
      Ads[threadIdx.y][threadIdx.x] = 0.0f;

    if (ph_row < K && col < N)
      Bds[threadIdx.y][threadIdx.x] = B[ph_row * N + col];
    else
      Bds[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();
  
    for (int i = 0; i < TILE_WIDTH; ++i) {
      sum += Ads[threadIdx.y][i] * Bds[i][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < M && col < N)
    C[row * N + col] = sum;
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
  CHECK_ERR(cudaMalloc(&A_d, size_a * sizeof(float)));
  CHECK_ERR(cudaMalloc(&B_d, size_b * sizeof(float)));
  CHECK_ERR(cudaMalloc(&C_d, size_c * sizeof(float)));


  // copy data to host
  CHECK_ERR(cudaMemcpy(A_d, A_h, size_a * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_ERR(cudaMemcpy(B_d, B_h, size_b * sizeof(float), cudaMemcpyHostToDevice));

  // compute expected answer on the cpu
  matmul_cpu(A_h, B_h, C_h_cpu_result, M, N, K);

  // kernel parameters
  dim3 block(TILE_WIDTH, TILE_WIDTH);
  dim3 grid(ceil(N / (float) TILE_WIDTH), ceil(M / (float) TILE_WIDTH));

  // comparison parameters
  const int WARMUP_COUNT = 3;
  const int REPEAT_COUNT = 10;
  float eps = 1e-4f;

  cudaEvent_t start, stop;
  CHECK_ERR(cudaEventCreate(&start));
  CHECK_ERR(cudaEventCreate(&stop));

  // tiled matmul kernel
  for (int i = 0; i < WARMUP_COUNT; ++i) {
    matmul_kernel_tiled<<<grid, block>>>(A_d, B_d, C_d, M, N, K);
  }
  CHECK_ERR(cudaDeviceSynchronize());

  // timed run
  CHECK_ERR(cudaEventRecord(start));
  for (int i = 0; i < REPEAT_COUNT; ++i) {
    matmul_kernel_tiled<<<grid, block>>>(A_d, B_d, C_d, M, N, K);
  }
  CHECK_ERR(cudaEventRecord(stop));
  CHECK_ERR(cudaEventSynchronize(stop));

  // copy result to host
  CHECK_ERR(cudaMemcpy(C_h, C_d, size_c * sizeof(float), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  for (int i = 0; i < size_c; ++i) {
    if (fabsf(C_h_cpu_result[i] - C_h[i]) > eps) {
      fprintf(stderr, "result mismatch");
      return 1;
    }
  }

  float ms = 0;
  CHECK_ERR(cudaEventElapsedTime(&ms, start, stop));
  ms /= REPEAT_COUNT;

  printf("ok tiled matmul: %fms\n", ms);

  // free memory
  free(A_h);
  free(B_h);
  free(C_h);
  free(C_h_cpu_result);
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  return 0;
}
