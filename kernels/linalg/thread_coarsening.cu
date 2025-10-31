#define UTILS_IMPLEMENTATION
#include "../../include/utils.h"

#define TILE_WIDTH 16
#define COARSE_FACTOR 4

int M = 1000;
int K = 1000;
int N = 1000;

// MATMUL KERNEL
// C = A x B
//
// Dimensions (row, col)
// A = (M, K)
// B = (K, N)
// C = (M, N)

__global__ void matmul_thread_coarsening(float *A, float *B, float *C, int M, int N, int K) {
  // deine shared variables to hold the tiles
  __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  // tiles already done = blockIdx * blockIdx * COARSE_FACTOR
  // we add threadIdx.x to get the column within the first tile
  int col = blockIdx.x * blockDim.x * COARSE_FACTOR + threadIdx.x;

  // init zero output
  float c_values[COARSE_FACTOR];
  for (int i = 0; i < COARSE_FACTOR; ++i) {
    c_values[i] = 0.f;
  }

  // phases
  for (int ph = 0; ph < ceil(K / (float) TILE_WIDTH); ++ph) {
    int ph_row = ph * TILE_WIDTH + threadIdx.y;
    int ph_col = ph * TILE_WIDTH + threadIdx.x;

    if (row < M && ph_col < K)
      Ads[threadIdx.y][threadIdx.x] = A[row * K + ph_col];
    else 
      Ads[threadIdx.y][threadIdx.x] = 0.0f;

    // sequentially load the relevant B's into shared memory
    for (int fac = 0; fac < COARSE_FACTOR; ++fac) {
      int fac_col = col + fac * TILE_WIDTH;

      if (ph_row < K && fac_col < N)
        Bds[threadIdx.y][threadIdx.x] = B[(ph * TILE_WIDTH + threadIdx.y) * N + fac_col];
      else
        Bds[threadIdx.y][threadIdx.x] = 0.0f;

      __syncthreads();

      for (int i = 0; i < TILE_WIDTH; i++) {
        c_values[fac] += Ads[threadIdx.y][i] * Bds[i][threadIdx.x];
      }
      __syncthreads();
    }
  }

  // write output
  for (int fac = 0; fac < COARSE_FACTOR; ++fac) {
    int fac_col = col + fac * TILE_WIDTH; 
    if (row < M && fac_col < N)
      C[row * N + fac_col] = c_values[fac];
  }

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

  int grid_x = ceil(N / (float) (TILE_WIDTH * COARSE_FACTOR));
  int grid_y = ceil(M / (float) TILE_WIDTH);

  dim3 block(TILE_WIDTH, TILE_WIDTH);
  dim3 grid(grid_x, grid_y);

  // comparison parameters
  const int WARMUP_COUNT = 3;
  const int REPEAT_COUNT = 10;

  cudaEvent_t start, stop;
  CHECK_ERR(cudaEventCreate(&start));
  CHECK_ERR(cudaEventCreate(&stop));

  // warmup run
  for (int i = 0; i < WARMUP_COUNT; ++i) {
    matmul_thread_coarsening<<<grid, block>>>(A_d, B_d, C_d, M, N, K);
  }
  CHECK_ERR(cudaDeviceSynchronize());

  // timed run
  CHECK_ERR(cudaEventRecord(start));
  for (int i = 0; i < REPEAT_COUNT; ++i) {
    matmul_thread_coarsening<<<grid, block>>>(A_d, B_d, C_d, M, N, K);
  }
  CHECK_ERR(cudaEventRecord(stop));
  CHECK_ERR(cudaEventSynchronize(stop));

  // copy result to host
  CHECK_ERR(cudaMemcpy(C_h, C_d, size_c * sizeof(float), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  compare_arr(C_h_cpu_result, C_h, size_c, EPSILON);

  float ms = 0;
  CHECK_ERR(cudaEventElapsedTime(&ms, start, stop));
  ms /= REPEAT_COUNT;

  printf("ok coarsed matmul: %fms\n", ms);

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
