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

// TODO: add comments explaining the indexing for future me

__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= M || col >= N) return;

  float sum = 0;
  for (int i = 0; i < K; ++i) {
    sum += A[row * K + i] * B[col + i * N]; 
  }

  C[row * N + col] = sum;
}

__global__ void matmul_kernel_b_transpose(float *A, float *B, float *C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= M || col >= N) return;

  float sum = 0;
  for (int i = 0; i < K; ++i) {
    sum += A[row * K + i] * B[col * K + i]; 
  }

  C[row * N + col] = sum;
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

int main() {
  int size_a = M * K;
  int size_b = K * N;
  int size_c = M * N;

  // allocate memory on host
  float *A_h = rand_init(size_a);
  float *B_h = rand_init(size_b);
  float *C_h = (float*) malloc(size_c * sizeof(float));
  float *C_h_cpu_result = (float*) malloc(size_c * sizeof(float));

  // compute the transpose of B
  float *B_h_transpose = transpose_arr(B_h, K, N);

  // allocate memory on device
  float *A_d, *B_d, *B_d_transpose, *C_d;
  CHECK_ERR(cudaMalloc(&A_d, size_a * sizeof(float)));
  CHECK_ERR(cudaMalloc(&B_d, size_b * sizeof(float)));
  CHECK_ERR(cudaMalloc(&B_d_transpose, size_b * sizeof(float)));
  CHECK_ERR(cudaMalloc(&C_d, size_c * sizeof(float)));

  // move the A and B matrices to device
  CHECK_ERR(cudaMemcpy(A_d, A_h, size_a * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_ERR(cudaMemcpy(B_d, B_h, size_b * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_ERR(cudaMemcpy(B_d_transpose, B_h_transpose, size_b * sizeof(float), cudaMemcpyHostToDevice));

  // compute result on CPU for comparison
  matmul_cpu(A_h, B_h, C_h_cpu_result, M, N, K);

  // kernel parameters
  dim3 block(16, 16);
  dim3 grid(ceil(N / 16.0), ceil(M / 16.0));

  // comparison parameter
  const int WARMUP_COUNT = 3;
  const int REPEAT_COUNT = 10;
  float eps = 1e-4f;

  cudaEvent_t start, stop;

  CHECK_ERR(cudaEventCreate(&start));
  CHECK_ERR(cudaEventCreate(&stop));


  // matmul kernel
  // warmup
  for (int i = 0; i < WARMUP_COUNT; ++i) {
    matmul_kernel<<<grid, block>>>(A_d, B_d, C_d, M, N, K);
  }
  CHECK_ERR(cudaDeviceSynchronize());


  // timed run
  CHECK_ERR(cudaEventRecord(start));
  for (int i = 0; i < REPEAT_COUNT; ++i) {
    matmul_kernel<<<grid, block>>>(A_d, B_d, C_d, M, N, K);
  }
  CHECK_ERR(cudaEventRecord(stop));
  CHECK_ERR(cudaEventSynchronize(stop));

  // copy result to host
  CHECK_ERR(cudaMemcpy(C_h, C_d, size_c * sizeof(float), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  for (int i = 0; i < size_c; i++) {
    if (fabsf(C_h_cpu_result[i] - C_h[i]) > eps) {
      fprintf(stderr, "result mismatch");
      return 1;
    }
  }

  float ms = 0;
  CHECK_ERR(cudaEventElapsedTime(&ms, start, stop));
  ms /= REPEAT_COUNT;

  printf("ok matmul kernel: %fms\n", ms);





  // matmul kernel with b transpose
  // warmup
  for (int i = 0; i < WARMUP_COUNT; ++i) {
    matmul_kernel_b_transpose<<<grid, block>>>(A_d, B_d_transpose, C_d, M, N, K);
  }
  CHECK_ERR(cudaDeviceSynchronize());

  // timed run
  CHECK_ERR(cudaEventRecord(start));
  for (int i = 0; i < REPEAT_COUNT; ++i) {
    matmul_kernel_b_transpose<<<grid, block>>>(A_d, B_d_transpose, C_d, M, N, K);
  }
  CHECK_ERR(cudaEventRecord(stop));
  CHECK_ERR(cudaEventSynchronize(stop));

  // copy result to host
  CHECK_ERR(cudaMemcpy(C_h, C_d, size_c * sizeof(float), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  for (int i = 0; i < size_c; i++) {
    if (fabsf(C_h_cpu_result[i] - C_h[i]) > eps) {
      fprintf(stderr, "result mismatch");
      return 1;
    }
  }
  
  CHECK_ERR(cudaEventElapsedTime(&ms, start, stop));
  ms /= REPEAT_COUNT;

  printf("ok matmul kernel with b transpose: %fms\n", ms);
  
  

  // free memory
  free(A_h);
  free(B_h);
  free(C_h);
  free(C_h_cpu_result);
  free(B_h_transpose);
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(B_d_transpose);
  cudaFree(C_d);


  return 0;
}
