# define LINALG_IMPLEMENTATION
# include "../../include/linalg.h"

int N = 5;
int FILTER_SIZE = 1;
int FILTER_WIDTH = 2 * FILTER_SIZE + 1;

int main() {
  // allocate memory for the input
  int size = N * N;
  int size_bytes = size * sizeof(float);
  float *A = (float*) malloc(size_bytes);
  for (int i = 0; i < size; ++i) {
    A[i] = 1.0f;
  } 
  print_arr_2d(A, N, size);

  // allocate memory for the filter
  int size_f = FILTER_WIDTH * FILTER_WIDTH; 
  int size_f_bytes = size_f * sizeof(float);
  float *F = (float*) malloc(size_f_bytes);
  // scale each element by a factor
  F[FILTER_SIZE * FILTER_WIDTH + FILTER_SIZE] = 2.0;
  print_arr_2d(F, FILTER_WIDTH, size_f);

  // allocate the output
  float *C = (float*) malloc(size_bytes);
  // conv_cpu(A, C, F, FILTER_SIZE, N);

  // perform convolution (iterate over each cell)
  // for (int row = 0; row < N; ++row) {
  //   for (int col = 0; col < N; ++col) {
  //     float sum = 0.0;
  //
  //     for (int frow = 0; frow < FILTER_WIDTH; ++frow) {
  //       for (int fcol = 0; fcol < FILTER_WIDTH; ++fcol) {
  //         int nrow = row + frow - FILTER_SIZE; 
  //         int ncol = col + fcol - FILTER_SIZE; 
  //         sum += A[nrow * N + ncol] * F[frow * FILTER_WIDTH + fcol]; 
  //       }
  //     }
  //     C[row * N + col] = sum;
  //   }
  // }

  print_arr_2d(C, N, size);
}

// TODO: add documentation
void conv_cpu(float *N, float *P, float *F, int FILTER_SIZE, int WIDTH) {
  int FILTER_WIDTH = 2 * FILTER_SIZE + 1;
  for (int row = 0; row < WIDTH; ++row) {
    for (int col = 0; col < WIDTH; ++col) {
      float sum = 0.0;
      for (int f_row = 0; f_row < FILTER_WIDTH; ++f_row) {
        for (int f_col = 0; f_col < FILTER_WIDTH; ++f_col) {
          int n_row = row + f_row - FILTER_SIZE;
          int n_col = col + f_col - FILTER_SIZE;
          sum += N[n_row * WIDTH + n_col] * F[f_row * FILTER_WIDTH + f_col];
        }
      }
      P[row * N + col] = sum;
    }
  }
}
