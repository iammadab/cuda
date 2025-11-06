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

  // perform convolution
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      float sum = 0.0;

      // TODO make this cleaner
      for (int frow = 0; frow < FILTER_WIDTH; ++frow) {
        for (int fcol = 0; fcol < FILTER_WIDTH; ++fcol) {
          int nrow = row + frow - 1; 
          int ncol = col + fcol - 1; 
          sum += A[nrow * N + ncol] * F[frow * FILTER_WIDTH + fcol]; 
        }
      }
      C[row * N + col] = sum;
    }
  }

  print_arr_2d(C, N, size);
}
