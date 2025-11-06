# define LINALG_IMPLEMENTATION
# include "../../include/linalg.h"

# define N 10
# define FILTER_SIZE 3

int main() {
  // allocate memory for the input
  int size = N * N;
  int size_bytes = size * sizeof(float);
  float *A = (float*) malloc(size_bytes);
  for (int i = 0; i < size; ++i) {
    A[i] = 1.0f;
  } 
  print_arr_2d(A, 10, size);

  // // allocate memory for the filter
  // int size_f = FILTER_SIZE * FILTER_SIZE; 
  // int size_f_bytes = size_f * sizeof(float);
  // float *F = (float*) malloc(size_f_bytes);
  // for (int i = 0; i < size_f; ++i) {
  //   F[i] = 1.0f;
  // }
  // print_arr(F, size_f);
  //
  // // allocate the output
  // float *C = (float *) malloc(size_bytes);
  //
  // // perform convolution
  // for (int row = 0; row < N; ++row) {
  //   for (int col = 0; col < N; ++col) {
  //     // the above loop will iterate over all the threads in the kernel
  //     // we need each of them to compute their convolution
  //     // we can use the blur size
  //     // we can start at -FILTER_SIZE to FILTER_SIZE
  //
  //     float sum = 0.0;
  //     // TODO make this cleaner
  //     for (int frow = -FILTER_SIZE; frow < FILTER_SIZE + 1; ++frow) {
  //       for (int fcol = -FILTER_SIZE; fcol < FILTER_SIZE + 1; ++fcol) {
  //         int nrow = row + frow; 
  //         int ncol = col + fcol; 
  //         sum += A[nrow * N + ncol] * F[(frow + FILTER_SIZE) * FILTER_SIZE + (fcol + FILTER_SIZE)];
  //       }
  //     }
  //     C[row * N + col] = sum;
  //   }
  // }
  //
  // print_arr(C, size);
}
