#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "utils.h"

// void vecAdd(float *A_h, float *B_h, float *C_h, int n) {
//   for (int i = 0; i < n; ++i) {
//     C_h[i] = A_h[i] + B_h[i];
//   }
// }

int main() {
  // set the seed for the random number generator
  srand(time(NULL));

  // allocate memory for the vectors
  int N = 10;

  float *A_h = rand_init(N);
  float *B_h = rand_init(N);
  float *C_h = (float*) malloc(N * sizeof(float));

  print_arr(A_h, N);
  print_arr(B_h, N);
  print_arr(C_h, N);
  
  // free allocated memory
  free(A_h);
  free(B_h);
  free(C_h);
}


