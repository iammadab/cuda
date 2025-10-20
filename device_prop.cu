#include <stdio.h>

#define UTILS_IMPLEMENTATION
#include "utils.h"

int main() {
  int device_count;

  check_err(cudaGetDeviceCount(&device_count));

  printf("Detected %d CUDA capable device(s)", device_count);

  return 0;
}
