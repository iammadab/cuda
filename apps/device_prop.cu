#include <stdio.h>

#define UTILS_IMPLEMENTATION
#include "utils.h"

int main() {
  int device_count;
  cudaDeviceProp device_prop;

  CHECK_ERR(cudaGetDeviceCount(&device_count));

  printf("Detected %d CUDA capable device(s)\n\n", device_count);

  for (int i = 0; i < device_count; ++i) {
    CHECK_ERR(cudaGetDeviceProperties(&device_prop, i));

    printf("Device: %s\n", device_prop.name);
    printf("Device Capability: %d.%d\n", device_prop.major, device_prop.minor);
    printf("Global Memory: %.2f GB\n", (float) device_prop.totalGlobalMem / (1024 * 1024 * 1024));
    printf("Constant Memory: %d bytes\n", device_prop.totalConstMem);
    printf("Max Shared Memory per Block: %.2f KB \n", (float) device_prop.sharedMemPerBlock / 1024);
    printf("SMs: %d\n", device_prop.multiProcessorCount);
    printf("Max Blocks per SM: %d\n", device_prop.maxBlocksPerMultiProcessor);
    printf("Max Thread per Block: %d\n", device_prop.maxThreadsPerBlock);
    printf("Max Block Dim: (%d, %d, %d)\n", device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1], device_prop.maxThreadsDim[2]);
    printf("Max Grid Dim: (%d, %d, %d)\n", device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2]);
    printf("L2 Cache Size: %.2f KB \n", (float) device_prop.l2CacheSize / (1024));
  }


  return 0;
}
