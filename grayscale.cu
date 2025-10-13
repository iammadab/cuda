#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include "utils.h"

// TODO: check_err

__global__ void rgb_to_grayscale_kernel(unsigned char *in, unsigned char *out, size_t pixels) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (id < pixels) {
    size_t offset = id * 3;

    // grayscale computation
    // g = 0.299 * r + 0.587 * g + 0.114 * b;
    out[id] = 0.299f * in[offset] + 0.587f * in[offset + 1] + 0.114f * in[offset + 2];
  }
}

int main () {
  int desired_channels = 3; // rgb
  int width, height, channels;

  // load img
  unsigned char *img_data = stbi_load("images/sheeps.jpg", &width, &height, &channels, desired_channels);
  if (!img_data) {
    fprintf(stderr, "Failed to load image: %s", stbi_failure_reason());
    return 1;
  }

  size_t pixels = width * height;
  size_t total_img_bytes = pixels * desired_channels;

  // allocate GPU memory for input image and grayscale output
  unsigned char *img_d, *grayscale_d;
  check_err(cudaMalloc(&img_d, total_img_bytes * sizeof(unsigned char)));
  check_err(cudaMalloc(&grayscale_d, pixels * sizeof(unsigned char)));

  // copy image data to the GPU
  check_err(cudaMemcpy(img_d, img_data, total_img_bytes, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = ceil(pixels / 256.0);
  rgb_to_grayscale_kernel<<<grid, block>>>(img_d, grayscale_d, pixels);

  // copy grayscale data from the gpu
  unsigned char *grayscale_h = (unsigned char *) malloc(pixels);
  check_err(cudaMemcpy(grayscale_h, grayscale_d, pixels, cudaMemcpyDeviceToHost));

  // write grayscale to file
  int stride = width * desired_channels; // tightly packed, no data alignment
  if (!stbi_write_png("images/sheeps_grayscale.png", width, height, 1, grayscale_h, width)) {
      fprintf(stderr, "Write failed!\n");
      stbi_image_free(img_data);
      return 1;
  }

  // free memory
  stbi_image_free(img_data);
  cudaFree(img_d);
  cudaFree(grayscale_d);
  free(grayscale_h);

  return 0;
}
