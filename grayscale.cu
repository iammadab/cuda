#define STB_IMAGE_IMPLEMENTATION
#include "./external/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./external/stb_image_write.h"

#define UTILS_IMPLEMENTATION
#include "utils.h"

#include <stdio.h>

__global__ void rgb_to_grayscale_kernel(unsigned char *in, unsigned char *out, size_t pixels) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (id < pixels) {
    size_t offset = id * 3;

    // grayscale computation
    // g = 0.299 * r + 0.587 * g + 0.114 * b;
    out[id] = 0.299f * in[offset] + 0.587f * in[offset + 1] + 0.114f * in[offset + 2];
  }
}

int main (int argc, char **argv) {
  int desired_channels = 3; // rgb
  int width, height, channels;

  if (argc < 3) {
    fprintf(stderr, "please pass the input and output file\n");
    return 1;
  }

  const char *input_file = argv[1];
  const char *output_file = argv[2];

  if (!input_file || !output_file) {
    fprintf(stderr, "please pass the input and output file\n");
    return 1;
  }

  // load img
  unsigned char *img_data = stbi_load(input_file, &width, &height, &channels, desired_channels);
  if (!img_data) {
    fprintf(stderr, "Failed to load image: %s", stbi_failure_reason());
    return 1;
  }

  size_t pixels = width * height;
  size_t total_img_bytes = pixels * desired_channels;

  // allocate GPU memory for input image and grayscale output
  unsigned char *img_d, *grayscale_d;
  CHECK_ERR(cudaMalloc(&img_d, total_img_bytes * sizeof(unsigned char)));
  CHECK_ERR(cudaMalloc(&grayscale_d, pixels * sizeof(unsigned char)));

  // copy image data to the GPU
  CHECK_ERR(cudaMemcpy(img_d, img_data, total_img_bytes, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = ceil(pixels / 256.0);
  rgb_to_grayscale_kernel<<<grid, block>>>(img_d, grayscale_d, pixels);

  // copy grayscale data from the gpu
  unsigned char *grayscale_h = (unsigned char *) malloc(pixels);
  CHECK_ERR(cudaMemcpy(grayscale_h, grayscale_d, pixels, cudaMemcpyDeviceToHost));

  // write grayscale to file
  if (!stbi_write_png(output_file, width, height, 1, grayscale_h, width)) {
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
