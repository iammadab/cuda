#define STB_IMAGE_IMPLEMENTATION
#include "./external/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./external/stb_image_write.h"

#define UTILS_IMPLEMENTATION
#include "utils.h"

#include <stdio.h>

__global__ void grayscale_blur(unsigned char *in, unsigned char out*, size_t width, size_t height) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < height && col < width) {
    int pixel_sum = 0;
    int pixel_count = 0;

    // sum all pixel values within tile whose 
    // center is defined by the current pixel
    for (int i = -1; i < 2; ++i) {
      for (int j = -1; j < 2; ++j) {
        int new_row = row + i;
        int new_col = col + j;

        // only contributes to the sum if within pixel box
        if (new_row >= 0 && new_row < height && new_col >= 0 && new_col < width) {
          pixel_sum += in[new_row * width + new_col];
          pixel_count += 1;
        }
      }
    }

    out[row * width + col] = (unsigned char) ((float) pixel_sum / pixel_count);
  } 
}

int main () {
  int desired_channels = 1; // grayscale
  int width, height, channels;

  const char *input_file = "images/sheeps.jpg";
  const char *output_file = "images/sheeps_grayscale_blur.png";

  // load img
  unsigned char *img_h = stbi_load(input_file, &width, &height, &channels, desired_channels);
  if (!img_h) {
    fprintf(stderr, "Failed to load image: %s", stbi_failure_reason());
    return 1;
  }

  size_t pixels = width * height;

  // allocate GPU memory for input image and grayscale_blur output
  unsigned char *img_d, *blur_d;
  check_err(cudaMalloc(&img_d, pixels * sizeof(unsigned char)));
  check_err(cudaMalloc(&blur_d, pixels * sizeof(unsigned char)));

  // copy image data to the GPU
  check_err(cudaMemcpy(img_d, img_h, pixels, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = ceil(pixels / 256.0);

  // TODO: launch kernel here

  // copy grayscale data from the gpu
  unsigned char *blur_h = (unsigned char *) malloc(pixels);
  check_err(cudaMemcpy(blur_h, blur_d, pixels, cudaMemcpyDeviceToHost));

  // write grayscale to file
  if (!stbi_write_png(output_file, width, height, 1, blur_h, width)) {
      fprintf(stderr, "Write failed!\n");
      stbi_image_free(img_h);
      return 1;
  }

  // free memory
  stbi_image_free(img_h);
  cudaFree(img_d);
  cudaFree(blur_d);
  free(blur_h);

  return 0;
}
