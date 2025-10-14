#define STB_IMAGE_IMPLEMENTATION
#include "./external/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./external/stb_image_write.h"

#define UTILS_IMPLEMENTATION
#include "utils.h"

#include <stdio.h>

__global__ void grayscale_blur(unsigned char *in, unsigned char *out, int width, int height) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

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

int main (int argc, char **argv) {
  int desired_channels = 1; // grayscale
  int width, height, channels;

  const char *input_file = argv[1];
  const char *output_file = argv[2];

  if (!input_file || !output_file) {
    fprintf(stderr, "please pass the input and output file\n");
    return 1;
  }

  // load img
  unsigned char *img_h = stbi_load(input_file, &width, &height, &channels, desired_channels);
  if (!img_h) {
    fprintf(stderr, "Failed to load image: %s", stbi_failure_reason());
    return 1;
  }

  int pixels = width * height;

  // allocate GPU memory for input image and grayscale_blur output
  unsigned char *img_d, *blur_d;
  check_err(cudaMalloc(&img_d, pixels * sizeof(unsigned char)));
  check_err(cudaMalloc(&blur_d, pixels * sizeof(unsigned char)));

  // copy image data to the GPU
  check_err(cudaMemcpy(img_d, img_h, pixels, cudaMemcpyHostToDevice));

  dim3 block(16, 16, 1);
  dim3 grid(ceil(width / 16.0), ceil(height / 16.0));

  grayscale_blur<<<grid, block>>>(img_d, blur_d, width, height);

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
