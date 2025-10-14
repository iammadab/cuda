#define STB_IMAGE_IMPLEMENTATION
#include "./external/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./external/stb_image_write.h"

#define UTILS_IMPLEMENTATION
#include "utils.h"

#include <stdio.h>

// I am doing a grayscale blur
// hence the data I am getting is in grayscale or can be converted to grayscale
// then I need to average the pixels in the kernel
// same idea one thread per pixel output
// each thread just gets its surrounding kernel

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
