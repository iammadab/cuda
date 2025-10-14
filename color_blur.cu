#define STB_IMAGE_IMPLEMENTATION
#include "./external/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./external/stb_image_write.h"

#define UTILS_IMPLEMENTATION
#include "utils.h"

#include <stdio.h>

#ifndef BLUR_SIZE
#define BLUR_SIZE 5
#endif

__global__ void color_blur(unsigned char *in, unsigned char *out, int width, int height, int channels) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= height || col >= width) return;

  int r_sum = 0;
  int g_sum = 0;
  int b_sum = 0;
  int pixel_count = 0;


  for (int i = -BLUR_SIZE; i < BLUR_SIZE + 1; ++i) {
    for (int j = -BLUR_SIZE; j < BLUR_SIZE + 1; ++j) {
      int new_row = row + i;
      int new_col = col + j;

      if (new_row >= 0 && new_row < height && new_col >= 0 && new_col < width) {
        int r_id = (new_row * width + new_col) * channels;
        r_sum += in[r_id];
        g_sum += in[r_id + 1];
        b_sum += in[r_id + 2];
        pixel_count += 1;
      }
    }
  }

  int r_id = (row * width + col) * channels;
  out[r_id] = (unsigned char)((float) r_sum / pixel_count);
  out[r_id + 1] = (unsigned char)((float) g_sum / pixel_count);
  out[r_id + 2] = (unsigned char)((float) b_sum / pixel_count);
}

int main(int argc, char **argv) {
  if (argc != 3) {
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
  int channels = 3;
  int width, height;

  unsigned char *img_h = stbi_load(input_file, &width, &height, NULL, channels);
  if (!img_h) {
    fprintf(stderr, "Failed to load image: %s", stbi_failure_reason());
    return 1;
  }
  
  int pixels = width * height;
  int total_bytes = pixels * channels;

  // allocate GPU memory
  unsigned char *img_d, *blur_d;
  check_err(cudaMalloc(&img_d, total_bytes));
  check_err(cudaMalloc(&blur_d, total_bytes));

  // copy image data to the gpu
  check_err(cudaMemcpy(img_d, img_h, total_bytes, cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid(ceil(width / 16.0), ceil(height / 16.0));

  color_blur<<<grid, block>>>(img_d, blur_d, width, height, channels);

  // copy blur data from the gpu
  unsigned char *blur_h = (unsigned char *) malloc(total_bytes);
  check_err(cudaMemcpy(blur_h, blur_d, total_bytes, cudaMemcpyDeviceToHost));

  // write 
  if (!stbi_write_png(output_file, width, height, channels, blur_h, width * channels)) {
    fprintf(stderr, "Write failed\n");
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
