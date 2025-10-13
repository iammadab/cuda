#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>


int main () {
  int desired_channels = 3;
  int width, height, channels;

  unsigned char *img_data = stbi_load("images/sheeps.jpg", &width, &height, &channels, desired_channels);
  if (!img_data) {
    fprintf(stderr, "Failed to load image: %s", stbi_failure_reason());
    return 1;
  }

  size_t total_img_bytes = (size_t)width * height * channels;

  // write img to a file
  int stride = width * desired_channels; // tightly packed, no data alignment
  if (!stbi_write_png("output.png", width, height, desired_channels, img_data, width * desired_channels)) {
      fprintf(stderr, "Write failed!\n");
      stbi_image_free(img_data);
      return 1;
  }

  stbi_image_free(img_data);
  return 0;
}
