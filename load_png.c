#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <stdio.h>

int main() {
  int width, height, channels;

  // Load the PNG into memory
  unsigned char *data = stbi_load("input.png", &width, &height, &channels, 3);
  if (!data) {
    fprintf(stderr, "Failed to load image: %s\n", stbi_failure_reason());
    return 1;
  }

  printf("Image loaded successfully!\n");
  printf("Width: %d, Height: %d, Channels requested: 3, Original channels: %d\n", width, height, channels);

  // print first 5 pixels
  for (int i = 0; i < 5; ++i) {
    int idx = i * 3;
    printf("Pixel %d: R = %d, G = %d, B = %d\n", i, data[idx], data[idx + 1], data[idx + 2]);
  }

  // free
  stbi_image_free(data);
  return 0;
}

