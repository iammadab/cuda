#define STB_IMAGE_IMPLEMENTATION
#include <stdio.h>
#include "stb_image.h"

int main () {
  int width, height, channels;

  unsigned char *img_data = stbi_load("images/sheeps.jpg", &width, &height, &channels, 3);
  if (!img_data) {
    fprintf(stderr, "Failed to load image: %s", stbi_failure_reason());
    return 1;
  }

  stbi_image_free(img_data);
  return 0;
}
