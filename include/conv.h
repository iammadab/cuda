#ifndef CONV_H
#define CONV_H

void conv_cpu(float *input, float *output, float *kernel, int KERNEL_RADIUS, int WIDTH);

#endif // !CONV_H

       
#ifdef CONV_IMPLEMENTATION

// Applies a convolution kernel to every cell in the input
// to produce each cell of the output
void conv_cpu(float *input, float *output, float *kernel, int KERNEL_RADIUS, int WIDTH) {
  int KERNEL_WIDTH = 2 * KERNEL_RADIUS + 1;
  for (int row = 0; row < WIDTH; ++row) {
    for (int col = 0; col < WIDTH; ++col) {
      float sum = 0.0;
      for (int f_row = 0; f_row < KERNEL_WIDTH; ++f_row) {
        for (int f_col = 0; f_col < KERNEL_WIDTH; ++f_col) {
          int n_row = row + f_row - KERNEL_RADIUS;
          int n_col = col + f_col - KERNEL_RADIUS;
          sum += input[n_row * WIDTH + n_col] * kernel[f_row * KERNEL_WIDTH + f_col];
        }
      }
      output[row * WIDTH + col] = sum;
    }
  }
}

#endif // !CONV_IMPLEMENTATION
