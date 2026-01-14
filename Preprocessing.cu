#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA Kernel: HWC BGR/RGB (uint8) -> CHW RGB (float) + Normalization
__global__ void preprocess_kernel(const unsigned char *input, float *output,
                                  int width, int height, bool is_bgr) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int in_idx = (y * width + x) * 3;

    // CHW offsets
    int pixel_count = width * height;
    int r_offset = 0;
    int g_offset = pixel_count;
    int b_offset = 2 * pixel_count;

    // RGB order handling
    unsigned char v0 = input[in_idx];
    unsigned char v1 = input[in_idx + 1];
    unsigned char v2 = input[in_idx + 2];

    float r, g, b;
    if (is_bgr) {
      b = (static_cast<float>(v0) - 127.5f) / 127.5f;
      g = (static_cast<float>(v1) - 127.5f) / 127.5f;
      r = (static_cast<float>(v2) - 127.5f) / 127.5f;
    } else {
      r = (static_cast<float>(v0) - 127.5f) / 127.5f;
      g = (static_cast<float>(v1) - 127.5f) / 127.5f;
      b = (static_cast<float>(v2) - 127.5f) / 127.5f;
    }

    output[r_offset + y * width + x] = r;
    output[g_offset + y * width + x] = g;
    output[b_offset + y * width + x] = b;
  }
}

extern "C" void launch_preprocess(const unsigned char *d_input, float *d_output,
                                  int width, int height, bool is_bgr,
                                  cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  preprocess_kernel<<<grid, block, 0, stream>>>(d_input, d_output, width,
                                                height, is_bgr);
}
