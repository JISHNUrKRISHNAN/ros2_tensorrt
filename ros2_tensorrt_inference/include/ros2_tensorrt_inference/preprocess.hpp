#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
namespace ros2_trt {
// POD passed by value to the CUDA kernel. RGB normalization order is always R,G,B.
struct Normalization { float scale; float mean[3]; float stddev[3]; };
void preprocess(const uint8_t* input, int width, int height, size_t step, bool bgr,
                float* output, int out_width, int out_height,
                Normalization normalization, cudaStream_t stream);
}
