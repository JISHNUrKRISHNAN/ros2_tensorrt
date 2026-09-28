#include "ros2_tensorrt_inference/preprocess.hpp"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdint>
namespace {
struct DeviceMemory {
  void* pointer=nullptr;
  ~DeviceMemory(){if(pointer) cudaFree(pointer);}
};
}
TEST(CudaPreprocess, PaddedBgrBilinearAndNormalization) {
  int count=0;
  if(cudaGetDeviceCount(&count)!=cudaSuccess || count==0) GTEST_SKIP()<<"No CUDA GPU available";
  ASSERT_EQ(cudaSetDevice(0),cudaSuccess);
  // Two pixels per row, two padding bytes. BGR spatial means = 25,50,75.
  const uint8_t source[]={10,20,30,20,40,60,255,255,
                          30,60,90,40,80,120,255,255};
  DeviceMemory input,output;
  ASSERT_EQ(cudaMalloc(&input.pointer,sizeof(source)),cudaSuccess);
  ASSERT_EQ(cudaMalloc(&output.pointer,3*sizeof(float)),cudaSuccess);
  ASSERT_EQ(cudaMemcpy(input.pointer,source,sizeof(source),cudaMemcpyHostToDevice),cudaSuccess);
  ros2_trt::Normalization norm{0.01f,{0.1f,0.2f,0.3f},{0.5f,1.0f,2.0f}};
  ros2_trt::preprocess(static_cast<const uint8_t*>(input.pointer),2,2,8,true,
    static_cast<float*>(output.pointer),1,1,norm,nullptr);
  ASSERT_EQ(cudaGetLastError(),cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(),cudaSuccess);
  float result[3];
  ASSERT_EQ(cudaMemcpy(result,output.pointer,sizeof(result),cudaMemcpyDeviceToHost),cudaSuccess);
  EXPECT_NEAR(result[0],1.3f,1e-6f);  // (75*.01-.1)/.5
  EXPECT_NEAR(result[1],0.3f,1e-6f);  // (50*.01-.2)/1
  EXPECT_NEAR(result[2],-0.025f,1e-6f); // (25*.01-.3)/2
}
