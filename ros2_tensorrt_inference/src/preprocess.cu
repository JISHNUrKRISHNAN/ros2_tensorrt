#include "ros2_tensorrt_inference/preprocess.hpp"
namespace ros2_trt {
__global__ void resize_normalize(const uint8_t* src, int sw, int sh, size_t step,
    bool bgr, float* dst, int dw, int dh, Normalization norm) {
  const size_t i=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  const size_t plane=static_cast<size_t>(dw)*dh;
  if (i>=plane) return;
  const int x=static_cast<int>(i%dw), y=static_cast<int>(i/dw);
  // Half-pixel bilinear resize, clamped at the edges. This is stretch, not letterbox.
  const float sx=fminf(fmaxf((x+0.5f)*sw/dw-0.5f,0.0f),static_cast<float>(sw-1));
  const float sy=fminf(fmaxf((y+0.5f)*sh/dh-0.5f,0.0f),static_cast<float>(sh-1));
  const int x0=static_cast<int>(sx), y0=static_cast<int>(sy);
  const int x1=min(x0+1,sw-1), y1=min(y0+1,sh-1);
  const float ax=sx-x0, ay=sy-y0;
  for(int c=0;c<3;++c) {
    const int sc=bgr ? 2-c : c;
    const float a=src[static_cast<size_t>(y0)*step+static_cast<size_t>(x0)*3+sc];
    const float b=src[static_cast<size_t>(y0)*step+static_cast<size_t>(x1)*3+sc];
    const float d=src[static_cast<size_t>(y1)*step+static_cast<size_t>(x0)*3+sc];
    const float e=src[static_cast<size_t>(y1)*step+static_cast<size_t>(x1)*3+sc];
    const float pixel=(a+(b-a)*ax)*(1-ay)+(d+(e-d)*ax)*ay;
    dst[static_cast<size_t>(c)*plane+i]=(pixel*norm.scale-norm.mean[c])/norm.stddev[c];
  }
}
void preprocess(const uint8_t* input,int width,int height,size_t step,bool bgr,
                float* output,int out_width,int out_height,Normalization norm,cudaStream_t stream) {
  const size_t n=static_cast<size_t>(out_width)*out_height;
  resize_normalize<<<static_cast<unsigned int>((n+255)/256),256,0,stream>>>(
    input,width,height,step,bgr,output,out_width,out_height,norm);
}
}
