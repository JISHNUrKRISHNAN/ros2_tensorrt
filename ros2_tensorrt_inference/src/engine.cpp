#include "ros2_tensorrt_inference/engine.hpp"
#include "ros2_tensorrt_inference/core.hpp"
#include "ros2_tensorrt_inference/preprocess.hpp"
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferVersion.h>
#include <cuda_runtime.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
//#if NV_TENSORRT_MAJOR != 10
//#error "This package targets TensorRT 10.x. Validate/migrate before using another major version."
//#endif
namespace ros2_trt {
namespace {
void check(cudaError_t status,const char* where) {
  if(status!=cudaSuccess) throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(status));
}
struct Logger : nvinfer1::ILogger {
  void log(Severity severity,const char* message) noexcept override {
    if(severity<=Severity::kWARNING) std::cerr << "[TensorRT] " << message << '\n';
  }
};
// Noncopyable resource owners also protect constructor failure paths.
struct Buffer {
  void* device=nullptr; void* host=nullptr;
  Buffer()=default; Buffer(const Buffer&)=delete; Buffer& operator=(const Buffer&)=delete;
  ~Buffer(){ if(device) cudaFree(device); if(host) cudaFreeHost(host); }
  void allocate(size_t bytes,bool pinned) {
    check(cudaMalloc(&device,bytes),"cudaMalloc");
    if(pinned) check(cudaMallocHost(&host,bytes),"cudaMallocHost");
  }
};
struct Stream {
  cudaStream_t value=nullptr;
  ~Stream(){if(value) cudaStreamDestroy(value);}
};
struct Event {
  cudaEvent_t value=nullptr;
  ~Event(){if(value) cudaEventDestroy(value);}
};
std::vector<int64_t> shape_of(nvinfer1::Dims dims) {
  if(dims.nbDims<=0) throw std::runtime_error("Invalid or scalar tensor shape");
  return {dims.d,dims.d+dims.nbDims};
}
}
struct Engine::Impl {
  EngineConfig config;
  Logger logger; // Must outlive runtime, engine, and context.
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  Stream stream;
  Event start,stop;
  Buffer image,input;
  std::vector<std::unique_ptr<Buffer>> output_buffers;
  std::vector<OutputView> views;
  int width=0,height=0;
  float elapsed=0;
  ~Impl(){
    // Drain even when an exception occurred after enqueue, before freeing any buffer.
    cudaSetDevice(config.device);
    if(stream.value) cudaStreamSynchronize(stream.value);
  }
};
Engine::Engine(const EngineConfig& config):impl_(std::make_unique<Impl>()) {
  auto& p=*impl_; p.config=config;
  validate_normalization(config.mean,config.stddev,config.scale);
  if(config.max_image_bytes==0) throw std::invalid_argument("max_image_bytes must be positive");
  check(cudaSetDevice(config.device),"cudaSetDevice");
  if(!initLibNvInferPlugins(&p.logger,"")) throw std::runtime_error("TensorRT plugin initialization failed");
  std::ifstream file(config.path,std::ios::binary|std::ios::ate);
  if(!file) throw std::runtime_error("Cannot open engine: "+config.path);
  const auto end=file.tellg();
  if(end<=0) throw std::runtime_error("Engine is empty/unreadable");
  std::vector<char> bytes(static_cast<size_t>(end));
  file.seekg(0);
  if(!file.read(bytes.data(),static_cast<std::streamsize>(bytes.size())))
    throw std::runtime_error("Failed reading engine file");
  p.runtime.reset(nvinfer1::createInferRuntime(p.logger));
  if(!p.runtime) throw std::runtime_error("createInferRuntime failed");
  p.engine.reset(p.runtime->deserializeCudaEngine(bytes.data(),bytes.size()));
  if(!p.engine) throw std::runtime_error("Engine deserialization failed; check GPU/TRT compatibility");
  p.context.reset(p.engine->createExecutionContext());
  if(!p.context) throw std::runtime_error("createExecutionContext failed");
  check(cudaStreamCreateWithFlags(&p.stream.value,cudaStreamNonBlocking),"create stream");
  check(cudaEventCreate(&p.start.value),"create start event");
  check(cudaEventCreate(&p.stop.value),"create stop event");
  int inputs=0;
  for(int i=0;i<p.engine->getNbIOTensors();++i) {
    const char* name=p.engine->getIOTensorName(i);
    if(p.engine->getTensorDataType(name)!=nvinfer1::DataType::kFLOAT ||
       p.engine->getTensorFormat(name)!=nvinfer1::TensorFormat::kLINEAR ||
       p.engine->getTensorLocation(name)!=nvinfer1::TensorLocation::kDEVICE ||
       p.engine->isShapeInferenceIO(name))
      throw std::runtime_error(std::string("Requires device FP32 LINEAR execution I/O: ")+name);
    const auto shape=shape_of(p.engine->getTensorShape(name));
    const auto count=checked_volume(shape); // Reject dynamic dimensions before allocation.
    if(p.engine->getTensorIOMode(name)==nvinfer1::TensorIOMode::kINPUT) {
      if(++inputs!=1 || shape.size()!=4 || shape[0]!=1 || shape[1]!=3 ||
         shape[2]>std::numeric_limits<int>::max() || shape[3]>std::numeric_limits<int>::max() ||
         static_cast<uint64_t>(shape[2])*shape[3] > 2147483647ULL)
        throw std::runtime_error("Requires exactly one static NCHW [1,3,H,W] input");
      p.height=static_cast<int>(shape[2]); p.width=static_cast<int>(shape[3]);
      p.input.allocate(count*sizeof(float),false);
      if(!p.context->setTensorAddress(name,p.input.device)) throw std::runtime_error("Input address rejected");
    } else {
      auto buffer=std::make_unique<Buffer>(); buffer->allocate(count*sizeof(float),true);
      if(!p.context->setTensorAddress(name,buffer->device)) throw std::runtime_error("Output address rejected");
      p.views.push_back({name,shape,static_cast<const float*>(buffer->host),count});
      p.output_buffers.push_back(std::move(buffer));
    }
  }
  if(inputs!=1 || p.views.empty()) throw std::runtime_error("Engine requires one input and at least one output");
  p.image.allocate(config.max_image_bytes,true);
}
Engine::~Engine()=default;
const std::vector<OutputView>& Engine::infer(const uint8_t* data,size_t bytes,int width,int height,size_t step,bool bgr) {
  auto& p=*impl_;
  // Primary CUDA contexts are per device; the worker sets its thread-local device explicitly.
  check(cudaSetDevice(p.config.device),"worker cudaSetDevice");
  if(!data || width<=0 || height<=0 || step>std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("Invalid inference image");
  const auto needed=validate_image(width,height,static_cast<uint32_t>(step),bytes,
    bgr ? "bgr8" : "rgb8",p.config.max_image_bytes);
  std::memcpy(p.image.host,data,needed);
  check(cudaEventRecord(p.start.value,p.stream.value),"record start");
  check(cudaMemcpyAsync(p.image.device,p.image.host,needed,cudaMemcpyHostToDevice,p.stream.value),"image H2D");
  Normalization norm{p.config.scale,{p.config.mean[0],p.config.mean[1],p.config.mean[2]},
    {p.config.stddev[0],p.config.stddev[1],p.config.stddev[2]}};
  preprocess(static_cast<const uint8_t*>(p.image.device),width,height,step,bgr,
    static_cast<float*>(p.input.device),p.width,p.height,norm,p.stream.value);
  check(cudaGetLastError(),"preprocess launch");
  if(!p.context->enqueueV3(p.stream.value)) throw std::runtime_error("TensorRT enqueueV3 failed");
  for(size_t i=0;i<p.views.size();++i)
    check(cudaMemcpyAsync(p.output_buffers[i]->host,p.output_buffers[i]->device,
      p.views[i].count*sizeof(float),cudaMemcpyDeviceToHost,p.stream.value),"output D2H");
  check(cudaEventRecord(p.stop.value,p.stream.value),"record stop");
  check(cudaStreamSynchronize(p.stream.value),"inference synchronization");
  check(cudaEventElapsedTime(&p.elapsed,p.start.value,p.stop.value),"elapsed GPU time");
  return p.views;
}
const std::vector<OutputView>& Engine::outputs() const {return impl_->views;}
float Engine::gpu_ms() const {return impl_->elapsed;}
int Engine::width() const {return impl_->width;}
int Engine::height() const {return impl_->height;}
}
