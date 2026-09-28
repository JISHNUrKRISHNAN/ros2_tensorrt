#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
namespace ros2_trt {
struct EngineConfig {
  std::string path;
  int device=0;
  size_t max_image_bytes=1920*1080*3;
  float scale=1.0f/255.0f;
  std::array<float,3> mean{0.485f,0.456f,0.406f};
  std::array<float,3> stddev{0.229f,0.224f,0.225f};
};
struct OutputView {
  std::string name;
  std::vector<int64_t> shape;
  const float* data=nullptr;
  size_t count=0;
};
// One context, one stream, one worker. Views remain valid only until next infer/destruction.
class Engine {
 public:
  explicit Engine(const EngineConfig& config);
  ~Engine();
  Engine(const Engine&)=delete;
  Engine& operator=(const Engine&)=delete;
  const std::vector<OutputView>& infer(const uint8_t* image, size_t bytes,
      int width, int height, size_t step, bool bgr);
  const std::vector<OutputView>& outputs() const;
  float gpu_ms() const;
  int width() const;
  int height() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace ros2_trt
