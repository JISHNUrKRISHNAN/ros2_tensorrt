#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace ros2_trt {
// A single place to validate externally supplied dimensions before multiplication.
inline size_t checked_volume(const std::vector<int64_t>& shape) {
  if (shape.empty()) throw std::invalid_argument("Scalar outputs are unsupported");
  size_t count = 1;
  for (auto dim : shape) {
    if (dim <= 0 || static_cast<uint64_t>(dim) > std::numeric_limits<size_t>::max() / count)
      throw std::invalid_argument("Nonpositive/dynamic/overflowing tensor shape");
    count *= static_cast<size_t>(dim);
  }
  if (count > std::numeric_limits<size_t>::max() / sizeof(float))
    throw std::invalid_argument("Tensor byte count overflows");
  return count;
}
inline size_t validate_image(uint32_t width, uint32_t height, uint32_t step,
                            size_t data_size, const std::string& encoding, size_t capacity) {
  if (encoding != "rgb8" && encoding != "bgr8")
    throw std::invalid_argument("Only rgb8/bgr8 images are supported");
  if (!width || !height || width > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
      height > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
      static_cast<uint64_t>(step) < static_cast<uint64_t>(width) * 3)
    throw std::invalid_argument("Invalid image dimensions/step");
  const uint64_t bytes = static_cast<uint64_t>(step) * height;
  if (bytes > data_size || bytes > capacity)
    throw std::invalid_argument("Image data truncated or exceeds max_image_bytes");
  return static_cast<size_t>(bytes);
}
inline void validate_normalization(const std::array<float,3>& mean,
                                   const std::array<float,3>& stddev, float scale) {
  if (!std::isfinite(scale) || scale <= 0) throw std::invalid_argument("scale must be positive and finite");
  for (size_t i=0; i<3; ++i)
    if (!std::isfinite(mean[i]) || !std::isfinite(stddev[i]) || stddev[i] <= 0)
      throw std::invalid_argument("mean must be finite and stddev positive and finite");
}
inline std::pair<std::vector<int32_t>, std::vector<float>> top_k(
    const float* values, size_t count, size_t k, bool softmax) {
  if (!count || !k || count > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
    throw std::invalid_argument("Invalid classification size/top_k");
  std::vector<float> scores(values, values + count);
  for (float x : scores) if (!std::isfinite(x)) throw std::runtime_error("Nonfinite model output");
  if (softmax) {
    const float maximum = *std::max_element(scores.begin(), scores.end());
    double sum=0;
    for (float& x : scores) { x=std::exp(x-maximum); sum += x; }
    for (float& x : scores) x=static_cast<float>(x/sum);
  }
  std::vector<int32_t> ids(count);
  std::iota(ids.begin(), ids.end(), 0);
  k=std::min(k,count);
  std::partial_sort(ids.begin(), ids.begin()+k, ids.end(), [&scores](int32_t a,int32_t b){
    return scores[a] == scores[b] ? a < b : scores[a] > scores[b];
  });
  ids.resize(k);
  std::vector<float> selected; selected.reserve(k);
  for (auto id : ids) selected.push_back(scores[id]);
  return {std::move(ids),std::move(selected)};
}
} // namespace ros2_trt
