#include "ros2_tensorrt_inference/core.hpp"
#include <gtest/gtest.h>
using namespace ros2_trt;
TEST(Shape, RejectsDynamicZeroAndOverflow) {
  EXPECT_EQ(checked_volume({1,3,224,224}),150528u);
  EXPECT_THROW(checked_volume({1,-1,224}),std::invalid_argument);
  EXPECT_THROW(checked_volume({0}),std::invalid_argument);
  EXPECT_THROW(checked_volume({}),std::invalid_argument);
  EXPECT_THROW(checked_volume({INT64_MAX,INT64_MAX}),std::invalid_argument);
}
TEST(Image, RespectsStrideAndRejectsTruncation) {
  EXPECT_EQ(validate_image(2,2,8,16,"bgr8",16),16u);
  EXPECT_THROW(validate_image(2,2,8,15,"bgr8",16),std::invalid_argument);
  EXPECT_THROW(validate_image(2,2,8,16,"rgb8",12),std::invalid_argument);
  EXPECT_THROW(validate_image(2,2,5,16,"rgb8",16),std::invalid_argument);
  EXPECT_THROW(validate_image(2,2,8,16,"mono8",16),std::invalid_argument);
  EXPECT_THROW(validate_image(0,2,8,16,"rgb8",16),std::invalid_argument);
}
TEST(Decode, StableSoftmaxAndDeterministicTies) {
  const float x[]={1000,1001,1001};
  const auto result=top_k(x,3,9,true);
  EXPECT_EQ(result.first,(std::vector<int32_t>{1,2,0}));
  EXPECT_NEAR(std::accumulate(result.second.begin(),result.second.end(),0.0f),1.0f,1e-6f);
  EXPECT_NEAR(result.second[0],0.4223188f,1e-6f);
}
TEST(Decode, PreservesScoresAndRejectsNonfinite) {
  const float x[]={0.1f,0.8f,0.1f};
  auto result=top_k(x,3,1,false);
  EXPECT_EQ(result.first[0],1); EXPECT_FLOAT_EQ(result.second[0],0.8f);
  const float invalid[]={std::numeric_limits<float>::infinity()};
  EXPECT_THROW(top_k(invalid,1,1,true),std::runtime_error);
  EXPECT_THROW(top_k(x,3,0,false),std::invalid_argument);
}
TEST(Normalization, RejectsZeroStddev) {
  EXPECT_NO_THROW(validate_normalization({0,0,0},{1,1,1},1.0f/255));
  EXPECT_THROW(validate_normalization({0,0,0},{1,0,1},1),std::invalid_argument);
  EXPECT_THROW(validate_normalization({0,0,0},{1,1,1},0),std::invalid_argument);
}
