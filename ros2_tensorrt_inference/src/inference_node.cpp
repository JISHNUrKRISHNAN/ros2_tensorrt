#include "ros2_tensorrt_inference/core.hpp"
#include "ros2_tensorrt_inference/engine.hpp"
#include "ros2_tensorrt_inference/msg/inference_result.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
namespace ros2_trt {
using Clock=std::chrono::steady_clock;
using Image=sensor_msgs::msg::Image;
using Result=ros2_tensorrt_inference::msg::InferenceResult;
static double milliseconds(Clock::duration duration) {
  return std::chrono::duration<double,std::milli>(duration).count();
}
class InferenceNode final : public rclcpp::Node {
 public:
  InferenceNode():Node("tensorrt_inference") {
    // Configuration is a startup contract: accepted settings cannot change under the worker.
    rcl_interfaces::msg::ParameterDescriptor descriptor; descriptor.read_only=true;
    auto string_param=[this,&descriptor](const char* key,const char* value) {
      return declare_parameter<std::string>(key,value,descriptor);
    };
    EngineConfig config;
    config.path=string_param("engine_path","");
    if(config.path.empty()) throw std::invalid_argument("engine_path is required");
    const int64_t device=declare_parameter<int64_t>("device_id",0,descriptor);
    const int64_t capacity=declare_parameter<int64_t>("max_image_bytes",6220800,descriptor);
    const int64_t warmup=declare_parameter<int64_t>("warmup_iterations",20,descriptor);
    const int64_t k=declare_parameter<int64_t>("top_k",5,descriptor);
    if(device<0 || device>std::numeric_limits<int>::max() || capacity<3 ||
       static_cast<uint64_t>(capacity)>std::numeric_limits<size_t>::max() || warmup<0 || k<=0)
      throw std::invalid_argument("Invalid device, capacity, warmup, or top_k");
    config.device=static_cast<int>(device); config.max_image_bytes=static_cast<size_t>(capacity);
    capacity_=config.max_image_bytes; top_k_=static_cast<size_t>(k);
    config.scale=static_cast<float>(declare_parameter<double>("scale",1.0/255.0,descriptor));
    auto mean=declare_parameter<std::vector<double>>("mean",{0.485,0.456,0.406},descriptor);
    auto stddev=declare_parameter<std::vector<double>>("stddev",{0.229,0.224,0.225},descriptor);
    if(mean.size()!=3 || stddev.size()!=3) throw std::invalid_argument("mean/stddev need three RGB values");
    for(size_t i=0;i<3;++i){config.mean[i]=static_cast<float>(mean[i]);config.stddev[i]=static_cast<float>(stddev[i]);}
    mode_=string_param("output_mode","raw");
    if(mode_!="raw" && mode_!="classification" && mode_!="both")
      throw std::invalid_argument("output_mode must be raw, classification, or both");
    softmax_=declare_parameter<bool>("apply_softmax",true,descriptor);
    const auto input_topic=string_param("image_topic","/camera/image_raw");
    const auto output_topic=string_param("result_topic","/inference/results");
    engine_=std::make_unique<Engine>(config);
    if(mode_!="raw") {
      const auto& outputs=engine_->outputs();
      if(outputs.size()!=1 || !(outputs[0].shape.size()==1 ||
          (outputs[0].shape.size()==2 && outputs[0].shape[0]==1)))
        throw std::invalid_argument("Classification requires exactly one [K] or [1,K] output");
    }
    // Warmup is excluded from measured results and occurs before accepting camera frames.
    const uint8_t pixel[3]={0,0,0};
    for(int64_t i=0;i<warmup;++i) engine_->infer(pixel,3,1,1,3,false);
    auto qos=rclcpp::SensorDataQoS().keep_last(1);
    publisher_=create_publisher<Result>(output_topic,qos);
    subscription_=create_subscription<Image>(input_topic,qos,
      [this](Image::ConstSharedPtr image){receive(std::move(image));});
    RCLCPP_INFO(get_logger(),"Ready: %dx%d RGB NCHW, mode=%s, input=%s output=%s",
      engine_->width(),engine_->height(),mode_.c_str(),input_topic.c_str(),output_topic.c_str());
    worker_=std::thread([this]{run();});
  }
  ~InferenceNode() override {
    {std::lock_guard<std::mutex> lock(mutex_); stopping_=true; pending_.reset();}
    condition_.notify_one();
    if(worker_.joinable()) worker_.join(); // Never destroy context/buffers while CUDA is running.
  }
  bool failed() const {return failed_.load();}
 private:
  void receive(Image::ConstSharedPtr image) {
    const auto receipt=Clock::now();
    ++received_;
    try {validate_image(image->width,image->height,image->step,image->data.size(),image->encoding,capacity_);}
    catch(const std::exception& error) {
      ++invalid_;
      RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),5000,"Rejected image: %s",error.what());
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if(stopping_) return;
      if(pending_) ++overwritten_;
      pending_=std::move(image); pending_time_=receipt;
    }
    condition_.notify_one();
  }
  void run() noexcept {
    try {
      while(true) {
        Image::ConstSharedPtr frame; Clock::time_point receipt;
        {
          std::unique_lock<std::mutex> lock(mutex_);
          condition_.wait(lock,[this]{return stopping_ || pending_;});
          if(stopping_) return;
          frame=std::move(pending_); receipt=pending_time_;
        }
        const auto begin=Clock::now();
        const auto& outputs=engine_->infer(frame->data.data(),frame->data.size(),
          static_cast<int>(frame->width),static_cast<int>(frame->height),frame->step,frame->encoding=="bgr8");
        Result result;
        result.header=frame->header; result.sequence=++sequence_; result.backend="tensorrt";
        if(mode_=="raw" || mode_=="both") {
          result.tensors.reserve(outputs.size());
          for(const auto& output:outputs) {
            ros2_tensorrt_inference::msg::Tensor tensor;
            tensor.name=output.name; tensor.shape=output.shape;
            tensor.data.assign(output.data,output.data+output.count);
            result.tensors.push_back(std::move(tensor));
          }
        }
        if(mode_!="raw") {
          auto decoded=top_k(outputs[0].data,outputs[0].count,top_k_,softmax_);
          result.class_ids=std::move(decoded.first); result.scores=std::move(decoded.second);
        }
        const auto finished=Clock::now();
        result.queue_ms=milliseconds(begin-receipt);
        result.processing_ms=milliseconds(finished-begin);
        result.receipt_to_result_ms=milliseconds(finished-receipt);
        result.gpu_pipeline_ms=engine_->gpu_ms();
        result.received_total=received_; result.overwritten_total=overwritten_; result.invalid_total=invalid_;
        publisher_->publish(std::move(result));
      }
    } catch(const std::exception& error) {
      failed_=true;
      RCLCPP_FATAL(get_logger(),"Inference worker failed: %s",error.what());
      // A CUDA failure can poison the context; stop instead of publishing stale results.
      rclcpp::shutdown();
    } catch(...) {
      failed_=true; RCLCPP_FATAL(get_logger(),"Unknown inference failure"); rclcpp::shutdown();
    }
  }
  std::unique_ptr<Engine> engine_;
  rclcpp::Publisher<Result>::SharedPtr publisher_;
  rclcpp::Subscription<Image>::SharedPtr subscription_;
  std::string mode_;
  size_t capacity_=0,top_k_=5;
  bool softmax_=true,stopping_=false;
  std::atomic<bool> failed_{false};
  std::atomic<uint64_t> received_{0},overwritten_{0},invalid_{0};
  uint64_t sequence_=0;
  std::mutex mutex_;
  std::condition_variable condition_;
  Image::ConstSharedPtr pending_;
  Clock::time_point pending_time_;
  std::thread worker_; // Last member; explicit destructor joins before other members disappear.
};
}
int main(int argc,char** argv) {
  rclcpp::init(argc,argv);
  try {
    auto node=std::make_shared<ros2_trt::InferenceNode>();
    rclcpp::spin(node);
    const bool failed=node->failed(); node.reset();
    if(rclcpp::ok()) rclcpp::shutdown();
    return failed ? 1 : 0;
  } catch(const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("tensorrt_inference"),"Startup/executor failure: %s",error.what());
    if(rclcpp::ok()) rclcpp::shutdown();
    return 1;
  }
}
