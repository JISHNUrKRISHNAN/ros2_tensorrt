# ROS 2 TensorRT inference — laptop demo and reusable runtime

A complete ROS 2 source package for an Ubuntu/NVIDIA laptop. The real inference node is C++17 + CUDA + TensorRT 10, with GPU image preprocessing, reusable pinned/device buffers, a latest-frame mailbox, typed results, and a throughput benchmark.

**Performance goal: more than 30 delivered results/second. This is a target, not a measured GPU claim.** Camera FPS, model, resolution, GPU, power settings, and ROS transport determine actual performance. See [performance.md](docs/performance.md).

## Start here

1. Follow [the laptop demo guide](docs/laptop_demo.md) for setup, engine creation, build, browser demo, and measurement.
2. Read [the code walkthrough](docs/code_walkthrough.md) for an explanation of every source file and function.
3. Read [the model contract and parameters](docs/model_contract.md) before substituting a trained model.
4. Check [validation status](docs/validation.md) to distinguish tests actually run from hardware checks still required.

The basic demo is a **deterministic RGB mean model**, not a trained classifier. It takes a `[1,3,224,224]` image and returns three mean channel intensities. Moving colors make inference visible in a browser. The same tiny ONNX graph runs through the real TensorRT backend. A clearly labeled CPU reference is included for rehearsal and numerical comparison, and never silently replaces TensorRT.

## Pipeline

```text
/camera/image_raw (rgb8 or bgr8)
          |
          v
ROS callback: validate image, replace one pending frame
          |
          v
Single worker: copy to pinned host memory -> asynchronous H2D
          |
          v
CUDA: bilinear resize + BGR/RGB conversion + normalize + HWC-to-NCHW
          |
          v
TensorRT: setTensorAddress (startup), enqueueV3 (per frame)
          |
          v
D2H -> synchronize -> optional top-k -> /inference/results
          |                                 |
          v                                 v
browser dashboard                     benchmark.py
```

The queue holds one pending frame in addition to the frame being processed. A newer frame replaces an older pending frame. This bounds application backlog; it does not guarantee zero sensor/DDS delay. Input and result QoS are best effort, keep last 1.

## Supported baseline

| Item | Contract |
|---|---|
| OS / ROS | Primary: Ubuntu 24.04 + ROS 2 Jazzy; intended source compatibility with Ubuntu 22.04 + Humble, not locally tested |
| TensorRT | 10.x C++ SDK, standard runtime and standard plugins; other major versions rejected at compile time |
| CUDA | Toolkit and driver supported by the selected TensorRT 10 release and GPU |
| Engine input | Exactly one fixed-size FP32, LINEAR, device input `[1,3,H,W]`, RGB NCHW |
| Engine outputs | One or more fixed-size FP32 LINEAR device tensors; raw mode publishes all |
| Precision | FP32 boundary buffers; optional FP16 kernels inside the engine |
| Images | `sensor_msgs/Image`, `rgb8` or `bgr8`, row padding supported |
| Classification | Optional, exactly one `[K]` or `[1,K]` output, top-k and optional stable softmax |
| Demo | Spatial RGB mean, fixed 224×224; mean=0, std=1, scale=1/255 |

Dynamic shapes, batching >1, FP16/INT8 boundary buffers, shape tensors, letterboxing, object-detection decoding/NMS, and segmentation decoding are deliberately outside this baseline. Unsupported engine contracts fail at startup. Raw mode can transport detector outputs, but does **not** turn them into detections.

## Package contents

```text
ros2_tensorrt_inference/
├── CMakeLists.txt, package.xml, LICENSE
├── include/ros2_tensorrt_inference/
│   ├── core.hpp           # CPU validation and classification utilities
│   ├── engine.hpp         # Runtime API, configuration, output views
│   └── preprocess.hpp     # CUDA preprocessing API
├── src/
│   ├── engine.cpp         # Resource ownership, engine validation, inference
│   ├── preprocess.cu      # Fused GPU preprocessing
│   └── inference_node.cpp # ROS parameters, latest-frame worker, publishing
├── models/rgb_mean.onnx    # Bundled, validated demo graph
├── msg/                   # Tensor and InferenceResult interfaces
├── config/                # Generic and RGB demo normalization/settings
├── launch/                # Generic inference and one-command laptop demo
├── scripts/               # Engine builder, ONNX fixture, source, dashboard,
│                          # CPU reference, numerical verifier, benchmark
├── test/                  # CPU correctness tests and GPU preprocessing test
└── docs/                  # Setup, code explanation, model contract, performance
```

## Fast rehearsal without GPU tools

After sourcing ROS, put this folder under a workspace's `src/`, then:

```bash
colcon build --packages-select ros2_tensorrt_inference --cmake-args -DBUILD_GPU_NODE=OFF
source install/setup.bash
ros2 launch ros2_tensorrt_inference demo.launch.py backend:=cpu_reference
```

Open **http://127.0.0.1:8765**. This uses ROS 2 and NumPy, with no TensorRT acceleration. Install ROS dependencies as described in the demo guide. For your NVIDIA laptop, follow the full GPU build guide instead.

## References

The runtime follows NVIDIA's [TensorRT 10 named-tensor migration guidance](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/api/tensorrt-8x-to-10x-c-api.html). TensorRT 10 and 11 precision/build APIs differ, so the package intentionally fixes the major version. Consult NVIDIA's [TensorRT installation guide](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/installing-tensorrt/installing.html) and [10-to-11 migration guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x.html) before changing versions.

The tiny demo ONNX graph is bundled under `models/`. No trained weights or GPU-specific serialized engine are bundled. Build an engine on the target laptop. Only load engines you trust. Replace the example package maintainer before publishing this as your own project.
