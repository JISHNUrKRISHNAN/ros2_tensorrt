# Ubuntu + NVIDIA laptop demo

## 1. Prepare the software

Use Ubuntu 24.04 with ROS 2 Jazzy, or Ubuntu 22.04 with ROS 2 Humble. Install ROS using the official [Jazzy Ubuntu guide](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html) or [Humble Ubuntu guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html). Commands below use Jazzy; substitute `humble` on Ubuntu 22.04.

```bash
source /opt/ros/jazzy/setup.bash
sudo apt update
sudo apt install build-essential cmake python3-colcon-common-extensions python3-rosdep   python3-numpy python3-pil python3-venv ros-jazzy-ament-cmake-gtest
nvidia-smi
nvcc --version
```

`nvidia-smi` must show your NVIDIA GPU and driver. If it fails, fix the driver first; a CPU rehearsal still works. `nvcc` must exist for the CUDA preprocessing kernel. Install a CUDA toolkit and **TensorRT 10.x C++ SDK** compatible with your GPU and driver using NVIDIA's [installation instructions](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/installing-tensorrt/installing.html). A Python-only TensorRT installation is insufficient: this project needs `NvInfer.h`, `libnvinfer`, `libnvinfer_plugin`, and `trtexec`. Do not install an unspecified latest major version and assume compatibility.

For a TensorRT tar installation, adapt this one path:

```bash
export TRT_ROOT="$HOME/opt/TensorRT-10.x.x.x"
export PATH="$TRT_ROOT/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$TRT_ROOT/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
trtexec --help
```

For a system installation, `trtexec` may be under `/usr/src/tensorrt/bin/`; set `TRTEXEC` to its full path when using the builder script. Do not overwrite a working driver/toolkit merely to match an example path. The CUDA version reported by `nvidia-smi` is a driver capability, not confirmation that a toolkit is installed.

## 2. Create a ROS workspace

Extract the supplied ZIP and place the **entire** `ros2_tensorrt_inference` folder under `~/trt_demo_ws/src/`. You should have `~/trt_demo_ws/src/ros2_tensorrt_inference/package.xml`.

```bash
mkdir -p ~/trt_demo_ws/src
cd ~/trt_demo_ws
# Copy/extract the package into src/ before continuing.
# If rosdep has never been initialized: sudo rosdep init
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

CUDA/TensorRT are external SDK prerequisites, not installed by this package's rosdep manifest. Use the ROS Python interpreter; don't activate a model-export virtual environment while building ROS.

## 3. Build the real GPU node

For system-installed CUDA/TensorRT:

```bash
cd ~/trt_demo_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ros2_tensorrt_inference   --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_GPU_NODE=ON -DBUILD_TESTING=ON
source install/setup.bash
```

For a TensorRT tar installation, add `-DTENSORRT_ROOT="$TRT_ROOT"` to the CMake arguments. If CUDA is not found, add `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`. For portability, specify your GPU architecture with `-DCMAKE_CUDA_ARCHITECTURES=<compute-capability-without-dot>` after looking it up; otherwise CMake uses the compiler's default architecture. Never guess the architecture. CUDA/TensorRT releases can drop support for older GPUs.

The GPU executable is built by default. `BUILD_GPU_NODE=OFF` builds only interfaces, CPU tests, and demo tools. It does not produce an inference executable that secretly runs on the CPU. After changing the build option, rebuild and source the new install; an old executable can remain in a reused install tree, so use a clean workspace when switching demonstrations.

## 4. Build the basic model

The validated tiny ONNX graph is already bundled. The shortest path is:

```bash
cd ~/trt_demo_ws
mkdir -p models
cp src/ros2_tensorrt_inference/models/rgb_mean.onnx models/rgb_mean.onnx
src/ros2_tensorrt_inference/scripts/build_engine.sh models/rgb_mean.onnx models/rgb_mean.engine fp16
```

Skip the following regeneration commands if you used the bundled graph. To regenerate or change the input size, use a separate virtual environment only for ONNX generation:

```bash
cd ~/trt_demo_ws
python3 -m venv .model-venv
.model-venv/bin/pip install 'onnx>=1.14,<2'
mkdir -p models
.model-venv/bin/python src/ros2_tensorrt_inference/scripts/make_test_model.py   --output models/rgb_mean.onnx
# Keep the ROS terminal outside the virtual environment.
src/ros2_tensorrt_inference/scripts/build_engine.sh   models/rgb_mean.onnx models/rgb_mean.engine fp16
```

If `trtexec` is not on PATH:

```bash
TRTEXEC=/usr/src/tensorrt/bin/trtexec   src/ros2_tensorrt_inference/scripts/build_engine.sh   models/rgb_mean.onnx models/rgb_mean.engine fp16
```

The builder refuses to overwrite an existing engine; choose a new filename when changing precision. The graph is `GlobalAveragePool -> Flatten`. Its three outputs are the spatial means of preprocessed red, green, and blue. FP16 is permitted inside the engine, while external input/output tensors remain FP32 LINEAR. The SDK's `trtexec --help` describes its available options. TensorRT 11 uses different precision rules and is not supported by this builder.

## 5. Run the visual GPU demo

```bash
cd ~/trt_demo_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch ros2_tensorrt_inference demo.launch.py   backend:=tensorrt engine_path:="$PWD/models/rgb_mean.engine" fps:=60.0
```

Open **http://127.0.0.1:8765** in your laptop browser. You will see an animated color frame, the dominant channel, channel intensity bars, delivered result rate, and processing/GPU timings. The badge must say **TensorRT · NVIDIA GPU**. The percentages are average channel intensities, not class confidence probabilities. The preview is the latest image, not a timestamp-matched image/result pair.

The source requests 60 FPS so that a result rate above 30 is possible. ROS timers and actual machine load may produce a lower rate. Browser previews update at 5 Hz independently. Use Ctrl+C to stop all launched nodes. If port 8765 is occupied, pass `port:=8766` and open that port.

## 6. Measure the >30 FPS target

In a second terminal with the same ROS/workspace environment:

```bash
ros2 run ros2_tensorrt_inference benchmark.py   --seconds 30 --warmup-seconds 5 --target-fps 30   --require-backend tensorrt --json "$PWD/benchmark_gpu.json"
```

Exit code 0 means **strictly more than 30 results/second** reached the benchmark subscriber during the measurement window and every sampled backend was TensorRT. Exit code 2 means the target/backend/completion requirement was not met. Keep the JSON with GPU, model, resolution, precision, and power-mode notes. A passing tiny-model result is not evidence that YOLO or another larger model will pass.

For a cleaner performance run, stop the visual demo and run the GPU node and source separately (no dashboard):

```bash
# Terminal A
ros2 launch ros2_tensorrt_inference inference.launch.py   engine_path:="$PWD/models/rgb_mean.engine"   params_file:="$PWD/src/ros2_tensorrt_inference/config/demo.yaml"
# Terminal B
ros2 run ros2_tensorrt_inference publish_test_images.py --ros-args -p fps:=60.0
# Terminal C: use the benchmark command above.
```

## 7. Verify numerical output

With the nonanimated source and GPU node above running:

```bash
ros2 run ros2_tensorrt_inference verify_fixture.py --backend tensorrt
```

For the constant RGB image `[64,128,192]` and demo normalization, the expected raw output is approximately `[0.2509804,0.5019608,0.7529412]`, class order `[2,1,0]`. The verifier checks shape, values, and backend across multiple frames. Do not use it against the animated source or generic ImageNet normalization file.

## 8. Rehearse without CUDA

```bash
colcon build --packages-select ros2_tensorrt_inference   --cmake-args -DBUILD_GPU_NODE=OFF -DBUILD_TESTING=ON
source install/setup.bash
ros2 launch ros2_tensorrt_inference demo.launch.py backend:=cpu_reference
```

This runs the same fixture arithmetic in NumPy and displays a **CPU reference · no TensorRT** badge. It tests the presentation/ROS path and cannot prove GPU speed. The CPU node does not load arbitrary models and does not use the C++ worker architecture. If benchmarking rehearsal, explicitly pass `--require-backend cpu_reference`.

## Optional: use a real camera

Run a ROS camera driver publishing `rgb8` or `bgr8` on `/camera/image_raw` at more than 30 FPS, and launch the inference node with `config/demo.yaml` plus the dashboard separately:

```bash
ros2 run ros2_tensorrt_inference demo_dashboard.py
```

Stop the synthetic source first: two publishers on the same image topic make performance and correctness ambiguous. Configure the driver to output RGB/BGR; this package does not decode MJPEG, YUYV, Bayer, or compressed-image transport. The model then reports the camera scene's mean color. For a trained model, follow the model contract instead of interpreting arbitrary scores as RGB.

## Suggested two-minute presentation

1. Explain the model: "Each frame is resized on the GPU and the model returns three average color intensities."
2. Show the ROS image and result topics and the live TensorRT badge.
3. Point out how dominant colors change with the frame.
4. Show the benchmark JSON: delivered FPS and p95 processing/queue time.
5. Explain latest-frame replacement: an overloaded pipeline favors fresh data rather than growing an old-frame queue.
6. State the scope: basic model demonstration, with trained-model speed and accuracy requiring separate validation.

## Troubleshooting

| Symptom | Action |
|---|---|
| `nvidia-smi` fails | Repair the laptop NVIDIA driver; this package does not install or repair drivers |
| `nvcc`/`NvInfer.h` missing | Install the compatible CUDA toolkit/TensorRT 10 C++ SDK; Python TensorRT alone is insufficient |
| Wrong TensorRT major error | Use TensorRT 10 headers and libraries together; do not mix 10 and 11 |
| `libnvinfer.so.10` not found | Add the matching tar SDK `lib` path to `LD_LIBRARY_PATH`, or fix system library installation |
| Deserialization failure | Rebuild the engine for the current GPU/SDK and register any required custom plugins |
| Dynamic-shape rejection | Export fixed ONNX dimensions; fixed optimization-profile bounds alone do not change this contract |
| No benchmark results | Check topic names, best-effort QoS, matching `ROS_DOMAIN_ID`/RMW and network namespace/firewall configuration |
| Dashboard page fails | Check its process logs, port collision and local socket access; change `port` if needed |
| Incorrect colors/output | Check RGB vs BGR, `demo.yaml` normalization, and whether the source is animated |
| <30 delivered FPS | Verify actual source FPS, disable preview for measurement, use Release build and inspect the performance guide |

Run only one inference publisher per result topic. Benchmark sequence checks cannot reliably disambiguate competing publishers with independent counters.
