# Validation status

Delivery checks completed on 2026-09-26:

| Check | Result |
|---|---|
| ROS 2 Jazzy interface generation and CPU/tool installation | Passed with `colcon build`, GPU option OFF |
| TensorRT engine source syntax against official TensorRT 10.13 and CUDA 12.9 headers | Passed; no GPU linking/execution |
| Actual C++ ROS node compilation against generated message types | Passed (object compilation; no GPU backend linked) |
| CPU gtest cases | 5 passed: dimensions, image bounds/stride, softmax/top-k, normalization |
| Python parse and shell syntax | Passed |
| Bundled ONNX graph | ONNX checker passed; ONNX reference evaluator returned the expected RGB means |
| Live ROS CPU reference numerical verifier | 10 correct result frames |
| Browser dashboard | Visually inspected with live image, changing channels and explicit CPU badge |
| Delivered-result benchmark in CPU rehearsal | 377 results in 10 seconds, 37.7 FPS; p95 processing 17.06 ms; not a GPU result |
| Required TensorRT backend against CPU stream | Correctly rejected with exit code 2 |
| HTTP state and orderly CPU node shutdown | Passed |

The CPU rehearsal used 640×480 input, a requested 60 Hz source, the 224×224 fixture and the dashboard. Its measured rate is environment-specific and **does not establish TensorRT performance**. The actual C++ GPU worker was not executed. CUDA kernel compilation, backend linking, engine creation and GPU execution remain unverified.  GPU compilation, engine construction, inference correctness and TensorRT FPS need a functioning NVIDIA driver plus CUDA/TensorRT SDK on the target laptop.

## Reproduce CPU and interface checks

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ros2_tensorrt_inference   --cmake-args -DBUILD_GPU_NODE=OFF -DBUILD_TESTING=ON
colcon test --packages-select ros2_tensorrt_inference
colcon test-result --verbose
```

## Reproduce GPU checks

```bash
colcon build --packages-select ros2_tensorrt_inference   --cmake-args -DBUILD_GPU_NODE=ON -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
colcon test --packages-select ros2_tensorrt_inference
colcon test-result --verbose
```

Then build the ONNX fixture, run the nonanimated source and real inference node with demo normalization, run `verify_fixture.py --backend tensorrt`, and execute the FPS benchmark. Also test BGR camera input, actual camera stride/resolution, overload, Ctrl+C shutdown, and an invalid engine path on the target system.

A skipped CUDA test or a CPU-reference result does not validate TensorRT execution.
