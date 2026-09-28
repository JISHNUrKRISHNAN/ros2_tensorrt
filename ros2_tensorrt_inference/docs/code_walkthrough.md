# Code walkthrough

This document explains the responsibilities and control flow of each part of the package. Comments in the sources cover the local implementation choices.

## Build and interfaces

**`package.xml`** declares ROS dependencies, interface generation, Python demo dependencies and gtest. It marks the package as a `rosidl_interface_packages` member because it creates ROS message types. The maintainer is a placeholder to replace before distribution. CUDA and TensorRT are separately installed SDKs.

**`CMakeLists.txt`** selects C++17; finds ament, ROS message libraries and threads; generates `Tensor` and `InferenceResult`; and obtains the C++ typesupport target. With `BUILD_GPU_NODE=ON`, it enables CUDA, locates TensorRT headers/libraries, compiles `trt_backend`, and links `inference_node` to ROS typesupport and the backend. The CUDA kernel uses CUDA C++17. The executable and scripts install under `lib/<package>` so `ros2 run` can locate them. Config, launch and documentation files install in the package share directory. The test block builds CPU unit tests, a ROS node compile check for CPU-only builds, and a CUDA preprocessing test for GPU builds.

**`msg/Tensor.msg`** represents a named dense FP32 tensor without assuming a task. **`msg/InferenceResult.msg`** combines model outputs, an unchanged source image header, backend identity, timing and counters. Keeping the backend in the message prevents a CPU demo from being mistaken for TensorRT in the benchmark.

## CPU utilities: `core.hpp`

`checked_volume` multiplies tensor dimensions safely. It rejects empty/scalar, nonpositive/dynamic, and overflowing shapes before GPU allocation. It checks both element count and FP32 byte count.

`validate_image` checks the supported encoding, integer-safe dimensions, row stride, payload length and preallocated capacity. Using `height*step` is essential: real camera rows can contain padding.

`validate_normalization` rejects nonfinite scale/mean/stddev, nonpositive scale and nonpositive standard deviations. This avoids division by zero and propagating invalid configuration into the CUDA kernel.

`top_k` copies values into a score vector. If requested, it applies numerically stable softmax by subtracting the maximum logit before exponentiation, accumulates the denominator in double precision, then normalizes. It partially sorts class indices for top-k and resolves ties by index. It rejects NaN/infinity in classification output. It deliberately does not label classes or decode detections.

## Runtime interface: `engine.hpp`

`EngineConfig` contains engine path, GPU index, image capacity and normalization. Model dimensions come from the engine itself.

`OutputView` holds a name, dimensions, element count and a read-only pointer into a pinned output buffer. The data is owned by `Engine`; it must be copied before the next inference or destruction. The node does that before publishing.

`Engine` exposes construction, inference, output metadata, GPU pipeline timing and model dimensions. The PImpl pointer hides TensorRT/CUDA headers from ROS code and allows the ROS node to be compiled independently of the GPU SDK. Copying an engine is disabled because the CUDA context/buffer ownership must remain unique.

## Runtime implementation: `engine.cpp`

`check` converts a CUDA error status to an exception with a useful operation name. The worker treats execution errors as fatal because a failed CUDA context must not be reused to publish stale results.

`Logger` forwards TensorRT warnings/errors to stderr. Its declaration precedes runtime/engine/context ownership so reverse destruction keeps it alive long enough.

`Buffer` owns one device allocation and optionally one page-locked host allocation. Its destructor releases both. Partial allocations are safe if later construction throws. `Stream` and `Event` similarly release CUDA stream/event handles. These wrappers are private and not copied.

`shape_of` turns TensorRT dimensions into the message-compatible integer vector and rejects missing/scalar shapes.

`Engine::Impl` owns the runtime, serialized-engine object, execution context, stream, timing events, raw image storage, model input storage, and all output buffers. Before its members are destroyed, its destructor selects the device and drains the stream. This matters when an exception happens after work has been enqueued.

`Engine::Engine` performs these startup steps:

1. Validate normalization/capacity and select the CUDA device.
2. Register standard plugins and read the serialized engine.
3. Create the TensorRT runtime, deserialize the engine and create one execution context.
4. Create a nonblocking CUDA stream and reusable timing events.
5. Inspect every I/O tensor by name. Require FP32, LINEAR, device-resident execution tensors; reject shape I/O and unresolved dimensions.
6. Require exactly one `[1,3,H,W]` input. Allocate its device buffer and bind its address once.
7. Allocate device plus pinned-host output buffers, record output metadata, and bind output addresses.
8. Allocate the reusable raw-image host/device buffers up to `max_image_bytes`.

The compile-time major-version guard intentionally targets TensorRT 10. Deserialization may fail for an engine built with incompatible software, GPU or plugins. Build engines on the target system instead of shipping an arbitrary plan.

`Engine::infer` runs on the single worker. It selects the device for that host thread, validates the image again at the backend boundary, and copies the ROS bytes into the reusable pinned host buffer. On the same CUDA stream it records a start event, transfers the image to device memory, launches preprocessing, calls `enqueueV3`, transfers each output to pinned host memory and records the stop event. It then waits for the stream and reads the elapsed event time. Outputs are safe for CPU access only after this synchronization.

The stream is asynchronous with respect to CPU submission, but this implementation completes one frame before starting the next GPU frame. It does not claim cross-frame double buffering or zero-copy transport. The main ROS callback can receive/replace a pending frame while the worker waits for the GPU.

The output/dimension/timing accessors simply expose the stored metadata. The default destructor releases the PImpl and its resource owners.

## GPU preprocessing: `preprocess.hpp` and `preprocess.cu`

`Normalization` is a small plain-data object copied into the kernel launch. The `preprocess` wrapper computes the number of 256-thread blocks required for the destination image and launches on the caller's stream.

`resize_normalize` assigns one thread to one destination pixel. A bounds check handles the last partially filled block. Half-pixel coordinates and edge clamping locate four source pixels. For each RGB channel, it selects the right byte offset (reversing BGR), interpolates, scales, normalizes and writes to `c*H*W + y*W + x` in planar NCHW memory. All these operations are fused into one kernel, avoiding separate resized and converted device images. The host checks launch errors immediately afterward; asynchronous errors surface at stream synchronization.

## ROS integration: `inference_node.cpp`

The `Clock` alias uses a steady clock so system-clock adjustments do not distort measured durations. `milliseconds` converts its durations. Camera timestamps remain in their original ROS header and are not mixed into steady-clock measurements.

The `InferenceNode` constructor declares read-only startup parameters, validates ranges and normalization lengths, creates the engine and checks the classification output contract. Warmup uses a black 1×1 RGB image resized by the normal kernel. It occurs before subscribing, so no camera frames queue behind warmup. The constructor creates depth-one best-effort topics, logs readiness and starts the worker only after all state is ready.

`receive` is deliberately short. It records receipt time, increments the callback counter, rejects bad images with throttled warnings, and under a mutex replaces the one pending shared image pointer. Replacing a pending pointer increments `overwritten_total`. It signals the worker condition variable. It does not run inference or copy all image bytes inside the ROS executor.

`run` waits without polling until there is a pending frame or shutdown. It moves the pending pointer out while holding the lock, then releases the lock before GPU work. New callbacks can therefore store the next frame during inference. The worker calls `Engine::infer`, copies raw output views if requested, runs top-k if requested, fills timings/counters, and publishes. A fatal exception logs the error, sets a failure flag, and shuts down the ROS context. The process subsequently exits nonzero.

The destructor marks stopping, discards pending work, notifies the condition variable and joins the worker. An in-flight inference is allowed to finish before its resources are destroyed. `main` initializes ROS, spins the executor, reads the failure flag, destroys the node and shuts down. Startup exceptions also produce a nonzero exit.

## Configuration and launch files

`config/inference.yaml` is a generic example with RGB ImageNet-style normalization and raw outputs. Those values are not universal. `config/demo.yaml` selects the exact arithmetic used by the RGB fixture and publishes both raw tensors and ordered channel scores.

`inference.launch.py` declares a required engine path and optional parameter file, loads YAML first and overrides the engine path afterward. `ParameterValue(..., value_type=str)` prevents path strings from being interpreted as YAML types.

`demo.launch.py` uses `OpaqueFunction` to inspect launch arguments, validates the backend, and creates a source, inference process and dashboard. The TensorRT branch requires an engine path. The CPU branch names its backend explicitly and never tries to deserialize a TensorRT engine. The color source is animated at the requested rate.

## Engine and demo tools

`make_test_model.py` parses size/path arguments, creates an ONNX graph with `GlobalAveragePool` and `Flatten`, declares static input/output shapes, checks the graph and saves it. It fixes a conservative ONNX opset and IR version. There are no learned weights or downloaded datasets.

`build_engine.sh` validates its arguments and precision, locates `trtexec`, rejects missing model files and existing output paths, and invokes the builder with FP32 boundary formats. In FP16 mode it allows internal half-precision tactics; in FP32 mode it disables TF32. Workspace is set to 1024 MiB. Quoted Bash arrays preserve paths containing spaces. Engine creation runs once, not in the frame loop.

`publish_test_images.py` defines the ROS `Source`. Its constructor configures rate/resolution, allocates a constant RGB image and starts a timer. `tick` optionally changes RGB values sinusoidally, stamps and publishes the frame. The nonanimated pattern is used for numerical verification. The requested timer rate is not proof of an achieved publishing rate.

`cpu_reference_demo.py` contains `resize_half_pixel`, a NumPy implementation of the CUDA sampling rule. `Reference.infer` validates stride/encoding, unpacks the image, converts BGR when needed, resizes to 224×224, averages channels and divides by 255. It fills the same ROS result interface with `backend=cpu_reference`. This is only the fixture's arithmetic, not a general ONNX runtime and not an emulation of TensorRT timing/concurrency.

`demo_dashboard.py` embeds the browser page and implements `Dashboard`. The image callback samples at most 5 Hz, packs padded rows, converts BGR to RGB and makes a small JPEG. The result callback stores backend, timings, channel values and recent arrival times. A separate local HTTP server serves the page and JSON state under a lock. Its request handler is read-only, binds to localhost and has no external JavaScript dependencies. The browser polls at 5 Hz and marks results stale after two seconds. `close` stops and joins the HTTP server. Preview and results are independently sampled.

`benchmark.py` parses measurement settings and subscribes to results. Its callback starts a monotonic warmup window upon the first result, records samples only during the fixed measurement window and rejects sequence restarts. The main loop enforces a startup timeout, produces p50/p95/p99 timings, computes delivered FPS as count/window length, records counter deltas/sequence gaps, and checks backend plus strict FPS threshold. The JSON output and exit code make acceptance reproducible. It stores full messages for the window, so use compact outputs or a shorter window for large models.

`verify_fixture.py` subscribes with sensor-compatible QoS, checks the expected backend, `[1,3]` output, known RGB means and class ordering, and exits only after several correct frames. It has a timeout to avoid hanging when no pipeline is running.

## Tests

`test_core.cpp` covers tensor shape overflow/dynamic rejection, padded image rows, truncation/encoding/capacity failures, large-logit softmax, deterministic ties, unchanged raw scores and invalid normalization. These target likely correctness failures rather than just mirroring function internals.

`test_preprocess.cpp` is compiled on the GPU build. It uploads a padded BGR fixture, exercises bilinear downsampling and channel normalization and compares against an independent expected value. If no CUDA device is present, it reports skipped, which is not evidence of a GPU pass. End-to-end engine correctness is checked with the ONNX fixture verifier on the actual laptop.
