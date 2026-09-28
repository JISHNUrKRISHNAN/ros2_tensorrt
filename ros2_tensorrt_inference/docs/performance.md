# Performance target and measurement

## What >30 FPS means

The default benchmark requires **more than 30 successful result messages per second delivered to its subscriber** for a 30-second window after warmup. At 30 FPS the nominal frame budget is 33.33 ms; use a source above 30 FPS for a strict >30 test. Throughput and latency are different: a queue can hide slow processing temporarily, and discarded frames can keep latency low while reducing completeness.

This package has not been GPU-benchmarked in the delivery environment. No number in the documentation is a measured TensorRT throughput claim. A tiny RGB-mean model is suitable to demonstrate mechanics; production model performance must be measured separately.

## Optimizations already in the code

- C++ inference worker independent of the ROS executor.
- One pending frame, with replacement, to avoid an ever-growing application queue.
- Best-effort depth-one ROS image/results topics for freshness-oriented streaming.
- Reused device input/output/image buffers and pinned host staging/output buffers.
- One fused CUDA resize/color/layout/normalization kernel.
- Named tensor addresses configured once; no engine parsing/building per frame.
- One nonblocking CUDA stream ordering H2D, preprocessing, inference and D2H.
- Warmup before accepting camera frames.
- Internal FP16 tactics when selected during engine building, with validated FP32 boundaries.

There is still a CPU copy from ROS image memory into pinned memory, a transfer to the GPU, output transfer back, result allocation, and DDS serialization. This is not a zero-copy or allocation-free pipeline. There is no overlap of successive frames' GPU work.

## Repeatable acceptance run

1. Record GPU model, driver, CUDA, TensorRT, ROS distribution, engine checksum, input camera/model dimensions, source FPS, precision and output mode.
2. Verify numerical correctness with the fixture or reference model before judging speed.
3. Warm up the GPU and use a consistent laptop power mode with AC power connected. Watch thermal throttling and competing GPU applications.
4. Run the source above 30 FPS, preferably 60 FPS. Measure achieved source rate separately if throughput is unexpectedly low.
5. Run `benchmark.py --require-backend tensorrt --seconds 30 --target-fps 30 --json benchmark_gpu.json`.
6. Repeat without the dashboard if it materially affects CPU/ROS traffic. Report whether visualization was enabled.
7. Inspect delivered FPS, p95/p99 processing and queue time, overwritten count and result sequence gaps together.

The script counts a full fixed wall-clock window. It does not overestimate capacity by calculating `1000 / inference_ms`. It does not measure camera-to-display latency. For that, correlate source timestamps using synchronized clocks and include transport/display timing.

## Isolate a bottleneck

| Observation | Likely next check |
|---|---|
| Small GPU time, low output rate | Source timer/camera rate, image serialization, CPU decode, publication, benchmark subscriber load |
| Large GPU pipeline time | Profile H2D, resize, engine execution and D2H separately with NVIDIA tooling |
| Rising replaced-frame count | Worker slower than arrival rate; expected overload policy, not necessarily a defect |
| High queue time | Worker processing time, OS scheduling, contention |
| Result sequence gaps | Subscriber/DDS loss, transport pressure; counters do not expose all network loss |
| Large raw model outputs | Compact model-specific postprocessing before publication |
| FP16 changes predictions | Check numerical/task accuracy; use FP32 or a validated precision strategy |

Use `trtexec --loadEngine=models/rgb_mean.engine --warmUp=1000 --duration=30` for a separate engine benchmark (check flags with the installed SDK's help). This excludes the ROS path and the package's real camera preprocessing. NVIDIA documents the distinction and benchmarking options in its [performance guidance](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html).

## Further work if the real model misses 30 FPS

Profile first. Potential improvements include a smaller model/resolution, validated FP16 or INT8 quantization, compact GPU postprocessing, CUDA graphs where compatible, or carefully implemented per-frame double buffering. Multiple in-flight contexts need separate buffers and safe ownership. For ROS zero-copy GPU transport, investigate a suitable GPU transport stack for the deployment; merely enabling intra-process communication here would not remove host-device copies.

Do not lower numerical standards or report frame dropping as improved inference capacity. Publish the achieved source rate, result rate, latency percentiles and replacement counts together.
