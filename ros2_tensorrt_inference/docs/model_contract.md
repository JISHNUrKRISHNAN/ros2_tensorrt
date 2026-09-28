# Model contract, parameters and messages

## Preprocessing and layout

Camera bytes are interleaved HWC RGB or BGR. `step` may include row padding. The callback verifies nonzero dimensions, at least `width*3` bytes per row, at least `step*height` bytes in the message, and the configured capacity. Eight-bit channels do not require byte swapping. Invalid frames are counted and skipped.

The CUDA kernel maps each output pixel with half-pixel coordinates:

```text
sx = clamp((x + 0.5) * source_width / model_width - 0.5, 0, source_width - 1)
sy = clamp((y + 0.5) * source_height / model_height - 0.5, 0, source_height - 1)
value = bilinear_interpolation(source, sx, sy)
output[c,y,x] = (value_RGB[c] * scale - mean[c]) / stddev[c]
```

It stretches the complete image to the model input size. It does not preserve aspect ratio, crop, letterbox, or perform antialias filtering. Training/export preprocessing must match; matching tensor dimensions alone does not establish correctness.

Input shape is read from the engine, never silently overridden by YAML. Static ONNX shapes are required. Creating a fixed optimization profile for a dynamic ONNX graph still leaves dynamic engine dimensions and is rejected. Export a genuinely static graph or extend the runtime to set and validate concrete shapes.

## Parameters (startup only)

| Parameter | Generic default | Meaning |
|---|---|---|
| `engine_path` | empty (required) | Serialized engine built for the target GPU/software stack |
| `device_id` | 0 | CUDA GPU index |
| `image_topic` | `/camera/image_raw` | Raw image subscription |
| `result_topic` | `/inference/results` | Result publication |
| `max_image_bytes` | 6220800 | Pinned/device raw-image capacity, includes row padding |
| `warmup_iterations` | 20 | Dummy inferences before subscribing |
| `scale` | 1/255 | Pixel scale before mean/std normalization |
| `mean` | [0.485,0.456,0.406] | RGB means, three finite values |
| `stddev` | [0.229,0.224,0.225] | RGB standard deviations, positive finite values |
| `output_mode` | raw | `raw`, `classification`, or `both` |
| `top_k` | 5 | Positive count, capped to number of classes |
| `apply_softmax` | true | Convert classification logits to probabilities |

Parameters are read-only after startup so runtime state cannot diverge from declared values. Restart to change them. `config/demo.yaml` uses zero mean, unit stddev, `both`, top-3 and no softmax. Raw outputs are never softmax-transformed. Use `apply_softmax=false` if the model already returns probabilities or if the outputs are independent channel intensities. Multilabel logits require sigmoid and a different decoder.

Launch `inference.launch.py` accepts `engine_path` and `params_file`. It passes the engine path last, overriding the parameter file. Set other values in a copy of the YAML. `demo.launch.py` accepts `backend`, `engine_path`, `fps`, and `port`. Its TensorRT branch assumes the bundled RGB fixture.

## Result messages

`Tensor.msg` stores the tensor name, original dimensions, and flattened row-major FP32 data. No output is assumed to be a detector or classifier in raw mode.

`InferenceResult.msg` includes:

- Original image header: camera timestamp and frame ID, without rewriting it to inference completion time.
- `backend`: `tensorrt` or `cpu_reference`; tools can enforce the expected path.
- `sequence`: increments once per successfully produced result; gaps at a subscriber can indicate lost deliveries.
- `tensors`: raw model values in raw/both modes.
- `class_ids`, `scores`: top-k zero-based indices in classification/both modes. They are not human-readable labels. Equal scores are ordered by class index.
- `queue_ms`: callback receipt to worker start, measured with a monotonic clock.
- `processing_ms`: worker start through output decode/copy to the result object; excludes ROS publication and serialization.
- `gpu_pipeline_ms`: CUDA event interval covering H2D, preprocessing, inference and D2H. It is **not** pure neural-network execution time.
- `receipt_to_result_ms`: queue plus processing; excludes camera exposure, transport before callback, and output publication.
- `received_total`, `overwritten_total`, `invalid_total`: lifetime counters for callbacks, pending-frame replacements, and invalid image rejects. They do not count all DDS/camera losses. Snapshots are atomic but not a transactional multi-counter snapshot.

CPU reference timing has no worker queue and reports GPU time as zero/not applicable. Counters and timing semantics specific to the GPU worker must not be inferred from CPU rehearsal results.

## Substituting a trained model

1. Export static batch-one RGB NCHW ONNX with FP32 input/output tensors.
2. Determine exact resize/crop/letterbox and normalization from the training pipeline. Modify CUDA preprocessing if needed rather than approximating it with unrelated means.
3. Build the engine on the target laptop. Use `raw` mode initially.
4. Compare reference-framework and TensorRT outputs on representative images, checking numerical tolerance and task-level accuracy.
5. Select classification mode only for `[K]`/`[1,K]` outputs. Map indices to the model's label file in your downstream application.
6. For detection, implement model-specific decoding, confidence filtering, NMS and coordinate mapping, then publish an appropriate detection message. None of these are supplied by raw tensor publication.
7. Benchmark after accuracy is verified. Large raw outputs can dominate D2H transfer and ROS serialization; add task-specific compact postprocessing where needed.

The runtime registers NVIDIA's standard TensorRT plugin library. Engines requiring custom plugin libraries need explicit registration/loading before deserialization; there is no automatic custom-plugin loader here.
