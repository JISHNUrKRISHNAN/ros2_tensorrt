# Original camera timestamp/frame, preserved for downstream association.
std_msgs/Header header
string backend
uint64 sequence
# Raw tensors are populated in raw/both modes.
Tensor[] tensors
# Classification indices/scores are populated in classification/both modes.
# Scores are softmax probabilities if apply_softmax=true; otherwise model values.
int32[] class_ids
float32[] scores
float64 queue_ms
# Includes host image copy, GPU pipeline and CPU decode; excludes publish/serialization.
float64 processing_ms
# CUDA event interval: H2D + preprocess + inference + D2H.
float64 gpu_pipeline_ms
# Since callback receipt, through decoding; excludes camera/DDS ingress and publishing.
float64 receipt_to_result_ms
uint64 received_total
uint64 overwritten_total
uint64 invalid_total
