#!/usr/bin/env python3
"""CPU implementation of the bundled RGB-mean fixture. Does NOT load TensorRT engines."""
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from ros2_tensorrt_inference.msg import InferenceResult, Tensor

def resize_half_pixel(image, height=224, width=224):
    """Same sampling coordinates as preprocess.cu, with NumPy instead of CUDA."""
    sh, sw, _ = image.shape
    x = np.clip((np.arange(width, dtype=np.float32)+0.5)*sw/width-0.5, 0, sw-1)
    y = np.clip((np.arange(height, dtype=np.float32)+0.5)*sh/height-0.5, 0, sh-1)
    x0, y0 = x.astype(np.int64), y.astype(np.int64)
    x1, y1 = np.minimum(x0+1,sw-1), np.minimum(y0+1,sh-1)
    ax, ay = (x-x0)[None,:,None], (y-y0)[:,None,None]
    a, b = image[y0[:,None],x0], image[y0[:,None],x1]
    c, d = image[y1[:,None],x0], image[y1[:,None],x1]
    return (a+(b-a)*ax)*(1-ay)+(c+(d-c)*ax)*ay

class Reference(Node):
    def __init__(self):
        super().__init__('cpu_reference_demo')
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.publisher = self.create_publisher(InferenceResult, '/inference/results', qos)
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.infer, qos)
        self.received, self.invalid, self.sequence = 0, 0, 0
        self.get_logger().warning('CPU REFERENCE DEMO: RGB mean fixture only; no TensorRT/GPU inference')
    def infer(self, image):
        begin = time.monotonic()
        self.received += 1
        needed = image.step * image.height
        if (image.encoding not in ('rgb8','bgr8') or image.width <= 0 or image.height <= 0
            or image.step < image.width*3 or needed > len(image.data) or needed > 6220800):
            self.invalid += 1
            return
        pixels = np.frombuffer(image.data, dtype=np.uint8, count=needed).reshape(image.height, image.step)
        pixels = pixels[:, :image.width*3].reshape(image.height,image.width,3).astype(np.float32)
        if image.encoding == 'bgr8':
            pixels = pixels[:,:,::-1]
        resized = resize_half_pixel(pixels)
        # The demo ONNX is GlobalAveragePool + Flatten, with scale=1/255, mean=0, std=1.
        values = (resized.mean(axis=(0,1))/255.0).astype(np.float32)
        order = np.argsort(-values, kind='stable')
        result = InferenceResult()
        result.header, result.backend = image.header, 'cpu_reference'
        self.sequence += 1
        result.sequence = self.sequence
        tensor = Tensor()
        tensor.name, tensor.shape, tensor.data = 'scores', [1,3], values.tolist()
        result.tensors = [tensor]
        result.class_ids, result.scores = order.astype(np.int32).tolist(), values[order].tolist()
        result.processing_ms = (time.monotonic()-begin)*1000
        result.receipt_to_result_ms = result.processing_ms
        result.gpu_pipeline_ms = 0.0  # Not applicable; backend field disambiguates.
        result.received_total, result.invalid_total = self.received, self.invalid
        self.publisher.publish(result)

def main():
    rclpy.init()
    node = Reference()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
if __name__ == '__main__':
    main()
