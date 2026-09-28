#!/usr/bin/env python3
"""Synthetic transport/load source. Does not prove model accuracy or camera performance."""
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

class Source(Node):
    def __init__(self):
        super().__init__('test_image_source')
        self.declare_parameter('fps', 60.0)
        self.declare_parameter('animate', False)
        self.frame_number = 0
        self.animate = self.get_parameter('animate').value
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('topic', '/camera/image_raw')
        fps = float(self.get_parameter('fps').value)
        width = int(self.get_parameter('width').value)
        height = int(self.get_parameter('height').value)
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            raise ValueError('fps, width, height must be positive and finite')
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.BEST_EFFORT)
        self.publisher = self.create_publisher(Image, self.get_parameter('topic').value, qos)
        self.image = Image()
        self.image.height, self.image.width = height, width
        self.image.encoding, self.image.step = 'rgb8', width * 3
        self.image.header.frame_id = 'synthetic_camera'
        # Known constant RGB value for end-to-end numerical validation.
        self.image.data = bytes([64, 128, 192]) * (width * height)
        self.timer = self.create_timer(1.0 / fps, self.tick)
    def tick(self):
        if self.animate:
            phase = self.frame_number * 0.025
            rgb = [int(127.5 * (1 + math.sin(phase + offset))) for offset in (0, 2.094, 4.189)]
            self.image.data = bytes(rgb) * (self.image.width * self.image.height)
            self.frame_number += 1
        self.image.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.image)

def main():
    rclpy.init()
    node = Source()
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
