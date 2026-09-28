#!/usr/bin/env python3
"""Verify constant RGB [64,128,192], demo normalization, and bundled RGB-mean model."""
import argparse
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from ros2_tensorrt_inference.msg import InferenceResult

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['tensorrt','cpu_reference'], default='tensorrt')
    parser.add_argument('--frames', type=int, default=10)
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error('frames must be positive')
    rclpy.init()
    node = Node('verify_rgb_fixture')
    count = 0
    def callback(message):
        nonlocal count
        if message.backend != args.backend:
            raise RuntimeError(f'Wrong backend: {message.backend}')
        if len(message.tensors) != 1 or list(message.tensors[0].shape) != [1,3]:
            raise RuntimeError('Expected one [1,3] tensor; use config/demo.yaml')
        np.testing.assert_allclose(message.tensors[0].data,
            np.array([64,128,192])/255, atol=0.003, rtol=0,
            err_msg='Incorrect RGB means; stop animation and check normalization/engine')
        if list(message.class_ids) != [2,1,0]:
            raise RuntimeError('Incorrect channel ranking')
        count += 1
    subscription = node.create_subscription(InferenceResult, '/inference/results', callback,
        QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
    deadline = time.monotonic()+20
    try:
        while rclpy.ok() and count < args.frames and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if count < args.frames:
            raise RuntimeError(f'Only {count}/{args.frames} results before timeout')
        print(f'PASS: {count} correct RGB-mean results from {args.backend}')
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
if __name__ == '__main__':
    main()
