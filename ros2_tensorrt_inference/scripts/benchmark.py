#!/usr/bin/env python3
"""Measure delivered results, not reciprocal inference time. Exit 2 on unmet target."""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from ros2_tensorrt_inference.msg import InferenceResult

FIELDS = ('queue_ms', 'processing_ms', 'gpu_pipeline_ms', 'receipt_to_result_ms')
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-backend', choices=['tensorrt', 'cpu_reference'], default='tensorrt')
    parser.add_argument('--topic', default='/inference/results')
    parser.add_argument('--seconds', type=float, default=30.0)
    parser.add_argument('--warmup-seconds', type=float, default=5.0)
    parser.add_argument('--startup-timeout', type=float, default=30.0)
    parser.add_argument('--target-fps', type=float, default=30.0)
    parser.add_argument('--json', type=Path)
    args, ros_args = parser.parse_known_args()
    if (not all(math.isfinite(x) for x in (args.seconds,args.warmup_seconds,args.startup_timeout,args.target_fps))
        or args.seconds <= 0 or args.startup_timeout <= 0 or args.target_fps <= 0 or args.warmup_seconds < 0):
        parser.error('Durations and target must be finite/positive (warmup may be zero)')
    rclpy.init(args=ros_args)
    node = Node('inference_benchmark')
    samples = []
    first_seen = None
    measurement_start = None
    last_sequence = None
    def callback(msg):
        nonlocal first_seen, measurement_start, last_sequence
        now = time.monotonic()
        if first_seen is None:
            first_seen = now
            measurement_start = now + args.warmup_seconds
        if measurement_start <= now < measurement_start + args.seconds:
            if last_sequence is not None and msg.sequence <= last_sequence:
                raise RuntimeError('Sequence restarted/reordered; use a single inference publisher')
            last_sequence = msg.sequence
            samples.append((now, msg))
    subscription = node.create_subscription(InferenceResult, args.topic, callback,
        QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT))
    started = time.monotonic()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            now = time.monotonic()
            if first_seen is None and now - started >= args.startup_timeout:
                raise RuntimeError('No results before startup timeout; check node/topic/QoS')
            if measurement_start is not None and now >= measurement_start + args.seconds:
                break
        report = {'topic': args.topic, 'window_seconds': args.seconds,
                  'results_received': len(samples), 'delivered_fps': len(samples) / args.seconds,
                  'target_fps_strictly_greater_than': args.target_fps,
                  'required_backend': args.require_backend,
                  'observed_backends': sorted({msg.backend for _, msg in samples}),
                  'measurement_completed': bool(measurement_start is not None and
                     time.monotonic() >= measurement_start + args.seconds)}
        for field in FIELDS:
            values = [getattr(message, field) for _, message in samples]
            report[field] = {key: float(np.percentile(values, percentile)) if values else None
                             for key, percentile in [('p50', 50), ('p95', 95), ('p99', 99)]}
        if samples:
            a, b = samples[0][1], samples[-1][1]
            report['counters_between_first_and_last_sample'] = {
                field: getattr(b, field) - getattr(a, field)
                for field in ('received_total', 'overwritten_total', 'invalid_total')}
            report['result_sequence_gaps'] = b.sequence - a.sequence + 1 - len(samples)
        report['passed'] = (report['measurement_completed'] and report['delivered_fps'] > args.target_fps
                            and report['observed_backends'] == [args.require_backend])
        text = json.dumps(report, indent=2, allow_nan=False)
        print(text)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(text + '\n')
        return 0 if report['passed'] else 2
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
if __name__ == '__main__':
    raise SystemExit(main())
