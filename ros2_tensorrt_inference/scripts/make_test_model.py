#!/usr/bin/env python3
"""Generate a tiny deterministic ONNX fixture, not a trained classifier. Requires onnx."""
import argparse
from pathlib import Path
import onnx
from onnx import TensorProto, helper

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)
    args = parser.parse_args()
    if args.height <= 0 or args.width <= 0:
        parser.error('height and width must be positive')
    # Spatial mean of each normalized RGB channel; output = [1,3].
    graph = helper.make_graph([
        helper.make_node('GlobalAveragePool', ['images'], ['pooled']),
        helper.make_node('Flatten', ['pooled'], ['scores'], axis=1)],
        'rgb_mean_fixture',
        [helper.make_tensor_value_info('images', TensorProto.FLOAT, [1,3,args.height,args.width])],
        [helper.make_tensor_value_info('scores', TensorProto.FLOAT, [1,3])])
    model = helper.make_model(graph, producer_name='ros2_tensorrt_inference',
                              opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print(args.output)
if __name__ == '__main__':
    main()
