"""One-command laptop demo. CPU is explicit; TensorRT never silently falls back."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def setup(context):
    backend = LaunchConfiguration('backend').perform(context)
    engine_path = LaunchConfiguration('engine_path').perform(context)
    fps = float(LaunchConfiguration('fps').perform(context))
    package = 'ros2_tensorrt_inference'
    if backend == 'cpu_reference':
        inference = Node(package=package, executable='cpu_reference_demo.py', output='screen')
    elif backend == 'tensorrt':
        if not engine_path:
            raise ValueError('backend:=tensorrt requires engine_path:=/absolute/path/rgb_mean.engine')
        params = str(Path(get_package_share_directory(package)) / 'config' / 'demo.yaml')
        inference = Node(package=package, executable='inference_node', name='tensorrt_inference',
                         parameters=[params, {'engine_path': engine_path}], output='screen')
    else:
        raise ValueError('backend must be cpu_reference or tensorrt')
    return [
        Node(package=package, executable='publish_test_images.py', output='screen',
             parameters=[{'fps': fps, 'animate': True, 'width': 640, 'height': 480}]),
        inference,
        Node(package=package, executable='demo_dashboard.py', output='screen',
             parameters=[{'port': int(LaunchConfiguration('port').perform(context))}]),
    ]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('backend', default_value='cpu_reference'),
        DeclareLaunchArgument('engine_path', default_value=''),
        DeclareLaunchArgument('fps', default_value='60.0'),
        DeclareLaunchArgument('port', default_value='8765'),
        OpaqueFunction(function=setup),
    ])
