"""Load a parameter file and override the required engine path from the CLI."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    default = str(Path(get_package_share_directory('ros2_tensorrt_inference')) / 'config' / 'inference.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('engine_path', description='Absolute path to a trusted TensorRT 10 engine'),
        DeclareLaunchArgument('params_file', default_value=default),
        Node(package='ros2_tensorrt_inference', executable='inference_node',
             name='tensorrt_inference', output='screen',
             parameters=[LaunchConfiguration('params_file'),
                         {'engine_path': ParameterValue(LaunchConfiguration('engine_path'), value_type=str)}]),
    ])
