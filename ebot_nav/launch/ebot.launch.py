#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    """    
    This launches:
    - ebot_nav.py: Dynamic Window Approach (DWA) navigation controller
    - shape_detector.py: Hough line-based shape detection for plant health monitoring
    - control.py: High-level control and coordination (if needed)
    """
    
    # Navigation node (DWA controller)
    nav_node = Node(
        package='ebot_nav',
        executable='ebot_nav.py',
        name='ebot_nav',
        output='screen',
        emulate_tty=True,
    )
    
    # Shape detector node (Hough line detection)
    shape_detector_node = Node(
        package='ebot_nav',
        executable='shape_detector.py',
        name='hough_line_detector',
        output='screen',
        emulate_tty=True,
    )
    
    # Log info
    log_info = LogInfo( msg='Launching eBot Navigation System (Team 5076)...')
    
    return LaunchDescription([
        log_info,
        nav_node,
        shape_detector_node,
    ])