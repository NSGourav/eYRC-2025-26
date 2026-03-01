#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import LogInfo

def generate_launch_description():
    """    
    This launches:
    - manipulation.py: Arm controller for pick and place operations
    - perception.py: Combined perception for ArUco markers and bad fruit detection
    """
    
    # Manipulation node
    manipulation_node = Node(
        package='ur5_control',
        executable='joint_manipulation.py',
        name='arm_controller_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'use_sim_time': False
        }]
    )
    
    # Perception node
    perception_node = Node(
        package='ur5_control',
        executable='perception.py',
        name='combined_perception',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'use_sim_time': False
        }]
    )
    
    # Log info
    log_info = LogInfo( msg='Launching UR5 Control - Manipulation and Perception nodes ...')
    
    return LaunchDescription([
        log_info,
        manipulation_node,
        perception_node
    ])