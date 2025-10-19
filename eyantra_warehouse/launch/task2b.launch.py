#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' 
*****************************************************************************************
*  Filename:       task2b.launch.py
*  Description:    Launch Ignition Gazebo Fortress world
*  Modified by:    Joel Devasia, Sahil
*  Author:         e-Yantra Team
*****************************************************************************************
'''

import os
from os.path import join
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    AppendEnvironmentVariable,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

def generate_launch_description():
    eyantra_warehouse = get_package_share_directory("eyantra_warehouse")
    ebot_description_pkg = get_package_share_directory("ebot_description")

    world_file = LaunchConfiguration(
        "world_file", 
        default=join(eyantra_warehouse, "worlds", "eyantra_warehouse_task2.world")
    )

    gz_sim_share = get_package_share_directory("ros_gz_sim")
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            join(gz_sim_share, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": PythonExpression(["'", world_file, " -r -v 4'"])
        }.items(),
    )

    spawn_ebot_model = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name", "ebot_model",
            "-x", "0.477611",
            "-y", "-1.9581",
            "-z", "0.0493",
            "-Y", "1.5707",
            "-file", join(eyantra_warehouse, "models", "ebot_model", "ebot_description.xacro.urdf.sdf"),
        ],
        output="screen",
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ebot_bridge',
        parameters=[{
            'config_file': os.path.join(eyantra_warehouse, 'config', 'bridge.yaml'),
            'qos_overrides./tf_static.publisher.durability': 'transient_local',
            'use_sim_time': True
        }],
        output='screen'
    )

    armed0 = TimerAction(
        period=2.0,
        actions=[
            Node(
                package="eyantra_warehouse",
                executable="armed0.py",
                name="_armed0",
                output="screen",
                parameters=[{"use_sim_time": True}],
            )
        ]
    )

    return LaunchDescription([
        # Set resource paths for Gazebo
        AppendEnvironmentVariable(
            name="IGN_GAZEBO_RESOURCE_PATH",
            value=join(eyantra_warehouse, "worlds")
        ),
        AppendEnvironmentVariable(
            name="IGN_GAZEBO_RESOURCE_PATH",
            value=join(eyantra_warehouse, "models")
        ),
        AppendEnvironmentVariable(
            name="IGN_GAZEBO_RESOURCE_PATH",
            value=join(ebot_description_pkg, "models")
        ),
        # Declare launch arguments
        DeclareLaunchArgument("world_file", default_value=world_file),
        gz_sim,
        bridge,
        armed0,
        spawn_ebot_model
    ])
