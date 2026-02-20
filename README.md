# Krishi Cobot (KC) theme for eYRC 2025-26

This repository contains the simulation setup for the Krishi Cobot (eYantra Robotics Competition 2025-26).

## Tasks

- `Task 1A` - LiDAR-Based Waypoint Navigation
- `Task 1B` - Object Pose Estimation - Aurco Marker + Bad Fruit
- `Task 1C` - Waypoint-Based Servo Control for Robotic Arm Manipulation
- `Task 2A` - Autonomous Navigation with Shape Detection
- `Task 2B` - Pick and Place Operations
- `Task 3A` - Remote Hardware Task - Object Pose Estimation
- `Task 3B` - Ebot-UR5 Arm Collaboration
- `Task 4A` - Remote Hardware: UR5 Arm Pick and Place
- `Task 4B` - Remote Hardware: Autonomous Navigation with Shape Detection
- `Task 4C` – Full Simulation Run
- `Task 5` - Hardware Mini-Theme Run

## Launch Commands

To launch Gazebo World for Task 1:
```bash
ros2 launch eyantra_warehouse task1.launch.py
```

To launch Gazebo World for Task 2A:
```bash
ros2 launch eyantra_warehouse task2.launch.py
```

To launch Gazebo World for Task 2B:
```bash
ros2 launch eyantra_warehouse task2b.launch.py
```

To spawn eBot in the Gazebo world:
```bash
ros2 launch ebot_description spawn_ebot.launch.py
```

To spawn Arm & Camera inside Gazebo World:
```bash
ros2 launch ur_simulation_gz spawn_arm.launch.py
```

To launch Gazebo World for Task 3B:
```bash
ros2 launch eyantra_warehouse task3b.launch.py
```

To launch Gazebo World for Task 4C:
```bash
ros2 launch eyantra_warehouse task4c.launch.py
```

## Workspace Structure

- `ebot_description/` - Contains ebot robot description, launch files, and configurations
- `eyantra_warehouse/` - Warehouse simulation environment and models
- `ur_description/` - Universal Robots arm description
- `ur_simulation_gz/` - Gazebo simulation for Universal Robots arm and its controllers
- `ur5_control/` - Control packages for UR5 arm
- `linkattacher_msgs/` - Custom messages for link attacher
