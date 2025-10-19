# Krishi Cobot (KC) theme for eYRC 2025-26

This repository contains the simulation setup for the Krishi Cobot (eYantra Robotics Competition 2025-26).

## Launch Commands

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

## Workspace Structure

- `ebot_description/` - Contains ebot robot description, launch files, and configurations
- `eyantra_warehouse/` - Warehouse simulation environment and models
- `ur_description/` - Universal Robots arm description
- `ur_simulation_gz/` - Gazebo simulation for Universal Robots arm and its controllers
- `ur5_control/` - Control packages for UR5 arm

Run nav script for task 1A:-
```
python3 ebot_nav_task!A.py
```