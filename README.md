# Krishi Cobot (KC) theme for eYRC 2025-26

This repository contains the simulation setup for the Krishi Cobot (eYantra Robotics Competition 2025-26).

## Launch Commands

To launch Gazebo World for Task 4C:
```bash
ros2 launch eyantra_warehouse task4c.launch.py
```

```bash
eyantra-autoeval evaluate --year 2025 --theme KC --task 4C
```

```bash
ros2 run ebot_nav shape_detector_task4c.py
```

```bash
ros2 run ebot_nav ebot_nav_task4c.py
```

```bash
ros2 run ebot_nav main.py
```

```bash
ros2 run ur5_control task4c_manipulation.py
```

```bash
ros2 run ur5_control task4c_perception.py
```


## Workspace Structure

- `ebot_description/` - Contains ebot robot description, launch files, and configurations
- `eyantra_warehouse/` - Warehouse simulation environment and models
- `ur_description/` - Universal Robots arm description
- `ur_simulation_gz/` - Gazebo simulation for Universal Robots arm and its controllers
- `ur5_control/` - Control packages for UR5 arm
- `linkattacher_msgs/` - Custom messages for link attacher