# Krishi Cobot (KC) theme for eYRC 2025-26

This repository contains the simulation setup for the Krishi Cobot (eYantra Robotics Competition 2025-26).

## Project Overview

Krishi Cobot (KC) is a ROS2-based autonomous agricultural automation system developed as part of the e-Yantra Robotics Competition (eYRC 2025–26). The project focuses on building a coordinated mobile manipulation pipeline for a smart vertical farm environment.

The system integrates an autonomous mobile robot (eBot) and a UR5 robotic arm to perform crop monitoring, plant health inspection, fertilizer coordination, and defective fruit removal. The mobile robot navigates greenhouse lanes using LiDAR-based localization and shape detection, while the UR5 arm performs vision-based object pose estimation and precise pick-and-place manipulation.

This project combines perception, navigation, manipulation, and inter-robot coordination into a unified ROS2 architecture. The perception pipeline uses ArUco marker detection and depth-based 3D localization, while the control layer integrates waypoint navigation, servo-based arm motion, and state-driven task execution.

## System Architecture

The Krishi Cobot system is built on a modular ROS2 architecture integrating perception, navigation, manipulation, and coordination layers.
The system consists of two primary robotic subsystems:

### Mobile Navigation Subsystem (eBot)
- LiDAR-based autonomous navigation
- Waypoint tracking inside greenhouse lanes
- Real-time shape detection during motion
- Publishes plant condition and task triggers

### Vision & Manipulation Subsystem (UR5)
- ArUco marker detection using OpenCV
- Depth-assisted 3D pose estimation of ArUco markers and detected fruits
- TF2-based frame transformations (camera → base → world)
- Servo-based end-effector control
- Pick-and-place execution for defective fruit removal

### Control & Coordination Layer
- State-driven task sequencing
- ROS2 topic-based communication between eBot and UR5
- Trigger-based collaboration logic
- Execution Flow: Navigate to Waypoint → Detect Target (LiDAR / Vision) → Compute 3D Pose (TF Transform) → Execute Manipulation → Update System State

The architecture enables an end-to-end autonomous workflow where the mobile robot navigates and detects plant conditions, then coordinates with the UR5 arm for corrective actions, completing a full agricultural automation cycle.

## Implementation Phases

### Phase 1– Core Foundations
- `1A` - LiDAR-Based Waypoint Navigation
- `1B` - Vision-Based Pose Estimation (ArUco + Fruit Detection)
- `1C` - Waypoint-Based Servo Control for Robotic Arm Manipulation

### Phase 2– Autonomous Integration
- `2A` - Autonomous Navigation with Shape Detection
- `2B` - Pick and Place Operations
- `3B` - eBot-UR5 Arm Collaboration
- `4C` – Full Simulation Run

### Phase 3– Hardware Deployment
- `3A` - Remote Hardware Task - Object Pose Estimation
- `4A` - Remote Hardware: UR5 Arm Pick and Place
- `4B` - Remote Hardware: Autonomous Navigation with Shape Detection
- `5` - Hardware Mini-Theme Run
- `6` - Full Hardware System Run

## Workspace Structure

- `HW_5076_eBot_nav/` - Hardware AMR files
- `HW_5076_ur5_control/` - Hardware ARM files
- `SW_5076_eBot_nav/` - Simulation AMR files
- `SW_5076_ur5_control/` - Simulation ARM files
- `eBot_description/` - Contains eBot robot description, launch files, and configurations
- `eyantra_warehouse/` - Warehouse simulation environment and models
- `ur_description/` - Universal Robots arm description
- `ur_simulation_gz/` - Gazebo simulation for Universal Robots arm and its controllers
- `ur5_control/` - Control packages for UR5 arm
- `linkattacher_msgs/` - Custom messages for link attacher

## Installation & Setup

### Prerequisites
- Ubuntu 22.04 (Jammy Jellyfish) 
- ROS 2 Humble
- Gazebo Fortress 

### Clone and Build
```bash
cd ~/eyrc/src
git clone https://github.com/NSGourav/eYRC-2025-26-krishi-cobot.git
mv eYRC-2025-26-krishi-cobot/* .
rm -rf eYRC-2025-26-krishi-cobot
```
```bash
cd ~/eyrc
colcon build
source install/setup.bash
```


## Launch Commands

To launch Gazebo World for Task 4C:
```bash
ros2 launch eyantra_warehouse task4c.launch.py
```
Launch Navigation Module
```bash
ros2 launch ebot_nav eBot.launch.py
```
Launch Manipulation Module
```bash
ros2 launch ur5_control arm.launch.py
```
Run Control file
```bash
ros2 run ebot_nav main.py
```

## ROS Bag for Testing

[Download ROS Bag](https://drive.google.com/drive/folders/1HXDsmIbGBEiUMtXQmhGnGeMhSDm0VA2i?usp=sharing)
