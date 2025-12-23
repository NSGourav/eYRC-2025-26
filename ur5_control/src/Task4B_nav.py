#!/usr/bin/env python3

"""
Clean DWA (Dynamic Window Approach) Planner with ROS2 Wrapper
Refactored from ebot_nav_task3b.py for easy tuning and understanding
"""

import numpy as np
from math import hypot, cos, sin, atan2, pi
import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from tf_transformations import euler_from_quaternion


# ============================================================================
# CONFIGURATION CLASS - All tunable DWA parameters in one place
# ============================================================================

class DWAConfig:
    """Centralized DWA configuration - all tunable parameters"""

    # Scoring weights (6 objectives) - tune these for different behaviors
    ALPHA = 1.8      # Heading towards goal (higher = more direct path)
    BETA = 1.2       # Clearance from obstacles (higher = safer paths)
    GAMMA = 1.8      # Forward velocity preference (higher = faster movement)
    DELTA = 2.5      # Progress towards goal (higher = more aggressive advancement)
    EPSILON = 0.4    # Rotation penalty (higher = smoother/straighter paths)
    ZETA = 0.3       # Smoothness/velocity continuity (higher = less jerky motion)

    # Time parameters
    DEL_T = 1.75     # Prediction horizon in seconds (how far ahead to simulate)
    DT = 0.05        # Simulation timestep in seconds

    # Velocity sampling - more samples = better paths but slower computation
    V_SAMPLES = 7    # Linear velocity samples
    W_SAMPLES = 13   # Angular velocity samples

    # Robot velocity limits (m/s and rad/s)
    V_MIN = 0.0
    V_MAX = 2.0
    W_MIN = -2.5
    W_MAX = 2.5

    # Robot acceleration limits (m/s^2 and rad/s^2)
    A_MIN = -1.0     # Linear deceleration
    A_MAX = 1.0      # Linear acceleration
    AL_MIN = -0.5    # Angular deceleration
    AL_MAX = 0.5     # Angular acceleration

    # Safety parameters
    bot_radius = 0.25         # Robot radius in meters
    safety_margin = 0.15      # Additional safety margin around robot
    max_clearance_norm = 3.0  # Maximum clearance for normalization
    min_traj_V = 0.3          # Minimum forward speed for candidate trajectories

    # Optimization parameters
    collision_check_interval = 2     # Check collision every N simulation steps
    early_termination_score = 5.8    # Stop search if score exceeds this


# ============================================================================
# DWA PLANNER - Pure algorithm (no ROS2 dependencies)
# ============================================================================

class DWAPlanner:
    """
    Pure DWA planning algorithm
    Can be tested independently and reused outside ROS2
    """

    def __init__(self, config):
        """
        Initialize DWA planner with configuration

        Args:
            config: DWAConfig instance with all parameters
        """
        self.config = config
        # Pre-compute squared safety distance for faster collision checks
        self.safety_distance_sq = (config.bot_radius + config.safety_margin) ** 2

    def compute_dynamic_window(self, current_vel):
        """
        Compute allowable velocity ranges based on current velocity and acceleration limits

        Args:
            current_vel: Tuple of (v, w) - current linear and angular velocities

        Returns:
            Tuple of ((v_min, v_max), (w_min, w_max))
        """
        curr_vx, curr_w = current_vel
        cfg = self.config

        # Dynamic window based on acceleration limits
        v_min = max(curr_vx + cfg.A_MIN * cfg.DEL_T, cfg.V_MIN)
        v_max = min(curr_vx + cfg.A_MAX * cfg.DEL_T, cfg.V_MAX)
        w_min = max(curr_w + cfg.AL_MIN * cfg.DEL_T, cfg.W_MIN)
        w_max = min(curr_w + cfg.AL_MAX * cfg.DEL_T, cfg.W_MAX)

        # Enforce minimum forward velocity when moving to goal
        v_min = max(v_min, cfg.min_traj_V)

        # Ensure valid ranges
        if v_min >= v_max:
            v_min, v_max = cfg.V_MIN, cfg.V_MAX
        if w_min >= w_max:
            w_min, w_max = cfg.W_MIN, cfg.W_MAX

        return (v_min, v_max), (w_min, w_max)

    def simulate_trajectory(self, pose, v, w):
        """
        Forward simulate trajectory using constant velocity model

        Args:
            pose: Tuple of (x, y, yaw) - starting pose
            v: Linear velocity
            w: Angular velocity

        Returns:
            List of trajectory points [(x, y, yaw), ...]
        """
        x, y, yaw = pose
        cfg = self.config
        trajectory = []
        steps = int(cfg.DEL_T / cfg.DT)

        for _ in range(steps):
            # Update position using constant velocity model
            x += v * cos(yaw) * cfg.DT
            y += v * sin(yaw) * cfg.DT
            yaw = (yaw + w * cfg.DT + pi) % (2 * pi) - pi  # Normalize to [-pi, pi]
            trajectory.append((x, y, yaw))

        return trajectory

    def check_collision(self, trajectory, obstacles_world):
        """
        Check if trajectory collides with obstacles

        Args:
            trajectory: List of (x, y, yaw) points
            obstacles_world: Nx2 numpy array of obstacle positions in world frame

        Returns:
            Tuple of (collision_detected: bool, min_clearance: float)
        """
        if len(obstacles_world) == 0:
            return False, float('inf')

        cfg = self.config
        min_clearance = float('inf')

        # Check collision at regular intervals for efficiency
        for step_idx, (x, y, _) in enumerate(trajectory):
            if step_idx % cfg.collision_check_interval == 0 or step_idx == len(trajectory) - 1:
                # Compute squared distances (avoid sqrt for speed)
                dists_sq = (obstacles_world[:, 0] - x)**2 + (obstacles_world[:, 1] - y)**2
                step_min_sq = float(np.min(dists_sq))

                # Check collision
                if step_min_sq < self.safety_distance_sq:
                    return True, 0.0

                # Track minimum clearance (need sqrt here)
                step_min = np.sqrt(step_min_sq)
                min_clearance = min(min_clearance, step_min)

        return False, min_clearance

    def score_trajectory(self, trajectory, goal, v, w, current_vel, current_dist, min_clearance):
        """
        Compute multi-objective score for trajectory

        Args:
            trajectory: List of (x, y, yaw) trajectory points
            goal: Tuple of (goal_x, goal_y)
            v: Linear velocity of this trajectory
            w: Angular velocity of this trajectory
            current_vel: Tuple of (current_v, current_w)
            current_dist: Current distance to goal
            min_clearance: Minimum clearance from obstacles

        Returns:
            Combined score (higher is better)
        """
        cfg = self.config
        goal_x, goal_y = goal
        curr_vx, curr_w = current_vel

        # Get final trajectory state
        xf, yf, th_f = trajectory[-1]

        # 1. Heading score: Align final heading with direction to goal
        heading_angle = atan2(goal_y - yf, goal_x - xf)
        heading_diff = self.normalize_angle(heading_angle - th_f)
        heading_score = 1.0 - abs(heading_diff) / pi

        # 2. Clearance score: Maintain distance from obstacles
        if min_clearance == float('inf'):
            clearance_norm = 1.0
        else:
            clearance_clamped = min(min_clearance, cfg.max_clearance_norm)
            clearance_norm = clearance_clamped / cfg.max_clearance_norm

        # 3. Velocity score: Prefer higher speeds
        if cfg.V_MAX - cfg.V_MIN > 1e-6:
            v_norm = (v - cfg.V_MIN) / (cfg.V_MAX - cfg.V_MIN)
            v_norm = np.clip(v_norm, 0.0, 1.0)
        else:
            v_norm = 0.0

        # 4. Progress score: Move closer to goal
        final_dist = hypot(xf - goal_x, yf - goal_y)
        raw_progress = max(0.0, current_dist - final_dist)
        denom = max(1e-3, current_dist)
        progress_norm = np.clip(raw_progress / denom, 0.0, 1.0)

        # 5. Rotation score: Penalize rotation (adaptive based on heading error)
        # If heading is poor, allow more rotation; if good, penalize rotation more
        rotation_penalty_factor = 0.3 + 0.7 * heading_score
        w_norm = 1.0 - abs(w) / cfg.W_MAX
        w_norm = np.clip(w_norm, 0.0, 1.0)

        # 6. Smoothness score: Prefer trajectories close to current velocity
        v_diff = abs(v - curr_vx) / max(cfg.V_MAX, 1e-3)
        w_diff = abs(w - curr_w) / max(cfg.W_MAX, 1e-3)
        smoothness_norm = 1.0 - 0.5 * (v_diff + w_diff)
        smoothness_norm = np.clip(smoothness_norm, 0.0, 1.0)

        # Combined score with weighted objectives
        score = (cfg.ALPHA * heading_score +
                cfg.BETA * clearance_norm +
                cfg.GAMMA * v_norm +
                cfg.DELTA * progress_norm +
                cfg.EPSILON * rotation_penalty_factor * w_norm +
                cfg.ZETA * smoothness_norm)

        return score

    def plan(self, current_pose, current_vel, goal, obstacles_world):
        """
        Main DWA planning function - finds best velocity command

        Args:
            current_pose: Tuple of (x, y, yaw)
            current_vel: Tuple of (v, w)
            goal: Tuple of (goal_x, goal_y)
            obstacles_world: Nx2 numpy array of obstacles in world frame

        Returns:
            Tuple of (best_v, best_w, best_trajectory)
            Returns (0.0, 0.0, None) if no valid trajectory found
        """
        cfg = self.config

        # Compute dynamic window
        (v_min, v_max), (w_min, w_max) = self.compute_dynamic_window(current_vel)

        # Sample velocity space
        v_samples = np.linspace(v_min, v_max, cfg.V_SAMPLES)
        w_samples = np.linspace(w_min, w_max, cfg.W_SAMPLES)

        # Calculate current distance to goal
        current_dist = hypot(current_pose[0] - goal[0], current_pose[1] - goal[1])

        # Evaluate all velocity combinations
        best_score = -float('inf')
        best_v = 0.0
        best_w = 0.0
        best_traj = None

        for v in v_samples:
            for w in w_samples:
                # Simulate trajectory
                trajectory = self.simulate_trajectory(current_pose, v, w)

                if len(trajectory) == 0:
                    continue

                # Check collision
                collision, min_clearance = self.check_collision(trajectory, obstacles_world)
                if collision:
                    continue

                # Score trajectory
                score = self.score_trajectory(
                    trajectory, goal, v, w, current_vel, current_dist, min_clearance
                )

                # Update best trajectory
                if score > best_score:
                    best_score = score
                    best_v = v
                    best_w = w
                    best_traj = trajectory

                    # Early termination if excellent trajectory found
                    if best_score >= cfg.early_termination_score:
                        return best_v, best_w, best_traj

        return best_v, best_w, best_traj

    @staticmethod
    def normalize_angle(angle):
        """Normalize angle to [-pi, pi] range"""
        while angle > pi:
            angle -= 2.0 * pi
        while angle < -pi:
            angle += 2.0 * pi
        return angle


# ============================================================================
# ROS2 NAVIGATOR NODE - Wrapper around DWA planner
# ============================================================================

class DWANavigatorNode(Node):
    """
    ROS2 wrapper for DWA planner
    Handles subscriptions, publications, and control loop
    """

    def __init__(self):
        super().__init__('dwa_navigator')

        # Initialize DWA planner
        self.config = DWAConfig()
        self.planner = DWAPlanner(self.config)

        # Robot state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.curr_vx = 0.0
        self.curr_w = 0.0
        self.obstacles = np.empty((0, 2))  # Obstacles in robot frame

        # Hardcoded waypoint array [x, y, yaw] (will be replaced by service calls later)
        self.waypoints = [
            [0.53, -1.95, 1.57],      # Waypoint 1
            [0.26, 1.1, 1.57],        # Waypoint 2
            [-1.53, 1.1, -1.57],      # Waypoint 3
            [-1.53, -5.52, -1.57],    # Waypoint 4
            [-3.56, -5.52, 1.57],     # Waypoint 5
            [-3.56, 1.1, 1.57]        # Waypoint 6
        ]
        self.w_index = 0
        self.waypoint_reached_threshold = 0.15  # meters
        self.yaw_threshold = 0.05  # radians (~3 degrees)

        # Velocity scaling parameters
        self.goal_slow_down_distance = 1.0  # Start slowing down within this distance
        self.obstacle_slow_down_distance = 0.8  # Slow down if obstacles closer than this
        self.min_velocity_scale = 0.3  # Minimum velocity scaling factor

        # Velocity smoothing parameters
        self.smoothing_ratio = 0.7
        self.prev_cmd = Twist()

        # Waypoint stability tracking
        self.stable_count = 0
        self.stable_threshold = 5  # Must be stable for this many cycles
        self.velocity_stable_threshold = 0.05  # Consider stopped if vel < this

        # Waypoint pause tracking
        self.is_pausing = False
        self.pause_start_time = None
        self.pause_duration = 2.0  # Pause for 2 seconds at each waypoint

        # LaserScan filter angles (±70° from front)
        self.min_lidar_angle = -70.0 * pi / 180.0
        self.max_lidar_angle = 70.0 * pi / 180.0

        # ROS2 subscriptions
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        # ROS2 publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Control loop timer (20Hz)
        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info('DWA Navigator initialized')
        self.get_logger().info(f'Waypoints: {len(self.waypoints)}')

    def odom_callback(self, msg):
        """Update robot pose and velocity from odometry"""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.current_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.curr_vx = msg.twist.twist.linear.x
        self.curr_w = msg.twist.twist.angular.z

    def scan_callback(self, msg):
        """Process laser scan to extract obstacles in robot frame"""
        pts = []
        angle = msg.angle_min

        for r in msg.ranges:
            # Filter by range validity and angle range
            if msg.range_min < r < msg.range_max:
                if self.min_lidar_angle <= angle <= self.max_lidar_angle:
                    x = r * cos(angle)
                    y = r * sin(angle)
                    pts.append((x, y))
            angle += msg.angle_increment

        self.obstacles = np.array(pts) if len(pts) > 0 else np.empty((0, 2))

    def transform_obstacles_to_world(self):
        """Transform obstacles from robot frame to world frame"""
        if len(self.obstacles) == 0:
            return np.empty((0, 2))

        # Rotation and translation transformation
        obs_world_x = (self.current_x +
                      self.obstacles[:, 0] * cos(self.current_yaw) -
                      self.obstacles[:, 1] * sin(self.current_yaw))
        obs_world_y = (self.current_y +
                      self.obstacles[:, 0] * sin(self.current_yaw) +
                      self.obstacles[:, 1] * cos(self.current_yaw))

        return np.column_stack((obs_world_x, obs_world_y))

    def control_loop(self):
        """Main control loop - run DWA and publish velocity commands"""

        # Check if all waypoints reached
        if self.w_index >= len(self.waypoints):
            self.stop_robot()
            self.get_logger().info('All waypoints reached!', once=True)
            return

        # Check if currently pausing at a waypoint
        if self.is_pausing:
            elapsed_time = time.time() - self.pause_start_time
            if elapsed_time >= self.pause_duration:
                # Pause complete - advance to next waypoint
                self.get_logger().info(f'Pause complete. Moving to waypoint {self.w_index + 2}/{len(self.waypoints)}')
                self.w_index += 1
                self.is_pausing = False
                self.pause_start_time = None
                self.stable_count = 0
            else:
                # Still pausing - send zero velocity
                self.stop_robot()
                self.get_logger().info(
                    f'Pausing at waypoint {self.w_index + 1}: {elapsed_time:.1f}/{self.pause_duration:.1f}s',
                    throttle_duration_sec=0.5
                )
            return

        # Get current goal waypoint
        goal_x, goal_y, goal_yaw = self.waypoints[self.w_index]

        # Check if position reached
        dist_to_goal = hypot(self.current_x - goal_x, self.current_y - goal_y)

        if dist_to_goal < self.waypoint_reached_threshold:
            # Position reached, now align yaw
            yaw_error = self.planner.normalize_angle(goal_yaw - self.current_yaw)

            if abs(yaw_error) > self.yaw_threshold:
                # Need to align yaw - rotate in place
                w_cmd = 2.0 * yaw_error  # Proportional control for yaw
                w_cmd = max(-2.5, min(2.5, w_cmd))  # Clamp to limits
                self.publish_smoothed(0.0, w_cmd)
                self.stable_count = 0  # Reset stability counter
                self.get_logger().info(
                    f'Aligning yaw at waypoint {self.w_index + 1}: '
                    f'error={yaw_error:.3f} rad', throttle_duration_sec=1.0
                )
                return
            else:
                # Both position and yaw reached - check if robot is stable/stopped
                if abs(self.curr_vx) < self.velocity_stable_threshold and \
                   abs(self.curr_w) < self.velocity_stable_threshold:
                    self.stable_count += 1
                else:
                    self.stable_count = 0

                if self.stable_count >= self.stable_threshold:
                    # Robot is stable at waypoint - start pause
                    if not self.is_pausing:
                        self.get_logger().info(
                            f'Reached waypoint {self.w_index + 1}/{len(self.waypoints)} '
                            f'at ({self.current_x:.2f}, {self.current_y:.2f}, {self.current_yaw:.2f}). '
                            f'Pausing for {self.pause_duration}s...'
                        )
                        self.is_pausing = True
                        self.pause_start_time = time.time()
                    # Send zero velocity during pause (handled in pause check above)
                    self.stop_robot()
                else:
                    # Still settling - send zero velocity
                    self.publish_smoothed(0.0, 0.0)
                return

        # Still navigating to waypoint - use DWA
        self.stable_count = 0  # Reset stability counter while moving

        # Transform obstacles to world frame
        obstacles_world = self.transform_obstacles_to_world()

        # Call DWA planner
        current_pose = (self.current_x, self.current_y, self.current_yaw)
        current_vel = (self.curr_vx, self.curr_w)
        goal = (goal_x, goal_y)

        v_cmd, w_cmd, best_traj = self.planner.plan(
            current_pose, current_vel, goal, obstacles_world
        )

        # Handle no valid trajectory case
        if best_traj is None:
            self.get_logger().warn('No valid trajectory found! Rotating in place...')
            v_cmd, w_cmd = 0.0, 0.4
        else:
            # Apply velocity scaling based on distance to goal and obstacles
            velocity_scale = 1.0

            # 1. Scale down velocity near goal
            if dist_to_goal < self.goal_slow_down_distance:
                goal_scale = max(self.min_velocity_scale,
                                dist_to_goal / self.goal_slow_down_distance)
                velocity_scale = min(velocity_scale, goal_scale)

            # 2. Scale down velocity near obstacles
            if len(obstacles_world) > 0:
                # Find minimum obstacle distance
                obs_dists = np.sqrt((obstacles_world[:, 0] - self.current_x)**2 +
                                   (obstacles_world[:, 1] - self.current_y)**2)
                min_obs_dist = float(np.min(obs_dists))

                if min_obs_dist < self.obstacle_slow_down_distance:
                    obs_scale = max(self.min_velocity_scale,
                                   min_obs_dist / self.obstacle_slow_down_distance)
                    velocity_scale = min(velocity_scale, obs_scale)

            # Apply scaling to linear velocity only (keep angular velocity for steering)
            v_cmd = v_cmd * velocity_scale

            # Log when slowing down
            if velocity_scale < 0.9:
                self.get_logger().info(
                    f'Velocity scaled to {velocity_scale:.2f} '
                    f'(dist_to_goal={dist_to_goal:.2f}m)',
                    throttle_duration_sec=2.0
                )

        # Publish smoothed velocity command
        self.publish_smoothed(v_cmd, w_cmd)

    def publish_smoothed(self, v, w):
        """Apply exponential smoothing and publish velocity command"""
        # Exponential smoothing to reduce jitter
        smoothed_v = (self.smoothing_ratio * v +
                     (1.0 - self.smoothing_ratio) * self.prev_cmd.linear.x)
        smoothed_w = (self.smoothing_ratio * w +
                     (1.0 - self.smoothing_ratio) * self.prev_cmd.angular.z)

        # Create and publish command
        cmd = Twist()
        cmd.linear.x = float(smoothed_v)
        cmd.angular.z = float(smoothed_w)
        self.cmd_pub.publish(cmd)

        # Save for next smoothing iteration
        self.prev_cmd = cmd

    def stop_robot(self):
        """Publish zero velocity to stop the robot"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)


# ============================================================================
# MAIN
# ============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = DWANavigatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
