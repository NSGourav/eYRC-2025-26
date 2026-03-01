#!/usr/bin/env python3
import time
import numpy as np
from math import hypot, cos, sin, atan2, pi
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, ColorRGBA, Float32,Bool
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker, MarkerArray
from tf_transformations import euler_from_quaternion
from std_srvs.srv import Trigger
from rclpy.callback_groups import ReentrantCallbackGroup,MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# DWA PARAMS --> sampling + forward rollout + scoring loop
ALPHA = 2.4                 # for more heading towards goal (increased)
BETA = 1.2                  # for clearance (increased to avoid tight spaces)
GAMMA = 1.8                 # for faster speeds (increased)
DELTA = 2.5                 # for reducing distance to current waypoint (significantly increased)
EPSILON = 0.18              # for penalizing rotation (reduced from 0.8 to allow more rotation)
ZETA = 0.3                  # for smoothness (prefer velocities close to current)

DEL_T = 1.75                # Reduced from 2.0 for faster computation
DT = 0.05
V_SAMPLES = 13              # Reduced from 9 for speed
W_SAMPLES = 13              # Reduced from 16 for speed
COLLISION_CHECK_INTERVAL = 2  # Check collision every N steps for speed
EARLY_TERMINATION_SCORE = 5.8  # If we find trajectory with score > this, accept it (updated for new terms)

class ebotNav3B(Node):
    def __init__(self):
        super().__init__('ebot_nav')

        # Declare ROS2 parameters
        self.sub_cb=ReentrantCallbackGroup()
        self.timer_cb=MutuallyExclusiveCallbackGroup()

        self.create_subscription(Odometry,"/odom",self.odom_callback,10)
        self.create_subscription(LaserScan,"/scan",self.scan_callback,10)
        self.create_subscription(String,"/set_immediate_goal",self.nav_callback,10,callback_group=self.sub_cb)

         # Publisher for cmd_vel
        self.cmd_pub=self.create_publisher(Twist,"/cmd_vel",10)
        self.nav_pub=self.create_publisher(Bool,"/immediate_goal_status",10)

        # Timer for control loop
        self.control_dt = DT
        self.timer_ = self.create_timer(self.control_dt, self.control_loop,callback_group=self.timer_cb)

        self.enable_viz = True
        self.flag_pose=False
        self.flag_ori=False
        if self.enable_viz:
            self.traj_pub = self.create_publisher(MarkerArray, "/dwa_trajectories", 10)
            self.best_traj_pub = self.create_publisher(Marker, "/dwa_best_trajectory", 10)
            self.goal_pub = self.create_publisher(Marker, "/dwa_goal", 10)
            self.waypoints_pub = self.create_publisher(MarkerArray, "/dwa_waypoints", 10)
            self.get_logger().info("Visualization ENABLED - RViz markers will be published")
        else:
            self.get_logger().info("Visualization DISABLED")

        # Robot initial state
        self.current_x   = 0.0
        self.current_y   = 0.0
        self.current_yaw = 1.57  # Facing forwards

        self.waypoints = None          # dock

        self.curr_vx = 0.0; self.curr_w = 0.0
        self.obstacles = np.empty((0,2))

        kp, ki, kd = (0.8, 0.0, 0.03)       # Path PID values
        self.lateral_pid = PID(kp, ki, kd, out_min=-2.5, out_max=2.5)
        ykp, yki, ykd = (20, 0.0, 0.02)     # Yaw PID values
        self.yaw_pid = PID(ykp, yki, ykd, out_min=-2.5, out_max=2.5)  # Updated to match W_MAX

        self.target_cycle_count = 0

        self.prev_cmd = Twist()

        self.bot_radius = 0.25; self.safety_margin = 0.15
        self.safety_distance = self.bot_radius + self.safety_margin
        self.safety_distance_sq = self.safety_distance ** 2  # Squared for faster collision checks

        self.min_lidar_angle= -70.0 * pi / 180.0    # usable range of LIDAR on ebot for navigation(full range=[-135,+135])
        self.max_lidar_angle= 70.0 * pi / 180.0

        self.V_MIN, self.V_MAX = 0.0, 0.3
        self.W_MIN, self.W_MAX = -1.0, 1.0        # Increased from ±1.5 to allow faster rotation
        self.A_MIN, self.A_MAX = -1.0, 1.0
        self.AL_MIN, self.AL_MAX = -1.0, 1.0      # Increased from ±0.2 for quicker rotation changes

        self.max_clearance_norm = 3.0
        self.lookahead_index = 3
        self.lookahead_index_close = 0
        self.smoothing_ratio = 0.7

        self.turn_speed_redn = 0.7
        self.target_thresh = 0.18
        self.yaw_thresh = 0.05

        self.dist_tolerance = 0.15    # if more than this, moving to target
        self.min_traj_V = 0.3  # minimal forward speed for candidate trajectories

        self.last_all_trajs = []
        self.last_best_traj = None
        self.start_time = time.time()
        self.get_logger().info("ebot_nav_task2A node started")

        # Publish waypoints visualization once (if enabled)
        if self.enable_viz:
            self.visualize_all_waypoints()

    def odom_callback(self,msg:Odometry):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_yaw = yaw
        self.curr_vx = msg.twist.twist.linear.x
        self.curr_w = msg.twist.twist.angular.z


    def nav_callback(self, msg: String):
        parts = msg.data.split(',')
        if len(parts) != 3:
            response_msg = String()
            response_msg.data = "ERROR: Invalid format. Use 'x,y,yaw'"
            self.get_logger().error("Invalid intermediate goal format. Expected 'x,y'")
            return

        x = float(parts[0].strip())
        y = float(parts[1].strip())
        yaw = float(parts[2].strip())
        self.waypoints = [x, y, yaw]
        self.flag_pose = False
        self.flag_ori = False
        self.get_logger().info(f"Goal pose: ({self.waypoints[0]:.2f},{self.waypoints[1]:.2f},{self.waypoints[2]:.2f})")
        time.sleep(1)


    def scan_callback(self,msg:LaserScan):
        pts= []
        angle = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                if self.min_lidar_angle <= angle <= self.max_lidar_angle:
                    x = r * cos(angle); y = r * sin(angle); pts.append((x,y))
            angle += msg.angle_increment
        if len(pts) > 0:
            self.obstacles = np.array(pts)
        else:
            self.obstacles = np.empty((0,2))

    def control_loop(self):

        if self.waypoints is None:
            self.get_logger().info("No waypoints set. Robot is idle.")
            self.stop_robot()
            return

        if (self.flag_pose==True) and (self.flag_ori==True):
            self.stop_robot()
            return
        # else:


        x_goal, y_goal, yaw_goal = self.waypoints
        dist_to_goal = hypot(self.current_x - x_goal, self.current_y - y_goal)
        goal_reached = dist_to_goal < self.target_thresh

        if goal_reached and self.flag_pose==False:
            self.flag_pose=True

        # Yaw align region
        if (self.flag_pose==True) and (self.flag_ori==False):
            yaw_error = self.normalize_angle(yaw_goal - self.current_yaw)
            if abs(yaw_error) > self.yaw_thresh:
                self.lateral_pid.reset()
                self.prev_cmd = Twist()
                w_cmd = self.yaw_pid.update(yaw_error, self.control_dt)
                w_cmd = max(-self.W_MAX, min(self.W_MAX, w_cmd))
                self.get_logger().info(f"Yaw-aligning: yaw_err={yaw_error:.3f}, yaw_cmd={w_cmd:.3f}")
                cmd = Twist(); cmd.linear.x = 0.0; cmd.angular.z = w_cmd
                self.publish_smoothed(cmd)
                return
            else:
                self.get_logger().info(f"Reached waypoint ({self.current_x:.2f},{self.current_y:.2f},{self.current_yaw:.2f})")
                self.flag_ori=True
                msg=Bool()
                msg.data=True
                self.nav_pub.publish(msg)
                self.stop_robot()
                return

        if self.flag_pose==True or self.flag_ori==True:
            return

        dist_to_goal = hypot(self.current_x - x_goal, self.current_y - y_goal)

        v_dwa, w_dwa, all_trajs, best_traj, info = self.dwa_modified(x_goal, y_goal, dist_to_goal)

        # storing last results
        if all_trajs: self.last_all_trajs = all_trajs
        if best_traj: self.last_best_traj = best_traj

        # Visualize trajectories in RViz (if enabled)
        if self.enable_viz:
            self.visualize_trajectories(all_trajs, best_traj, x_goal, y_goal)

        if best_traj is None or len(best_traj) == 0:
            cmd = Twist(); cmd.linear.x = 0.0; cmd.angular.z = 0.4
            self.publish_smoothed(cmd)
            return

        # lookahead
        if dist_to_goal < 0.5: look_idx = self.lookahead_index_close
        else: look_idx = self.lookahead_index
        look_idx = min(look_idx, len(best_traj)-1)

        px, py, _ = best_traj[look_idx]
        dx = px - self.current_x; dy = py - self.current_y
        xr = cos(-self.current_yaw) * dx - sin(-self.current_yaw) * dy
        yr = sin(-self.current_yaw) * dx + cos(-self.current_yaw) * dy
        lateral_error = yr
        w_correction = self.lateral_pid.update(lateral_error, self.control_dt)
        w_cmd = w_dwa + w_correction
        w_cmd = max(-self.W_MAX, min(self.W_MAX, w_cmd))
        v_cmd = v_dwa * max(0.05, 1.0 - self.turn_speed_redn * min(1.0, abs(w_correction)))
        v_cmd = max(0.0, min(self.V_MAX, v_cmd))

        cmd = Twist(); cmd.linear.x = float(v_cmd); cmd.angular.z = float(w_cmd)
        self.publish_smoothed(cmd)

    # DWA with progress and minimal-speed
    def dwa_modified(self, x_goal, y_goal, dist_to_goal):
        min_obs_dist = float('inf')
        if len(self.obstacles) > 0:
            dists = np.sqrt(self.obstacles[:,0]**2 + self.obstacles[:,1]**2)
            min_obs_dist = float(np.min(dists))

        # Adaptive minimum velocity: slow down in tight spaces
        enforced_v_min = self.V_MIN
        if dist_to_goal > self.dist_tolerance:
            # If very tight (obstacles < 0.8m), allow very slow movement
            if min_obs_dist < 0.8:
                enforced_v_min = max(enforced_v_min, 0.15)  # Reduce min speed in tight spaces
            else:
                enforced_v_min = max(enforced_v_min, self.min_traj_V)

        sampled_v_min = max((self.curr_vx + self.A_MIN * DEL_T), enforced_v_min)
        sampled_v_max = min((self.curr_vx + self.A_MAX * DEL_T), self.V_MAX)
        sampled_w_min = max((self.curr_w + self.AL_MIN * DEL_T), self.W_MIN)
        sampled_w_max = min((self.curr_w + self.AL_MAX * DEL_T), self.W_MAX)

        if sampled_v_min >= sampled_v_max:
            sampled_v_min, sampled_v_max = enforced_v_min, self.V_MAX
        if sampled_w_min >= sampled_w_max:
            sampled_w_min, sampled_w_max = self.W_MIN, self.W_MAX

        v_samples = np.linspace(sampled_v_min, sampled_v_max, V_SAMPLES)
        w_samples = np.linspace(sampled_w_min, sampled_w_max, W_SAMPLES)

        # Transform obstacles from robot frame to world frame ONCE
        if len(self.obstacles) > 0:
            obs_world_x = (self.current_x +
                           self.obstacles[:,0] * cos(self.current_yaw) -
                           self.obstacles[:,1] * sin(self.current_yaw))
            obs_world_y = (self.current_y +
                           self.obstacles[:,0] * sin(self.current_yaw) +
                           self.obstacles[:,1] * cos(self.current_yaw))
            obstacles_world = np.column_stack((obs_world_x, obs_world_y))
        else:
            obstacles_world = np.empty((0,2))

        all_trajs = []; best_traj = None
        best_score = -float('inf'); best_v = 0.0; best_w = 0.0
        valid_count = 0
        best_progress = 0.0
        current_dist = hypot(self.current_x - x_goal, self.current_y - y_goal)
        for v0 in v_samples:
            for w0 in w_samples:
                xi, yi, theta = self.current_x, self.current_y, self.current_yaw
                traj_points = []
                min_clearance = float('inf')
                collision = False
                steps = int(DEL_T / DT)
                # Simulate trajectory
                for step_idx in range(steps):
                    xi += v0 * cos(theta) * DT
                    yi += v0 * sin(theta) * DT
                    theta = (theta + w0 * DT + pi) % (2 * pi) - pi
                    # Collision check in world frame (every N steps or last step for safety)
                    if len(obstacles_world) > 0 and (step_idx % COLLISION_CHECK_INTERVAL == 0 or step_idx == steps - 1):
                        # Use squared distances for faster collision check
                        dists_sq = (obstacles_world[:,0] - xi)**2 + (obstacles_world[:,1] - yi)**2
                        step_min_sq = float(np.min(dists_sq))
                        if step_min_sq < self.safety_distance_sq:
                            collision = True
                            break
                        # Only compute sqrt for clearance tracking
                        step_min = np.sqrt(step_min_sq)
                        min_clearance = min(min_clearance, step_min)
                    traj_points.append((xi, yi, theta))
                # Skip if collision or no valid trajectory
                if collision or len(traj_points) == 0:
                    continue
                # Score the trajectory
                xf, yf, th_f = traj_points[-1]
                heading_angle = atan2(y_goal - yf, x_goal - xf)
                heading_diff = self.normalize_angle(heading_angle - th_f)
                heading_score_norm = 1.0 - abs(heading_diff) / pi
                if min_clearance == float('inf'):
                    clearance_norm = 1.0
                else:
                    clearance_clamped = min(min_clearance, self.max_clearance_norm)
                    clearance_norm = clearance_clamped / self.max_clearance_norm
                v_norm = 0.0
                if self.V_MAX - self.V_MIN > 1e-6:
                    v_norm = (v0 - self.V_MIN) / (self.V_MAX - self.V_MIN)
                    v_norm = np.clip(v_norm, 0.0, 1.0)
                final_dist = hypot(xf - x_goal, yf - y_goal)
                raw_progress = max(0.0, (current_dist - final_dist))
                denom = max(1e-3, current_dist)
                progress_norm = np.clip(raw_progress / denom, 0.0, 1.0)

                rotation_penalty_factor = 0.3 + 0.7 * heading_score_norm  # Range: 0.3 to 1.0
                w_norm = 1.0 - abs(w0) / self.W_MAX  # 1.0 for w=0, 0.0 for w=W_MAX
                w_norm = np.clip(w_norm, 0.0, 1.0)

                # Smoothness: prefer trajectories close to current velocity (reduces oscillation)
                v_diff = abs(v0 - self.curr_vx) / max(self.V_MAX, 1e-3)
                w_diff = abs(w0 - self.curr_w) / max(self.W_MAX, 1e-3)
                smoothness_norm = 1.0 - 0.5 * (v_diff + w_diff)
                smoothness_norm = np.clip(smoothness_norm, 0.0, 1.0)

                # Combined score with adaptive rotation penalty
                score = (ALPHA * heading_score_norm +
                        BETA * clearance_norm +
                        GAMMA * v_norm +
                        DELTA * progress_norm +
                        EPSILON * rotation_penalty_factor * w_norm +  # Adaptive rotation penalty
                        ZETA * smoothness_norm)
                # Store trajectory
                all_trajs.append(traj_points)
                valid_count += 1
                # Update best trajectory
                if score > best_score:
                    best_score = score
                    best_v = v0
                    best_w = w0
                    best_traj = traj_points
                    best_progress = progress_norm
                    # Early termination if we found an excellent trajectory
                    if best_score >= EARLY_TERMINATION_SCORE:
                        break
            # Early termination outer loop
            if best_score >= EARLY_TERMINATION_SCORE:
                break
        info = {'valid_count': valid_count, 'best_score': best_score, 'best_v': best_v, 'best_w': best_w, 'best_prog': best_progress}
        if best_traj is None:
            return 0.0, 0.0, all_trajs, None, info
        return best_v, best_w, all_trajs, best_traj, info

    def publish_smoothed(self, cmd: Twist):     # exponential smoothing and limits before publishing cmd_vel
        sm = Twist()
        sm.linear.x = self.smoothing_ratio * cmd.linear.x + (1.0 - self.smoothing_ratio) * self.prev_cmd.linear.x
        sm.angular.z = self.smoothing_ratio * cmd.angular.z + (1.0 - self.smoothing_ratio) * self.prev_cmd.angular.z
        sm.angular.z = max(-self.W_MAX, min(self.W_MAX, sm.angular.z))
        self.cmd_pub.publish(sm)
        self.prev_cmd = sm


    def stop_robot(self):
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0
        self.cmd_pub.publish(stop_msg)

    def visualize_all_waypoints(self):
        """Visualize all waypoints in RViz"""
        if self.waypoints is None:
            return
        marker_array = MarkerArray()
        x, y, yaw = self.waypoints

        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "waypoints"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.3
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.6)  # Orange
        marker_array.markers.append(marker)
        # Waypoint text label
        text_marker = Marker()
        text_marker.header.frame_id = "odom"
        text_marker.header.stamp = self.get_clock().now().to_msg()
        text_marker.ns = "waypoint_labels"
        text_marker.id = 1 + 100
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = x
        text_marker.pose.position.y = y
        text_marker.pose.position.z = 0.5
        text_marker.scale.z = 0.15
        text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        text_marker.text = f"WP_1"
        marker_array.markers.append(text_marker)

        self.waypoints_pub.publish(marker_array)

    def visualize_trajectories(self, all_trajs, best_traj, x_goal, y_goal):
        """Visualize all DWA trajectories and the best trajectory in RViz"""
        # Visualize all candidate trajectories
        marker_array = MarkerArray()
        for idx, traj in enumerate(all_trajs):
            if len(traj) == 0:
                continue
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "dwa_trajectories"
            marker.id = idx
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.01  # Line width
            marker.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.3)  # Light blue, semi-transparent
            marker.pose.orientation.w = 1.0

            for (x, y, _) in traj:
                p = Point()
                p.x = x; p.y = y; p.z = 0.05
                marker.points.append(p)
            marker_array.markers.append(marker)
        self.traj_pub.publish(marker_array)

        # Visualize best trajectory
        if best_traj and len(best_traj) > 0:
            best_marker = Marker()
            best_marker.header.frame_id = "odom"
            best_marker.header.stamp = self.get_clock().now().to_msg()
            best_marker.ns = "best_trajectory"
            best_marker.id = 0
            best_marker.type = Marker.LINE_STRIP
            best_marker.action = Marker.ADD
            best_marker.scale.x = 0.05  # Thicker line
            best_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # Bright green
            best_marker.pose.orientation.w = 1.0

            for (x, y, _) in best_traj:
                p = Point()
                p.x = x; p.y = y; p.z = 0.1
                best_marker.points.append(p)
            self.best_traj_pub.publish(best_marker)

        # Visualize goal marker
        goal_marker = Marker()
        goal_marker.header.frame_id = "odom"
        goal_marker.header.stamp = self.get_clock().now().to_msg()
        goal_marker.ns = "goal"
        goal_marker.id = 0
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        goal_marker.pose.position.x = x_goal
        goal_marker.pose.position.y = y_goal
        goal_marker.pose.position.z = 0.2
        goal_marker.pose.orientation.w = 1.0
        goal_marker.scale.x = 0.3
        goal_marker.scale.y = 0.3
        goal_marker.scale.z = 0.3

        goal_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)  # Red for waypoint goal
        self.goal_pub.publish(goal_marker)

    @staticmethod
    def normalize_angle(angle):
        while angle > pi: angle -= 2.0*pi
        while angle < -pi: angle += 2.0*pi
        return angle

class PID:      # PID controller class
    def __init__(self, kp, ki, kd, out_min=None, out_max=None):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.out_min = out_min; self.out_max = out_max
        self.integral = 0.0
        self.prev_err = None
    def reset(self):
        self.integral = 0.0; self.prev_err = None
    def update(self, e, dt):
        if dt <= 0: return 0.0
        deriv = 0.0 if (self.prev_err is None) else (e - self.prev_err) / dt
        self.integral += e * dt
        out = self.kp * e + self.ki * self.integral + self.kd * deriv
        self.prev_err = e
        if (self.out_min is not None) and (out < self.out_min): out = self.out_min
        if (self.out_max is not None) and (out > self.out_max): out = self.out_max
        return out

def main(args=None):
    rclpy.init()
    robot_controller = ebotNav3B()
    executor = MultiThreadedExecutor()
    executor.add_node(robot_controller)
    try:
        executor.spin()

    except KeyboardInterrupt:
        robot_controller.get_logger().warn(
            'Keyboard interrupt received, stopping robot...'
        )
        robot_controller.stop_robot()

    finally:
        executor.shutdown()
        robot_controller.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
    main()