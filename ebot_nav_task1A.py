#!/usr/bin/env python3

# ebot_nav_task1A.py    DWA-based navigation to waypoints with yaw-alignment at each waypoint
import os
import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Twist #, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion
from math import atan2, pi ,hypot, cos, sin

# from visualization_msgs.msg import Marker, MarkerArray
from concurrent.futures import ThreadPoolExecutor

# Tunable params of DWA planner      DWA --> sampling + forward rollout + scoring loop
ALPHA = 1.0                 # for more heading towards goal
BETA = 0.8                  # for more clearance
GAMMA = 0.6                 # for faster speeds
PROGRESS_WEIGHT = 1.2       # for reducing distance to current waypoint


ROBOT_RADIUS = 0.20
SAFETY_MARGIN = 0.30
SAFETY_DISTANCE = ROBOT_RADIUS + SAFETY_MARGIN

V_MIN_GLOBAL, V_MAX_GLOBAL = 0.0, 1.0
W_MIN_GLOBAL, W_MAX_GLOBAL = -1.5, 1.5
A_MIN, A_MAX = -1.0, 1.0
AL_MIN, AL_MAX = -0.2, 0.2

DEL_T = 2.0
DT = 0.05
V_SAMPLES = 7
W_SAMPLES = 12

MAX_CLEARANCE_NORM = 3.0
LOOKAHEAD_INDEX_DEFAULT = 3
LOOKAHEAD_INDEX_CLOSE = 0
CMD_SMOOTHING_ALPHA = 0.7

LATERAL_PID_GAINS = (0.8, 0.0, 0.03)
YAW_PID_GAINS     = (1.5, 0.0, 0.04)

TURN_SPEED_REDUCTION_K = 0.6

LIDAR_MIN_ANGLE = -60.0 * pi / 180.0    # usable range of LIDAR on ebot for navigation(full range=[-135,+135])
LIDAR_MAX_ANGLE =  60.0 * pi / 180.0

WAYPOINT_DIST_THRESH = 0.20
YAW_ANGLE_THRESH = 0.05
VEL_STABLE_THRESH = 0.06
ANG_VEL_STABLE_THRESH = 0.06
WAYPOINT_STABLE_CYCLES = 8

MAX_PUBLISHED_V = V_MAX_GLOBAL
MAX_PUBLISHED_W = W_MAX_GLOBAL

MIN_MOVE_DIST = 0.15    # if more than this, prefer moving
MIN_MOVE_SPEED = 0.08   # minimal forward speed to allow in candidate trajectories

class PID:      # PID controller
    def __init__(self, kp, ki, kd, out_min=None, out_max=None):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.out_min = out_min; self.out_max = out_max
        self.integral = 0.0
        self.prev_err = None
    def reset(self):
        self.integral = 0.0; self.prev_err = None
    def update(self, err, dt):
        if dt <= 0: return 0.0
        deriv = 0.0 if (self.prev_err is None) else (err - self.prev_err) / dt
        self.integral += err * dt
        out = self.kp * err + self.ki * self.integral + self.kd * deriv
        self.prev_err = err
        if (self.out_min is not None) and (out < self.out_min): out = self.out_min
        if (self.out_max is not None) and (out > self.out_max): out = self.out_max
        return out

class ebotNav(Node):
    def __init__(self):
        super().__init__('ebot_nav_task1A')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        # self.marker_pub = self.create_publisher(MarkerArray, '/dwa_trajectories', 10)

        cpu_count = os.cpu_count() or 1
        self.pool = ThreadPoolExecutor(max_workers=min(32, cpu_count + 4))

        # state
        self.current_x = 0.0; self.current_y = 0.0; self.current_yaw = 0.0
        self.curr_vx = 0.0; self.curr_w = 0.0
        self.obstacles = np.empty((0,2))

        self.waypoints = [
            (-1.53, -1.95, 1.57),
            (0.13,  1.24, 0.0),
            (0.38, -3.32, -1.57)
        ]
        self.w_index = 0

        kp, ki, kd = LATERAL_PID_GAINS
        self.lateral_pid = PID(kp, ki, kd, out_min=-1.5, out_max=1.5)
        ykp, yki, ykd = YAW_PID_GAINS
        self.yaw_pid = PID(ykp, yki, ykd, out_min=-1.5, out_max=1.5)

        self.waypoint_stable_counter = 0
        self.control_dt = DT
        self.timer = self.create_timer(self.control_dt, self.control_loop)
        self.prev_cmd = Twist()

        self.last_all_trajs = []
        self.last_best_traj = None

        self.get_logger().info("ebot_nav_task1A node started")

    def odom_callback(self, msg: Odometry):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_yaw = yaw
        self.curr_vx = msg.twist.twist.linear.x
        self.curr_w = msg.twist.twist.angular.z

    def scan_callback(self, msg: LaserScan):
        pts = []
        angle = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                if LIDAR_MIN_ANGLE <= angle <= LIDAR_MAX_ANGLE:
                    x = r * cos(angle); y = r * sin(angle); pts.append((x,y))
            angle += msg.angle_increment
        if len(pts) > 0:
            self.obstacles = np.array(pts)
        else:
            self.obstacles = np.empty((0,2))

    def control_loop(self):

        if self.w_index >= len(self.waypoints):
            self.stop_robot(); 
            self.get_logger().info("All waypoints reached."); 
            self.timer.cancel(); 
            return

        x_goal, y_goal, yaw_goal = self.waypoints[self.w_index]
        dist_to_goal = hypot(self.current_x - x_goal, self.current_y - y_goal)

        # Yaw align region
        if dist_to_goal < WAYPOINT_DIST_THRESH:
            yaw_error = self.normalize_angle(yaw_goal - self.current_yaw)

            if abs(yaw_error) > YAW_ANGLE_THRESH:
                self.lateral_pid.reset()
                self.prev_cmd = Twist()
                w_cmd = self.yaw_pid.update(yaw_error, self.control_dt)
                w_cmd = max(-MAX_PUBLISHED_W, min(MAX_PUBLISHED_W, w_cmd))
                self.get_logger().info(f"Yaw-aligning: yaw_err={yaw_error:.3f}, yaw_cmd={w_cmd:.3f}")
                cmd = Twist(); cmd.linear.x = 0.0; cmd.angular.z = w_cmd
                self.publish_smoothed(cmd)
                # self.publish_trajectories(self.last_all_trajs, self.last_best_traj)
                return
            else:
                if abs(self.curr_vx) < VEL_STABLE_THRESH and abs(self.curr_w) < ANG_VEL_STABLE_THRESH:
                    self.waypoint_stable_counter += 1
                else:
                    self.waypoint_stable_counter = 0
                # self.publish_trajectories(self.last_all_trajs, self.last_best_traj)
                if self.waypoint_stable_counter >= WAYPOINT_STABLE_CYCLES:
                    self.get_logger().info(f"Reached waypoint {self.w_index+1} (stable).({self.current_x:.2f},{self.current_y:.2f},{self.current_yaw:.2f})")
                    self.w_index += 1
                    self.waypoint_stable_counter = 0
                    self.lateral_pid.reset(); self.yaw_pid.reset()
                else:
                    cmd = Twist(); cmd.linear.x = 0.0; cmd.angular.z = 0.0
                    self.publish_smoothed(cmd)
                return

        # Running modified DWA to get (v,w)
        v_dwa, w_dwa, all_trajs, best_traj, info = self.dwa_modified(x_goal, y_goal, dist_to_goal)

        # storing last results
        if all_trajs: self.last_all_trajs = all_trajs
        if best_traj: self.last_best_traj = best_traj

        if best_traj is None or len(best_traj) == 0:
            cmd = Twist(); cmd.linear.x = 0.0; cmd.angular.z = 0.4
            self.publish_smoothed(cmd); 
            # self.publish_trajectories(all_trajs, best_traj); 
            return

        # lookahead
        if dist_to_goal < 0.5: look_idx = LOOKAHEAD_INDEX_CLOSE
        else: look_idx = LOOKAHEAD_INDEX_DEFAULT
        look_idx = min(look_idx, len(best_traj)-1)

        px, py, _ = best_traj[look_idx]
        dx = px - self.current_x; dy = py - self.current_y
        xr = cos(-self.current_yaw) * dx - sin(-self.current_yaw) * dy
        yr = sin(-self.current_yaw) * dx + cos(-self.current_yaw) * dy
        lateral_error = yr
        w_correction = self.lateral_pid.update(lateral_error, self.control_dt)
        w_cmd = w_dwa + w_correction
        w_cmd = max(-MAX_PUBLISHED_W, min(MAX_PUBLISHED_W, w_cmd))
        v_cmd = v_dwa * max(0.05, 1.0 - TURN_SPEED_REDUCTION_K * min(1.0, abs(w_correction)))
        v_cmd = max(0.0, min(MAX_PUBLISHED_V, v_cmd))

        cmd = Twist(); cmd.linear.x = float(v_cmd); cmd.angular.z = float(w_cmd)
        self.publish_smoothed(cmd)
        # self.publish_trajectories(all_trajs, best_traj)

    # DWA with path progress and minimal-speed 
    def dwa_modified(self, x_goal, y_goal, dist_to_goal):
        enforced_v_min = V_MIN_GLOBAL
        if dist_to_goal > MIN_MOVE_DIST:
            enforced_v_min = max(enforced_v_min, MIN_MOVE_SPEED)

        sampled_v_min = max((self.curr_vx + A_MIN * DEL_T), enforced_v_min)
        sampled_v_max = min((self.curr_vx + A_MAX * DEL_T), V_MAX_GLOBAL)
        sampled_w_min = max((self.curr_w + AL_MIN * DEL_T), W_MIN_GLOBAL)
        sampled_w_max = min((self.curr_w + AL_MAX * DEL_T), W_MAX_GLOBAL)

        if sampled_v_min >= sampled_v_max:
            sampled_v_min, sampled_v_max = enforced_v_min, V_MAX_GLOBAL
        if sampled_w_min >= sampled_w_max:
            sampled_w_min, sampled_w_max = W_MIN_GLOBAL, W_MAX_GLOBAL

        v_samples = np.linspace(sampled_v_min, sampled_v_max, V_SAMPLES)
        w_samples = np.linspace(sampled_w_min, sampled_w_max, W_SAMPLES)

        all_trajs = []; best_traj = None
        best_score = -float('inf'); best_v = 0.0; best_w = 0.0
        valid_count = 0
        best_progress = 0.0

        current_dist = hypot(self.current_x - x_goal, self.current_y - y_goal)

        def evaluate_pair(v0, w0):
            xi, yi, theta = self.current_x, self.current_y, self.current_yaw
            traj_points = []; min_clearance = float('inf'); collision = False
            steps = int(DEL_T / DT)
            for _ in range(steps):
                xi += v0 * cos(theta) * DT
                yi += v0 * sin(theta) * DT
                theta = (theta + w0 * DT + pi) % (2 * pi) - pi

                if len(self.obstacles) > 0:
                    dx = xi - self.current_x; dy = yi - self.current_y
                    xr = cos(-self.current_yaw) * dx - sin(-self.current_yaw) * dy
                    yr = sin(-self.current_yaw) * dx + cos(-self.current_yaw) * dy
                    dists = np.sqrt((self.obstacles[:,0] - xr)**2 + (self.obstacles[:,1] - yr)**2)
                    step_min = float(np.min(dists)); min_clearance = min(min_clearance, step_min)
                    if step_min < SAFETY_DISTANCE:
                        collision = True; break
                traj_points.append((xi, yi, theta))

            if collision or len(traj_points) == 0:
                return (-float('inf'), v0, w0, traj_points, 0.0, 0.0)

            xf, yf, th_f = traj_points[-1]
            heading_angle = atan2(y_goal - yf, x_goal - xf)
            heading_diff = self.normalize_angle(heading_angle - th_f)
            heading_score_norm = 1.0 - abs(heading_diff) / pi

            if min_clearance == float('inf'):
                clearance_norm = 1.0
            else:
                clearance_clamped = min(min_clearance, MAX_CLEARANCE_NORM)
                clearance_norm = clearance_clamped / MAX_CLEARANCE_NORM

            v_norm = 0.0
            if V_MAX_GLOBAL - V_MIN_GLOBAL > 1e-6:
                v_norm = (v0 - V_MIN_GLOBAL) / (V_MAX_GLOBAL - V_MIN_GLOBAL)
                v_norm = np.clip(v_norm, 0.0, 1.0)

            # progress- closeness of trajectory to goal_i relative to current_dist
            final_dist = hypot(xf - x_goal, yf - y_goal)
            raw_progress = max(0.0, (current_dist - final_dist))  # +ve if distance reduced
            
            denom = max(1e-3, current_dist)     
            progress_norm = np.clip(raw_progress / denom, 0.0, 1.0)     # normalization of progress by current_dist

            # combined score (higher is better)
            score = ALPHA * heading_score_norm + BETA * clearance_norm + GAMMA * v_norm + PROGRESS_WEIGHT * progress_norm
            return (score, v0, w0, traj_points, min_clearance, progress_norm)

        # submission of all (v0,w0) pairs to thread pool
        futures = []
        for v0 in v_samples:
            for w0 in w_samples:
                futures.append(self.pool.submit(evaluate_pair, float(v0), float(w0)))

        for f in futures:
            try:
                score, v0, w0, traj_points, min_clearance, progress_norm = f.result()
            except Exception as e:
                self.get_logger().warning(f"DWA worker failed: {e}")
                continue

            if traj_points:
                all_trajs.append(traj_points)
                valid_count += 1

            if score > best_score:
                best_score = score; best_v = v0; best_w = w0; best_traj = traj_points; best_progress = progress_norm

        info = {'valid_count': valid_count, 'best_score': best_score, 'best_v': best_v, 'best_w': best_w, 'best_prog': best_progress}
        if best_traj is None:
            return 0.0, 0.0, all_trajs, None, info
        return best_v, best_w, all_trajs, best_traj, info

    # visualization of KLD-sampling results with best trajectory
    # def publish_trajectories(self, all_trajs, best_traj):
    #     marker_array = MarkerArray(); marker_id = 0
    #     for traj in all_trajs:
    #         marker = Marker(); marker.header.frame_id = "odom"; marker.header.stamp = self.get_clock().now().to_msg()
    #         marker.ns = "dwa_trajs"; marker.id = marker_id; marker.type = Marker.LINE_STRIP; marker.action = Marker.ADD
    #         marker.scale.x = 0.02; marker.color.r = 0.5; marker.color.g = 0.5; marker.color.b = 0.5; marker.color.a = 0.4
    #         pts = []
    #         for (x,y,_) in traj:
    #             p = Point(); p.x=float(x); p.y=float(y); p.z=0.0; pts.append(p)
    #         marker.points = pts; marker_array.markers.append(marker); marker_id += 1
    #     if best_traj:
    #         marker = Marker(); marker.header.frame_id = "odom"; marker.header.stamp = self.get_clock().now().to_msg()
    #         marker.ns = "dwa_trajs"; marker.id = marker_id; marker.type = Marker.LINE_STRIP; marker.action = Marker.ADD
    #         marker.scale.x = 0.04; marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0; marker.color.a = 1.0
    #         pts = []
    #         for (x,y,_) in best_traj:
    #             p = Point(); p.x=float(x); p.y=float(y); p.z=0.0; pts.append(p)
    #         marker.points = pts; marker_array.markers.append(marker)
    #     self.marker_pub.publish(marker_array)

    def publish_smoothed(self, cmd: Twist):     # exponential smoothing and limits before publishing cmd_vel
        sm = Twist()
        sm.linear.x = CMD_SMOOTHING_ALPHA * cmd.linear.x + (1.0 - CMD_SMOOTHING_ALPHA) * self.prev_cmd.linear.x
        sm.angular.z = CMD_SMOOTHING_ALPHA * cmd.angular.z + (1.0 - CMD_SMOOTHING_ALPHA) * self.prev_cmd.angular.z
        sm.linear.x = max(0.0, min(MAX_PUBLISHED_V, sm.linear.x))
        sm.angular.z = max(-MAX_PUBLISHED_W, min(MAX_PUBLISHED_W, sm.angular.z))
        self.cmd_pub.publish(sm)
        self.prev_cmd = sm

    def stop_robot(self):
        self.cmd_pub.publish(Twist())
    
    @staticmethod   
    def normalize_angle(angle):
        while angle > pi: angle -= 2.0*pi
        while angle < -pi: angle += 2.0*pi
        return angle

    def __del__(self):
        try: self.pool.shutdown(wait=False)
        except Exception: pass

def main(args=None):
    rclpy.init(args=args)
    node = ebotNav()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown() 

if __name__ == '__main__':
    main()