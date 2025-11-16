#!/usr/bin/env python3
import os
import time
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist #, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion
from math import atan2, pi ,hypot, cos, sin

from concurrent.futures import ThreadPoolExecutor

# DWA-based navigation to waypoints with yaw-alignment at each waypoint
# DWA PARAMS --> sampling + forward rollout + scoring loop
ALPHA = 1.0                 # for more heading towards goal
BETA = 0.9                  # for more clearance
GAMMA = 0.9                 # for faster speeds
DELTA = 1.2       # for reducing distance to current waypoint

BOT_RADIUS = 0.20
SAFETY_MARGIN = 0.30
SAFETY_DISTANCE = BOT_RADIUS + SAFETY_MARGIN

V_MIN, V_MAX = 0.0, 2.0
W_MIN, W_MAX = -1.5, 1.5
A_MIN, A_MAX = -1.0, 1.0
AL_MIN, AL_MAX = -0.2, 0.2

DEL_T = 2.0
DT = 0.05
V_SAMPLES = 8
W_SAMPLES = 16

MAX_CLEARANCE_NORM = 3.0
LOOKAHEAD_INDEX = 3
LOOKAHEAD_INDEX_CLOSE = 0
SMOOTHING_ALPHA = 0.7

PATH_PID= (0.8, 0.0, 0.03)
YAW_PID= (1.5, 0.0, 0.04)

TURN_SPEED_REDUCTION_K = 0.6

LIDAR_MIN_ANGLE = -60.0 * pi / 180.0    # usable range of LIDAR on ebot for navigation(full range=[-135,+135])
LIDAR_MAX_ANGLE =  60.0 * pi / 180.0

TARGET_THRESH = 0.20
YAW_THRESH = 0.05
V_STABLE_THRESH = 0.06
W_STABLE_THRESH = 0.06
TARGET_STABLE_CYCLES = 8

V_PUB_MAX = V_MAX
W_PUB_MAX = W_MAX

DIST_TOL = 0.15    # if more than this, moving to target
MIN_TRAJ_V = 0.08   # minimal forward speed for candidate trajectories

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

class ebotNav(Node):
    def __init__(self):
        super().__init__('ebot_nav_task2A')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.shape_sub = self.create_subscription(String, '/detection_status', self.shape_callback, 10)

        # Robot initial state
        self.current_x = -1.5339
        self.current_y = -6.6156
        self.current_yaw = 1.57  # Facing forwards
        self.laser_data = []

        self.waypoints = [
            [0.26, -1.95, 1.57],       # P1: (x1, y1, yaw1)
            [-1.48, -0.67, -1.57],          # P2: (x2, y2, yaw2)
            [-1.53, -6.61, -1.57]        # P3: (x3, y3, yaw3)
        ]
        self.w_index=0
                
        cpu_count = os.cpu_count() or 1
        self.pool = ThreadPoolExecutor(max_workers=min(32, cpu_count + 4))

        self.curr_vx = 0.0; self.curr_w = 0.0
        self.obstacles = np.empty((0,2))

        kp, ki, kd = PATH_PID
        self.lateral_pid = PID(kp, ki, kd, out_min=-1.5, out_max=1.5)
        ykp, yki, ykd = YAW_PID
        self.yaw_pid = PID(ykp, yki, ykd, out_min=-1.5, out_max=1.5)

        self.target_cycle_count = 0
        self.control_dt = DT
        self.timer = self.create_timer(self.control_dt, self.control_loop)
        self.prev_cmd = Twist()

        self.last_all_trajs = []
        self.last_best_traj = None

        self.get_logger().info("ebot_nav_task2A node started")

    def odom_callback(self, msg: Odometry):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_yaw = yaw
        self.curr_vx = msg.twist.twist.linear.x
        self.curr_w = msg.twist.twist.angular.z
        if(hypot(self.current_x+1.5339, self.current_y+6.6156)<=0.05):
            init_msg = Twist()
            init_msg.angular.z = -16.0
            self.cmd_pub.publish(init_msg)  # rotate at start
            time.sleep(1.5)

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

    def shape_callback(self, msg: String):
        self.detected_shape_status = msg.data
        if self.detected_shape_status.startswith("DOCK_STATION"):
            self.get_logger().info("Docking station detected. Stopping the robot.")
            self.stop_robot()
        elif self.detected_shape_status.startswith("FERTILIZER_REQUIRED"):
            self.get_logger().info("Fertilizer required detected.")
            self.hold_position()
        elif self.detected_shape_status.startswith("BAD_HEALTH"):
            self.get_logger().info("Bad health Detected.")
            self.hold_position()

    def control_loop(self):

        if self.w_index >= len(self.waypoints):
            self.stop_robot(); 
            self.get_logger().info("All waypoints reached."); 
            self.timer.cancel(); 
            return

        x_goal, y_goal, yaw_goal = self.waypoints[self.w_index]
        dist_to_goal = hypot(self.current_x - x_goal, self.current_y - y_goal)

        # Yaw align region
        if dist_to_goal < TARGET_THRESH:
            yaw_error = self.normalize_angle(yaw_goal - self.current_yaw)

            if abs(yaw_error) > YAW_THRESH:
                self.lateral_pid.reset()
                self.prev_cmd = Twist()
                w_cmd = self.yaw_pid.update(yaw_error, self.control_dt)
                w_cmd = max(-W_PUB_MAX, min(W_PUB_MAX, w_cmd))
                self.get_logger().info(f"Yaw-aligning: yaw_err={yaw_error:.3f}, yaw_cmd={w_cmd:.3f}")
                cmd = Twist(); cmd.linear.x = 0.0; cmd.angular.z = w_cmd
                self.publish_smoothed(cmd)
                return
            else:
                if abs(self.curr_vx) < V_STABLE_THRESH and abs(self.curr_w) < W_STABLE_THRESH:
                    self.target_cycle_count += 1
                else:
                    self.target_cycle_count = 0
                if self.target_cycle_count >= TARGET_STABLE_CYCLES:
                    self.get_logger().info(f"Reached waypoint {self.w_index+1} (stable).({self.current_x:.2f},{self.current_y:.2f},{self.current_yaw:.2f})")
                    self.w_index += 1
                    self.target_cycle_count = 0
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
            return

        # lookahead
        if dist_to_goal < 0.5: look_idx = LOOKAHEAD_INDEX_CLOSE
        else: look_idx = LOOKAHEAD_INDEX
        look_idx = min(look_idx, len(best_traj)-1)

        px, py, _ = best_traj[look_idx]
        dx = px - self.current_x; dy = py - self.current_y
        xr = cos(-self.current_yaw) * dx - sin(-self.current_yaw) * dy
        yr = sin(-self.current_yaw) * dx + cos(-self.current_yaw) * dy
        lateral_error = yr
        w_correction = self.lateral_pid.update(lateral_error, self.control_dt)
        w_cmd = w_dwa + w_correction
        w_cmd = max(-W_PUB_MAX, min(W_PUB_MAX, w_cmd))
        v_cmd = v_dwa * max(0.05, 1.0 - TURN_SPEED_REDUCTION_K * min(1.0, abs(w_correction)))
        v_cmd = max(0.0, min(V_PUB_MAX, v_cmd))

        cmd = Twist(); cmd.linear.x = float(v_cmd); cmd.angular.z = float(w_cmd)
        self.publish_smoothed(cmd)

    # DWA with progress and minimal-speed 
    def dwa_modified(self, x_goal, y_goal, dist_to_goal):
        enforced_v_min = V_MIN
        if dist_to_goal > DIST_TOL:
            enforced_v_min = max(enforced_v_min, MIN_TRAJ_V)

        sampled_v_min = max((self.curr_vx + A_MIN * DEL_T), enforced_v_min)
        sampled_v_max = min((self.curr_vx + A_MAX * DEL_T), V_MAX)
        sampled_w_min = max((self.curr_w + AL_MIN * DEL_T), W_MIN)
        sampled_w_max = min((self.curr_w + AL_MAX * DEL_T), W_MAX)

        if sampled_v_min >= sampled_v_max:
            sampled_v_min, sampled_v_max = enforced_v_min, V_MAX
        if sampled_w_min >= sampled_w_max:
            sampled_w_min, sampled_w_max = W_MIN, W_MAX

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
            if V_MAX - V_MIN > 1e-6:
                v_norm = (v0 - V_MIN) / (V_MAX - V_MIN)
                v_norm = np.clip(v_norm, 0.0, 1.0)

            final_dist = hypot(xf - x_goal, yf - y_goal)
            raw_progress = max(0.0, (current_dist - final_dist))  # +ve if distance reduced
            
            denom = max(1e-3, current_dist)     
            progress_norm = np.clip(raw_progress / denom, 0.0, 1.0)     # normalization of progress by current_dist

            # combined score (higher is better)
            score = ALPHA * heading_score_norm + BETA * clearance_norm + GAMMA * v_norm + DELTA * progress_norm
            return (score, v0, w0, traj_points, min_clearance, progress_norm)

        # Multithreading: submission of all (v0,w0) pairs to thread pool
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

    def publish_smoothed(self, cmd: Twist):     # exponential smoothing and limits before publishing cmd_vel
        sm = Twist()
        sm.linear.x = SMOOTHING_ALPHA * cmd.linear.x + (1.0 - SMOOTHING_ALPHA) * self.prev_cmd.linear.x
        sm.angular.z = SMOOTHING_ALPHA * cmd.angular.z + (1.0 - SMOOTHING_ALPHA) * self.prev_cmd.angular.z
        sm.linear.x = max(0.0, min(V_PUB_MAX, sm.linear.x))
        sm.angular.z = max(-W_PUB_MAX, min(W_PUB_MAX, sm.angular.z))
        self.cmd_pub.publish(sm)
        self.prev_cmd = sm
    
    def hold_position(self):
        self.cmd_pub.publish(Twist())
        self.get_logger().info('Shape detected. Waiting for 2 seconds...')
        time.sleep(2.0)
        self.get_logger().info('Done waiting...Resuming navigation.')
        self.cmd_pub.publish(Twist())

    def stop_robot(self):
        self.cmd_pub.publish(Twist())
    
    @staticmethod   
    def normalize_angle(angle):
        while angle > pi: angle -= 2.0*pi
        while angle < -pi: angle += 2.0*pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    node = ebotNav()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown() 

if __name__ == '__main__':
    main()