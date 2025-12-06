#!/usr/bin/env python3
import time
import numpy as np
from math import hypot, cos, sin, atan2, pi
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from tf_transformations import euler_from_quaternion

# DWA PARAMS --> sampling + forward rollout + scoring loop
ALPHA = 1.0                 # for more heading towards goal
BETA = 0.9                  # for more clearance
GAMMA = 1.5                 # for faster speeds
DELTA = 0.9       # for reducing distance to current waypoint

DEL_T = 2.0
DT = 0.05
V_SAMPLES = 9
W_SAMPLES = 16

class ebotNav3B(Node):
    def __init__(self):
        super().__init__('ebot_nav')

        self.create_subscription(Odometry,"/odom",self.odom_callback,10)
        self.create_subscription(LaserScan,"/scan",self.scan_callback,10)
        self.create_subscription(String,"/detection_status",self.shape_callback,10)
        self.cmd_pub=self.create_publisher(Twist,"/cmd_vel",10)
        self.flag_stop=False

        # Robot initial state
        self.current_x = -1.5339
        self.current_y = -6.6156
        self.current_yaw = 1.57  # Facing forwards
        self.detected_shape_status = None

        self.waypoints = [
            [0.26, -1.95, 1.57],            # P1: (x1, y1, yaw1)
            [-1.48, -0.67, -1.57],          # P2: (x2, y2, yaw2)
            [-1.53, -6.61, -1.57]           # P3: (x3, y3, yaw3)
        ]
        self.w_index=0

        self.curr_vx = 0.0; self.curr_w = 0.0
        self.obstacles = np.empty((0,2))

        kp, ki, kd = (0.8, 0.0, 0.03)       # Path PID values
        self.lateral_pid = PID(kp, ki, kd, out_min=-1.5, out_max=2.0)
        ykp, yki, ykd = (20, 0.0, 0.02)     # Yaw PID values
        self.yaw_pid = PID(ykp, yki, ykd, out_min=-1.5, out_max=1.5)

        self.target_cycle_count = 0
        self.control_dt = DT
        self.timer = self.create_timer(self.control_dt, self.control_loop)
        self.prev_cmd = Twist()

        self.bot_radius = 0.25; self.safety_margin = 0.15
        self.safety_distance = self.bot_radius + self.safety_margin

        self.min_lidar_angle= -70.0 * pi / 180.0    # usable range of LIDAR on ebot for navigation(full range=[-135,+135])
        self.max_lidar_angle= 70.0 * pi / 180.0  

        self.V_MIN, self.V_MAX = 0.0, 2.0
        self.W_MIN, self.W_MAX = -1.5, 1.5
        self.A_MIN, self.A_MAX = -1.0, 1.0
        self.AL_MIN, self.AL_MAX = -0.2, 0.2
        
        self.max_clearance_norm = 3.0
        self.lookahead_index = 3
        self.lookahead_index_close = 0
        self.smoothing_ratio = 0.7

        self.turn_speed_redn = 0.7
        self.target_thresh = 0.15
        self.yaw_thresh = 0.05
        self.v_stable_thresh = 0.06
        self.w_stable_thresh = 0.06
        self.target_stable_cycle = 8

        self.dist_tolerance = 0.15    # if more than this, moving to target
        self.min_traj_V = 0.3  # minimal forward speed for candidate trajectories
        
        self.last_all_trajs = []
        self.last_best_traj = None
        self.start_time = time.time()
        self.get_logger().info("ebot_nav_task2A node started")


    def odom_callback(self,msg:Odometry):
        self.current_x=msg.pose.pose.position.x
        self.current_y=msg.pose.pose.position.y
        q=msg.pose.pose.orientation
        _,_,self.current_yaw=euler_from_quaternion([q.x,q.y,q.z,q.w])
        
        self.curr_vx = msg.twist.twist.linear.x
        self.curr_w = msg.twist.twist.angular.z

    def shape_callback(self,msg:String):
        self.detected_shape_status = msg.data
        if self.detected_shape_status.startswith("DOCK_STATION"):
            self.get_logger().info(f"{self.detected_shape_status}, Docking here. Stopping the robot.")
            self.flag_stop=True
            self.hold_position()
        elif self.detected_shape_status.startswith("FERTILIZER_REQUIRED"):
            self.get_logger().info(f"{self.detected_shape_status}, Delivering Fertilizer.")
            self.flag_stop=True
            self.hold_position()
        elif self.detected_shape_status.startswith("BAD_HEALTH"):
            self.get_logger().info(f"{self.detected_shape_status}, Found Bad health here.")
            self.flag_stop=True
            self.hold_position()

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
        if self.flag_stop:    return

        if self.w_index >= len(self.waypoints):
            self.stop_robot()
            self.get_logger().info("All waypoints reached.")
            self.get_logger().info(f"Total navigation time: {(time.time() - self.start_time):.2f} seconds")
            self.timer.cancel()
            return
        if(hypot(self.current_x+1.5339, self.current_y+6.6156)<=0.1 and self.current_yaw>=0.15):
            init_msg = Twist()
            init_msg.linear.x = 0.0
            init_msg.angular.z = -4.0
            self.cmd_pub.publish(init_msg)  # rotate at start
            return

        x_goal, y_goal, yaw_goal = self.waypoints[self.w_index]
        dist_to_goal = hypot(self.current_x - x_goal, self.current_y - y_goal)

        # Yaw align region
        if dist_to_goal < self.target_thresh:
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
                if abs(self.curr_vx) < self.v_stable_thresh and abs(self.curr_w) < self.w_stable_thresh:
                    self.target_cycle_count += 1
                else:
                    self.target_cycle_count = 0
                if self.target_cycle_count >= self.target_stable_cycle:
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
        """DWA - sequential evaluation"""
        enforced_v_min = self.V_MIN
        if dist_to_goal > self.dist_tolerance:
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

        all_trajs = []; best_traj = None
        best_score = -float('inf'); best_v = 0.0; best_w = 0.0
        valid_count = 0
        best_progress = 0.0
        current_dist = hypot(self.current_x - x_goal, self.current_y - y_goal)
        # Sequential evaluation (single/default-threaded)
        for v0 in v_samples:
            for w0 in w_samples:
                xi, yi, theta = self.current_x, self.current_y, self.current_yaw
                traj_points = []
                min_clearance = float('inf')
                collision = False
                steps = int(DEL_T / DT)
                # Simulate trajectory
                for _ in range(steps):
                    xi += v0 * cos(theta) * DT
                    yi += v0 * sin(theta) * DT
                    theta = (theta + w0 * DT + pi) % (2 * pi) - pi
                    # Collision check
                    if len(self.obstacles) > 0:
                        dx = xi - self.current_x
                        dy = yi - self.current_y
                        xr = cos(-self.current_yaw) * dx - sin(-self.current_yaw) * dy
                        yr = sin(-self.current_yaw) * dx + cos(-self.current_yaw) * dy
                        dists = np.sqrt((self.obstacles[:,0] - xr)**2 + (self.obstacles[:,1] - yr)**2)
                        step_min = float(np.min(dists))
                        min_clearance = min(min_clearance, step_min)
                        if step_min < self.safety_distance:
                            collision = True
                            break
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
                # Combined score
                score = ALPHA * heading_score_norm + BETA * clearance_norm + GAMMA * v_norm + DELTA * progress_norm
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
        info = {'valid_count': valid_count, 'best_score': best_score, 'best_v': best_v, 'best_w': best_w, 'best_prog': best_progress}
        if best_traj is None:
            return 0.0, 0.0, all_trajs, None, info
        return best_v, best_w, all_trajs, best_traj, info
    
    def publish_smoothed(self, cmd: Twist):     # exponential smoothing and limits before publishing cmd_vel
        sm = Twist()
        sm.linear.x = self.smoothing_ratio * cmd.linear.x + (1.0 - self.smoothing_ratio) * self.prev_cmd.linear.x
        sm.angular.z = self.smoothing_ratio * cmd.angular.z + (1.0 - self.smoothing_ratio) * self.prev_cmd.angular.z
        # sm.linear.x = max(0.0, min(V_PUB_MAX, sm.linear.x))
        sm.angular.z = max(-self.W_MAX, min(self.W_MAX, sm.angular.z))
        self.cmd_pub.publish(sm)
        self.prev_cmd = sm
        # self.get_logger().info(f"Published cmd_vel: v={sm.linear.x:.2f}, w={sm.angular.z:.2f}")

    def hold_position(self):
            hold_msg = Twist()
            hold_msg.linear.x = 0.0
            hold_msg.angular.z = 0.0
            self.cmd_pub.publish(hold_msg)
            self.get_logger().info('Shape detected. Waiting for 2 seconds...')
            time.sleep(2.0)
            self.get_logger().info('Done waiting...Resuming navigation.')
            self.flag_stop=False

    def stop_robot(self):
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0
        self.cmd_pub.publish(stop_msg)
            
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
    node=ebotNav3B()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown() 

if __name__=='__main__':
    main()