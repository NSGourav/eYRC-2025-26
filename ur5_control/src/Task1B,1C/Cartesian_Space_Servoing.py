#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
import numpy as np
import tf_transformations  # pip install transforms3d or tf-transformations

class EndEffectorServoNode(Node):
    def __init__(self):
        super().__init__('ee_servo_node')

        # Publisher: twist commands for MoveIt Servo
        self.pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)

        # Subscriber: get current end-effector pose
        self.sub = self.create_subscription(PoseStamped, '/tcp_pose_raw', self.pose_callback, 10)

        # --- List of waypoints (pos + ori) ---
        self.waypoints = [
            # (np.array([-0.214, -0.532, 0.557]), np.array([0.707, 0.028, 0.034, 0.707])),
            # (np.array([ -0.806, 0.010, 0.182]), np.array([-0.684, 0.726, 0.05, 0.008])),
            (np.array([-0.159, 0.501, 0.415]), np.array([0.029, 0.997, 0.045,0.033])),
            # (np.array([ -0.806, 0.010, 0.182]), np.array([-0.684, 0.726, 0.05, 0.008]))
        ]

        self.goal_idx = 0  # Start with first waypoint
        self.goal_position, self.goal_orientation = self.waypoints[self.goal_idx]

        self.current_position = None
        self.current_orientation = None
        self.timer = self.create_timer(0.1, self.control_loop)

    def pose_callback(self, msg):
        """Update current pose of the EE"""
        self.current_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        self.current_orientation = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])

    def control_loop(self):
        if self.current_position is None or self.current_orientation is None:
            return

        twist = Twist()

        # --- Position Control ---
        position_error = self.goal_position - self.current_position
        pos_dist = np.linalg.norm(position_error)

        if pos_dist > 0.01:  # 1cm tolerance
            v = 1.0 * position_error   # proportional control
            twist.linear.x = float(v[0]) 
            twist.linear.y = float(v[1])
            twist.linear.z = float(v[2])
            # --- Clamp linear velocity ---
            max_lin_vel = 0.1  # m/s
            v = np.clip(v, -max_lin_vel, max_lin_vel)
            twist.linear.x, twist.linear.y, twist.linear.z = v.tolist()

        # --- Orientation Control ---
        # Compute quaternion error: q_err = q_goal * q_current_inverse
        q_goal = self.goal_orientation
        q_curr = self.current_orientation
        q_curr_inv = tf_transformations.quaternion_inverse(q_curr)
        q_err = tf_transformations.quaternion_multiply(q_goal, q_curr_inv)
        q_err = np.array(q_err)

        if q_err[3] < 0:
            q_err = -q_err

        # Convert error quaternion to axis-angle
        angle = 2 * np.arccos(np.clip(q_err[3], -1.0, 1.0))
        if angle > 1e-3:  # if orientation error is significant
            vec = np.array(q_err[:3])   # convert to numpy array
            axis = vec / (np.linalg.norm(vec) + 1e-9)
            omega = 1.0 * angle * axis  # proportional gain
            twist.angular.x = float(omega[0])
            twist.angular.y = float(omega[1])
            twist.angular.z = float(omega[2])
            # --- Clamp angular velocity ---
            max_ang_vel = 0.5  # rad/s
            omega = np.clip(omega, -max_ang_vel, max_ang_vel)
            twist.angular.x, twist.angular.y, twist.angular.z = omega.tolist()

        # Publish twist
        self.pub.publish(twist)
        self.get_logger().info(f"EE pos_err={position_error}, angle_err={angle:.3f}")

        # Check if current goal is reached
        if pos_dist < 0.01 and angle < 0.05:  # both position + orientation tolerances
            self.get_logger().info(f"Reached waypoint {self.goal_idx+1}")
            self.goal_idx += 1
            if self.goal_idx < len(self.waypoints):
                self.goal_position, self.goal_orientation = self.waypoints[self.goal_idx]
                self.get_logger().info(f"Moving to next waypoint {self.goal_idx+1}")
            else:
                self.get_logger().info("All waypoints reached!")
                twist = Twist()  # stop
                self.pub.publish(twist)
                return

    # def control_loop(self):
        if self.current_position is None or self.current_orientation is None:
            return
        twist = Twist()
        roll,pitch,yaw=tf_transformations.euler_from_quaternion(self.current_orientation)
        # if abs(yaw-0.0)>0.1:
        #     twist.angular.z=-0.2
        # else:
        #     twist.angular.z=0.0
        # print("Pitch:",pitch)
        # if abs(pitch-0.0)> +1.57:
        #     twist.angular.y=-0.2
        # else:
        #     twist.angular.y=0.0
        # (np.array([-0.159, 0.501, 0.415]), np.array([0.029, 0.997, 0.045,0.033])),

        if abs(self.current_position[2]-0.415)>0.01:
            twist.linear.z=0.1
            
        else:   
            twist.linear.z=0.0

        if abs(self.current_position[1]-0.501)>0.01:
            twist.linear.y=0.1
        else:
            twist.linear.y=0.0
        
        if abs(self.current_position[0]+0.159)>0.01:
            twist.linear.x=-0.1
        else:
            twist.linear.x=0.0

        # if abs(self.current_position[2]-0.657)>0.01:
        #     twist.linear.z=0.1
            
        # else:   
        #     twist.linear.z=0.0

        # if abs(self.current_position[1]+0.482)>0.01:
        #     twist.linear.y=-0.1
        # else:
        #     twist.linear.y=0.0
        
        # if abs(self.current_position[0]+0.15)>0.01:
        #     twist.linear.x=-0.1
        # else:
        #     twist.linear.x=0.0
        # twist.angular.x = 0.0
        # twist.angular.y = 0.0
        # twist.angular.z = 0.0
        # twist.linear.x =  0.0
        # twist.linear.y = 0.0
        # twist.linear.z = 0.0
        self.pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = EndEffectorServoNode()
    rclpy.spin(node)

if __name__ == "__main__":
    main()
