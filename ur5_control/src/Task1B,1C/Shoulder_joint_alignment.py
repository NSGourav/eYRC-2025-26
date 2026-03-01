#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
import tf_transformations  # pip install transforms3d or tf-transformations

class EndEffectorServoNode(Node):
    def __init__(self):
        super().__init__('ee_servo_node')

        # Publisher: joint commands for shoulder alignment
        self.joint_pub = self.create_publisher(Float64MultiArray, '/delta_joint_cmds', 10)
        
        # Publisher: twist commands for position/orientation control
        self.twist_pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)

        # Subscriber: get current end-effector pose
        self.pose_sub = self.create_subscription(PoseStamped, '/tcp_pose_raw', self.pose_callback, 10)
        
        # Subscriber: get current joint states
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)

        # --- List of waypoints (pos + ori) ---
        self.waypoints = [
            (np.array([-0.159, 0.501, 0.415]), np.array([0.029, 0.997, 0.045, 0.033])),
        ]

        self.goal_idx = 0  # Start with first waypoint
        self.goal_position, self.goal_orientation = self.waypoints[self.goal_idx]

        # Shoulder link position in base frame
        self.shoulder_position = np.array([0.0, 0.0, 0.089])

        self.current_position = None
        self.current_orientation = None
        self.current_shoulder_angle = None
        self.shoulder_aligned = False  # Track if shoulder is aligned
        
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

    def joint_callback(self, msg):
        """Update current joint states"""
        try:
            shoulder_idx = msg.name.index('shoulder_pan_joint')
            self.current_shoulder_angle = msg.position[shoulder_idx] + 3.14159  # Adjust for UR5 shoulder offset
        except ValueError:
            self.get_logger().warn("shoulder_pan_joint not found in joint_states")

    def control_loop(self):
        if self.current_position is None or self.current_orientation is None or self.current_shoulder_angle is None:
            return

        # --- Phase 1: Align Shoulder toward waypoint using joint control ---
        if not self.shoulder_aligned:
            # Calculate desired shoulder angle from shoulder to waypoint in X-Y plane
            direction_to_goal = self.goal_position[:2] - self.shoulder_position[:2]
            desired_shoulder_angle = np.arctan2(direction_to_goal[1], direction_to_goal[0])
            
            # Calculate angle error
            angle_error = desired_shoulder_angle - self.current_shoulder_angle
            # Normalize to [-pi, pi]
            angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))

            # angle_error = desired_shoulder_angle - self.current_shoulder_angle
            # # if error magnitude is > 180 deg, force shortest direction
            # if angle_error > np.pi:
            #     angle_error -= 2 * np.pi
            # elif angle_error < -np.pi:
            #     angle_error += 2 * np.pi

            
            if abs(angle_error) > 0.05:  # 0.05 rad (~3 degrees) tolerance
                # Create joint velocity command - array of 6 joint velocities
                joint_cmd = Float64MultiArray()
                
                # Proportional control for shoulder velocity
                velocity = 100.5 * angle_error  # proportional gain
                max_vel = 0.7  # rad/s
                velocity = np.clip(velocity, -max_vel, max_vel)
                
                # Only move shoulder_pan_joint (index 0), others are 0
                joint_cmd.data = [float(velocity), 0.0, 0.0, 0.0, 0.0, 0.0]
                
                self.joint_pub.publish(joint_cmd)
                
                self.get_logger().info(f"Aligning shoulder: angle_err={np.degrees(angle_error):.2f} deg, "
                                     f"current={np.degrees(self.current_shoulder_angle):.2f} deg, "
                                     f"desired={np.degrees(desired_shoulder_angle):.2f} deg")
            else:
                # Shoulder aligned - stop all joint motion
                joint_cmd = Float64MultiArray()
                joint_cmd.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                self.joint_pub.publish(joint_cmd)
                
                self.shoulder_aligned = True
                self.get_logger().info("Shoulder aligned! Now moving to waypoint...")
        
        # --- Phase 2: Move to position and orientation using twist commands ---
        else:
            twist = Twist()
            
            # Position Control
            position_error = self.goal_position - self.current_position
            pos_dist = np.linalg.norm(position_error)

            if pos_dist > 0.01:  # 1cm tolerance
                v = 1.0 * position_error   # proportional control
                max_lin_vel = 0.1  # m/s
                v = np.clip(v, -max_lin_vel, max_lin_vel)
                twist.linear.x = float(v[0])
                twist.linear.y = float(v[1])
                twist.linear.z = float(v[2])

            # Orientation Control
            q_goal = self.goal_orientation
            q_curr = self.current_orientation
            q_curr_inv = tf_transformations.quaternion_inverse(q_curr)
            q_err = tf_transformations.quaternion_multiply(q_goal, q_curr_inv)
            q_err = np.array(q_err)

            if q_err[3] < 0:
                q_err = -q_err

            angle = 2 * np.arccos(np.clip(q_err[3], -1.0, 1.0))
            if angle > 0.05:  # orientation tolerance
                vec = np.array(q_err[:3])
                axis = vec / (np.linalg.norm(vec) + 1e-9)
                omega = 1.0 * angle * axis
                max_ang_vel = 0.5  # rad/s
                omega = np.clip(omega, -max_ang_vel, max_ang_vel)
                twist.angular.x = float(omega[0])
                twist.angular.y = float(omega[1])
                twist.angular.z = float(omega[2])

            self.twist_pub.publish(twist)
            self.get_logger().info(f"Moving: pos_err={pos_dist:.3f} m, angle_err={np.degrees(angle):.2f} deg")

            # Check if waypoint is reached
            if pos_dist < 0.01 and angle < 0.05:
                self.get_logger().info(f"Reached waypoint {self.goal_idx+1}")
                self.goal_idx += 1
                if self.goal_idx < len(self.waypoints):
                    self.goal_position, self.goal_orientation = self.waypoints[self.goal_idx]
                    self.shoulder_aligned = False  # Reset for next waypoint
                    self.get_logger().info(f"Moving to next waypoint {self.goal_idx+1}")
                else:
                    self.get_logger().info("All waypoints reached!")
                    twist = Twist()  # stop
                    self.twist_pub.publish(twist)
                    return


def main(args=None):
    rclpy.init(args=args)
    node = EndEffectorServoNode()
    rclpy.spin(node)

if __name__ == "__main__":
    main()