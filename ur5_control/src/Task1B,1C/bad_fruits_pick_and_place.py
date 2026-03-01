#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PointStamped
import numpy as np
import tf_transformations  # pip install transforms3d or tf-transformations

class EndEffectorServoNode(Node):
    def __init__(self):
        super().__init__('ee_servo_node')

        # Publisher: twist commands for MoveIt Servo
        self.pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)

        # Subscriber: get current end-effector pose
        self.sub_pose = self.create_subscription(PoseStamped, '/tcp_pose_raw', self.pose_callback, 10)
        
        # Subscriber: get bad fruit positions
        self.sub_fruits = self.create_subscription(PointStamped, '/bad_fruit_positions', self.fruit_callback, 10)

        # Drop-off location (position + orientation)
        self.dropoff_position = np.array([-0.756, 0.010, 0.182])
        self.dropoff_orientation = np.array([-0.684, 0.726, 0.05, 0.008])

        # Store all bad fruit positions as they come in
        self.bad_fruits = []
        self.fruit_orientations = []  # Store orientations for each fruit
        
        # Current state
        self.current_fruit_idx = 0  # Which fruit we're targeting
        self.at_dropoff = False  # Track if we're going to dropoff or fruit
        self.collection_started = False  # Flag to start after receiving all fruits
        
        self.goal_position = None
        self.goal_orientation = None

        self.current_position = None
        self.current_orientation = None
        
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # Timer to wait for all fruit messages before starting
        self.startup_timer = self.create_timer(2.0, self.start_collection)

    def fruit_callback(self, msg):
        """Collect bad fruit positions"""
        fruit_pos = np.array([msg.point.x, msg.point.y, msg.point.z])
        
        # Check if this fruit is already in our list (based on frame_id)
        fruit_id = msg.header.frame_id
        exists = False
        for i, (stored_id, _) in enumerate(self.bad_fruits):
            if stored_id == fruit_id:
                exists = True
                break
        
        if not exists:
            self.bad_fruits.append((fruit_id, fruit_pos))
            # Use a default orientation for picking (you can customize this)
            default_orientation = np.array([-0.684, 0.726, 0.05, 0.008])
            self.fruit_orientations.append(default_orientation)
            self.get_logger().info(f"Received bad fruit: {fruit_id} at position {fruit_pos}")

    def start_collection(self):
        """Start collection after initial wait period"""
        if not self.collection_started and len(self.bad_fruits) > 0:
            self.collection_started = True
            self.get_logger().info(f"Starting collection of {len(self.bad_fruits)} bad fruits")
            # Set first goal to first bad fruit
            self.goal_position = self.bad_fruits[0][1]
            self.goal_orientation = self.fruit_orientations[0]
            self.at_dropoff = False
            self.startup_timer.cancel()

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
        
        if not self.collection_started or self.goal_position is None:
            return

        twist = Twist()

        # --- Position Control ---
        position_error = self.goal_position - self.current_position
        pos_dist = np.linalg.norm(position_error)

        if pos_dist > 0.01:  # 1cm tolerance
            v = 1.0 * position_error   # proportional control
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
            # --- Clamp angular velocity ---
            max_ang_vel = 0.5  # rad/s
            omega = np.clip(omega, -max_ang_vel, max_ang_vel)
            twist.angular.x, twist.angular.y, twist.angular.z = omega.tolist()

        # Publish twist
        self.pub.publish(twist)
        
        location_type = "dropoff" if self.at_dropoff else f"fruit_{self.current_fruit_idx}"
        self.get_logger().info(f"Target: {location_type}, pos_err={pos_dist:.4f}m, angle_err={angle:.3f}rad")

        # Check if current goal is reached
        if pos_dist < 0.11 and angle < 0.05:  # both position + orientation tolerances
            if self.at_dropoff:
                # Just finished dropping off, go to next fruit
                self.get_logger().info(f"Dropped off fruit {self.current_fruit_idx}")
                self.current_fruit_idx += 1
                
                if self.current_fruit_idx < len(self.bad_fruits):
                    # Go to next bad fruit
                    self.goal_position = self.bad_fruits[self.current_fruit_idx][1]
                    self.goal_orientation = self.fruit_orientations[self.current_fruit_idx]
                    self.at_dropoff = False
                    self.get_logger().info(f"Moving to bad fruit {self.current_fruit_idx}")
                else:
                    # All fruits collected!
                    self.get_logger().info("All bad fruits collected and dropped off!")
                    twist = Twist()  # stop
                    self.pub.publish(twist)
                    return
            else:
                # Just reached a fruit, now go to dropoff
                self.get_logger().info(f"Reached bad fruit {self.current_fruit_idx}")
                self.goal_position = self.dropoff_position
                self.goal_orientation = self.dropoff_orientation
                self.at_dropoff = True
                self.get_logger().info("Moving to dropoff location")


def main(args=None):
    rclpy.init(args=args)
    node = EndEffectorServoNode()
    rclpy.spin(node)

if __name__ == "__main__":
    main()