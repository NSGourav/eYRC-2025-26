#!/usr/bin/env python3

# Team ID:          [ eYRC#5076 ]
# Author List:		[ Nagulapalli Shavaneeth Gourav, Pradeep J, Anand Vardhan, Raj Mohammad ]
# Filename:		    Task2B_bad_fruit_perception.py
# Functions:
#			        [ pose_callback, call_attach_service_async, call_detach_service_async, check_service_result,
#                        quaternion_multiply, quaternion_conjugate, compute_orientation_error_quaternion, get_target_transform, control_loop ]
# Nodes:		    Add your publishing and subscribing node
#			        Publishing Topics  - [ /delta_twist_cmds ]
#                   Subscribing Topics - [ /tcp_pose_raw]

import rclpy
from rclpy.node import Node
import tf2_ros
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped
from linkattacher_msgs.srv import AttachLink, DetachLink

class MarkerFollower(Node):
    def __init__(self):
        super().__init__("marker_follower")

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publisher: publish end-effector velocity commands
        self.cmd_pub = self.create_publisher(Twist, "/delta_twist_cmds", 10)
        
        # Subscriber: get current end-effector pose
        self.sub = self.create_subscription(PoseStamped, '/tcp_pose_raw', self.pose_callback, 10)

        # Service clients for attach/detach
        self.attach_client = self.create_client(AttachLink, '/attach_link')
        self.detach_client = self.create_client(DetachLink, '/detach_link')

        self.current_position = None
        self.current_orientation = None 
        self.timer = self.create_timer(0.1, self.control_loop)

        self.team_id = "5076"

        # Waypoint P3 - hardcoded
        self.waypoint_p3_pos = np.array([-0.806, 0.010, 0.382])
        self.waypoint_p3_quat = np.array([-0.684, 0.726, 0.05, 0.008])
        self.waypoint_p3_quat = self.waypoint_p3_quat / np.linalg.norm(self.waypoint_p3_quat)

        # Target sequence
        self.targets = [
            f'{self.team_id}_fertiliser_can',      # 0: attach fertiliser
            f'{self.team_id}_ebot_6',              # 1: detach fertiliser
            f'{self.team_id}_bad_fruit_0',         # 2: attach bad_fruit_0
            'waypoint_p3',                         # 3: detach bad_fruit_0
            f'{self.team_id}_bad_fruit_1',         # 4: attach bad_fruit_1
            'waypoint_p3',                         # 5: detach bad_fruit_1
            f'{self.team_id}_bad_fruit_2',         # 6: attach bad_fruit_2
            'waypoint_p3',                         # 7: detach bad_fruit_2
        ]
        self.current_index = 0
        self.startup_complete = False  # Flag to ensure all targets captured before starting

        self.lift_waypoint_pos = None  # Will be set dynamically after bad fruit attach
        self.lift_waypoint_quat = None
        self.needs_lift = False  # Flag to track if we need to do lift before waypoint

        # Initial transforms - captured once and never updated
        self.initial_transforms = {}  # Dictionary: {target_name: (position, quaternion)}

        self.pos_tol = 0.025  # 2.5 cm position tolerance
        self.ori_tol = 0.175  # ~10 degrees in radians
        
        # velocity gains
        self.vel_gain = 1.5  
        self.ang_vel_gain = 1.5   
        
        # PD terms for position
        self.prev_error = np.array([0.0, 0.0, 0.0])
        self.kd = 0.1   # Derivative gain
        
        # PD terms for orientation
        self.prev_ori_error = np.array([0.0, 0.0, 0.0])
        self.ori_kd = 0.05

        self.print_counter = 0
        self.print_interval = 10  # Print every 10 iterations (1 second at 0.1s timer)
        
        # Service call state tracking
        self.service_pending = False
        self.service_future = None
        self.service_type = None  # 'attach' or 'detach'

    def pose_callback(self, msg):
        """ Updates the current position and orientation of the end-effector """
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

    def call_attach_service_async(self, is_bad_fruit=False):
        """Call the attach_link service asynchronously"""

        # Check if service is available (with 0.1 second timeout)
        if not self.attach_client.wait_for_service(timeout_sec=0.1):
            self.get_logger().warn('Attach service not available')
            return False
        
        request = AttachLink.Request()
        
        # Set model and link names based on object type
        if is_bad_fruit:
            request.model1_name = 'bad_fruit'
            request.link1_name = 'body'
        else:
            request.model1_name = 'fertiliser_can'
            request.link1_name = 'body'
            
        # Gripper link
        request.model2_name = 'ur5'
        request.link2_name = 'wrist_3_link'
        
        # Make async service call
        self.service_future = self.attach_client.call_async(request)
        self.service_pending = True
        self.service_type = 'attach'
        self.get_logger().info(f'→ Calling attach service for {request.model1_name} (async)...')
        return True

    def call_detach_service_async(self, is_bad_fruit=False):
        """Call the detach_link service asynchronously"""

        # Check if service is available (with 0.1 second timeout)
        if not self.detach_client.wait_for_service(timeout_sec=0.1):
            self.get_logger().warn('Detach service not available')
            return False
        
        request = DetachLink.Request()
        
        # Set model and link names based on object type
        if is_bad_fruit:
            request.model1_name = 'bad_fruit'
            request.link1_name = 'body'
        else:
            request.model1_name = 'fertiliser_can'
            request.link1_name = 'body'
            
        # Gripper link
        request.model2_name = 'ur5'
        request.link2_name = 'wrist_3_link'
        
        # Make async service call
        self.service_future = self.detach_client.call_async(request)
        self.service_pending = True
        self.service_type = 'detach'
        self.get_logger().info(f'→ Calling detach service for {request.model1_name} (async)...')
        return True

    def check_service_result(self):
        """Check if pending service call has completed"""
        if not self.service_pending or self.service_future is None:
            return None
        
        if self.service_future.done():
            try:
                result = self.service_future.result()
                if result is not None:
                    self.get_logger().info(f'✓ Successfully {self.service_type}_fertiliser_can')
                    self.service_pending = False
                    self.service_future = None
                    return True # Successful
                else:
                    self.get_logger().error(f'Failed to {self.service_type}')
                    self.service_pending = False
                    self.service_future = None
                    return False # Failed
            except Exception as e:
                self.get_logger().error(f'Service call exception: {e}')
                self.service_pending = False
                self.service_future = None
                return False 
        
        return None  # Still pending

    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions [x, y, z, w]"""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])

    def quaternion_conjugate(self, q):
        """Return the conjugate of quaternion [x, y, z, w]"""
        return np.array([-q[0], -q[1], -q[2], q[3]])

    def compute_orientation_error_quaternion(self, target_quat, current_quat):
        """
        Compute orientation error using quaternions
        Returns angular velocity vector and angle error in radians
        """
        # Normalize quaternions
        target_quat = target_quat / np.linalg.norm(target_quat)
        current_quat = current_quat / np.linalg.norm(current_quat)
        
        # Compute quaternion error: q_error = q_target * q_current_inverse
        current_quat_inv = self.quaternion_conjugate(current_quat)
        q_error = self.quaternion_multiply(target_quat, current_quat_inv)
        
        # Ensure quaternion is in the shortest path (w should be positive)
        if q_error[3] < 0:
            q_error = -q_error
        
        # Extract angle from quaternion
        angle = 2.0 * np.arccos(np.clip(q_error[3], -1.0, 1.0))
        
        # Extract axis-angle representation for angular velocity
        if angle < 1e-6:  # Very small error
            return np.zeros(3), 0.0
        
        sin_half_angle = np.sin(angle / 2.0)
        if abs(sin_half_angle) < 1e-6:
            return np.zeros(3), 0.0
            
        axis = q_error[0:3] / sin_half_angle
        
        # Angular velocity vector (axis-angle representation)
        axis_angle = axis * angle
        
        return axis_angle, angle

    def get_target_transform(self, target):
        """
        Get initial transform for target. Returns the stored initial transform.
        Returns: (target_pos, target_quat) or (None, None) if not initialized
        """
        if target == 'waypoint_p3':
            return self.waypoint_p3_pos, self.waypoint_p3_quat

        if target == 'lift_waypoint':
            return self.lift_waypoint_pos, self.lift_waypoint_quat

        if target in self.initial_transforms:
            target_pos, target_quat = self.initial_transforms[target]
            return target_pos, target_quat
        else:
            return None, None

    def control_loop(self):
        # Check if any service call is pending and handle result
        if self.service_pending:
            service_result = self.check_service_result()
            if service_result is None:
                # Service still pending, keep robot stopped
                twist = Twist()
                self.cmd_pub.publish(twist)
                return
            elif service_result:

                # Check if we need to insert a lift waypoint
                if self.needs_lift and self.service_type == 'attach':
                    # Create lift waypoint at current position + 0.1m in z
                    self.lift_waypoint_pos = self.current_position.copy()
                    self.lift_waypoint_pos[2] += 0.1  # Lift 0.1m up
                    self.lift_waypoint_quat = self.current_orientation.copy()  # Keep same orientation
                    
                    print(f"\n→ Creating lift waypoint at: [{self.lift_waypoint_pos[0]:.3f}, {self.lift_waypoint_pos[1]:.3f}, {self.lift_waypoint_pos[2]:.3f}]")
                    
                    # Insert 'lift_waypoint' into targets at current_index + 1
                    self.targets.insert(self.current_index + 1, 'lift_waypoint')
                    self.needs_lift = False  # Reset flag
                
                # Service completed successfully, proceed to next target
                self.current_index += 1
                
                # Show info about next target if exists
                if self.current_index < len(self.targets):
                    next_target = self.targets[self.current_index]
                    print(f"\n→ Now moving to target: {next_target}")
                    if next_target in self.initial_transforms:
                        initial_pos, _ = self.initial_transforms[next_target]
                        print(f"  {next_target} initial position: [{initial_pos[0]:.3f}, {initial_pos[1]:.3f}, {initial_pos[2]:.3f}]")
                
                # Reset integral and derivative terms for new target
                self.prev_error = np.array([0.0, 0.0, 0.0])
                self.prev_ori_error = np.array([0.0, 0.0, 0.0])
            else:
                # Service failed, but still proceed to next target
                self.get_logger().warn('Service call failed, but continuing to next target')
                self.current_index += 1
        
        # Wait until we have current position data
        if self.current_position is None or self.current_orientation is None:
            self.get_logger().warn("Waiting for /tcp_pose_raw data...", throttle_duration_sec=1.0)
            return
        
        # Startup phase: Capture initial transforms for ALL targets before starting
        if not self.startup_complete:
            all_captured = True
            
            for target in self.targets:
                # Skip waypoint_p3 (hardcoded)
                if target == 'waypoint_p3':
                    continue
                
                # Skip if already captured
                if target in self.initial_transforms:
                    continue
                
                # Try to capture this target for the first time
                try:
                    tf = self.tf_buffer.lookup_transform("base_link", target, rclpy.time.Time())
                    target_pos = np.array([
                        tf.transform.translation.x,
                        tf.transform.translation.y,
                        tf.transform.translation.z
                    ])
                    target_quat = np.array([
                        tf.transform.rotation.x,
                        tf.transform.rotation.y,
                        tf.transform.rotation.z,
                        tf.transform.rotation.w
                    ])
                    
                    # Store initial transform (only once!)
                    self.initial_transforms[target] = (target_pos.copy(), target_quat.copy())
                    print(f"✓ Captured INITIAL transform for {target} at: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
                        
                except Exception as e:
                    all_captured = False
                    self.get_logger().warn(f"Waiting to see {target}...", throttle_duration_sec=1.0)
            
            # Count expected TF targets (excluding waypoint_p3)
            expected_tf_targets = len([t for t in self.targets if t != 'waypoint_p3'])
            
            # Check if all targets have been captured
            if all_captured and len(self.initial_transforms) == expected_tf_targets:
                self.startup_complete = True
                print("\n" + "="*60)
                print("All initial transforms captured! Starting movement...")
                print("="*60)
                
                # Print all initial transforms
                for target in self.targets:
                    if target == 'waypoint_p3':
                        # Print hardcoded waypoint
                        print(f"{target} [HARDCODED]: pos=[{self.waypoint_p3_pos[0]:.3f}, {self.waypoint_p3_pos[1]:.3f}, {self.waypoint_p3_pos[2]:.3f}], quat=[{self.waypoint_p3_quat[0]:.3f}, {self.waypoint_p3_quat[1]:.3f}, {self.waypoint_p3_quat[2]:.3f}, {self.waypoint_p3_quat[3]:.3f}]")
                    else:
                        # Print initial transform
                        pos, quat = self.initial_transforms[target]
                        print(f"{target} [INITIAL]: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], quat=[{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]")
                
                print("="*60 + "\n")
            else:
                # Don't move until all targets are captured
                twist = Twist()
                self.cmd_pub.publish(twist)
                return

        # Check if all targets completed
        if self.current_index >= len(self.targets):
            twist = Twist()  # Stop movement
            self.cmd_pub.publish(twist)
            self.get_logger().info("All targets reached!", throttle_duration_sec=1.0)
            return

        target = self.targets[self.current_index]

        # Get initial transform for current target
        target_pos, target_quat = self.get_target_transform(target)
        
        if target_pos is None:
            # Should never happen after startup, but handle gracefully
            twist = Twist()
            self.cmd_pub.publish(twist)
            self.get_logger().error(f"Initial transform for {target} not found!")
            return

        # Apply z-axis offset for can_6
        if target == f'{self.team_id}_ebot_6':
            target_pos = target_pos.copy()  # Make a copy to avoid modifying stored value
            target_pos[2] += 0.25  # Add 0.25 meters to z-axis

        # # Apply z-axis offset for bad fruits
        if target in [f'{self.team_id}_bad_fruit_0', f'{self.team_id}_bad_fruit_1', f'{self.team_id}_bad_fruit_2']:
            target_pos = target_pos.copy()  # Make a copy to avoid modifying stored value
            target_pos[2] += 0.02  # Add 0.05 meters to z-axis for bad fruits
        
        # Calculate position error
        pos_error = target_pos - self.current_position
        pos_error_norm = np.linalg.norm(pos_error)
        
        # Calculate position derivative term
        pos_error_derivative = (pos_error - self.prev_error) / 0.1
        self.prev_error = pos_error.copy()
        
        # Calculate orientation error using quaternions
        ori_error_axis_angle, ori_error_angle = self.compute_orientation_error_quaternion(
            target_quat, self.current_orientation
        )
        
        # Calculate orientation derivative term
        ori_error_derivative = (ori_error_axis_angle - self.prev_ori_error) / 0.1
        self.prev_ori_error = ori_error_axis_angle.copy()

        self.print_counter += 1
        if self.print_counter >= self.print_interval:
            self.print_counter = 0  # Reset counter
            print(f"\n[INITIAL TRANSFORM] Target: {target}")
            print(f"Target Position       -> x:{target_pos[0]:7.3f}, y:{target_pos[1]:7.3f}, z:{target_pos[2]:7.3f}")
            print(f"Current EE Position   -> x:{self.current_position[0]:7.3f}, y:{self.current_position[1]:7.3f}, z:{self.current_position[2]:7.3f}")
            print(f"Position Error        -> x:{pos_error[0]:7.3f}, y:{pos_error[1]:7.3f}, z:{pos_error[2]:7.3f}")
            print(f"Position Error Norm   -> {pos_error_norm:.4f} m")
            print(f"Orientation Error     -> x:{ori_error_axis_angle[0]:7.3f}, y:{ori_error_axis_angle[1]:7.3f}, z:{ori_error_axis_angle[2]:7.3f} rad")
            print(f"Orientation Error Angle-> {ori_error_angle:.4f} rad ({np.degrees(ori_error_angle):.2f}°)")

        # PID control for position
        twist = Twist()
        pos_control = (self.vel_gain * pos_error 
                      + self.kd * pos_error_derivative
                      )
        
        twist.linear.x = pos_control[0]
        twist.linear.y = pos_control[1]
        twist.linear.z = pos_control[2]
        
        # PID control for orientation (using axis-angle representation)
        ori_control = (self.ang_vel_gain * ori_error_axis_angle
                      + self.ori_kd * ori_error_derivative
                      )
        
        twist.angular.x = ori_control[0]
        twist.angular.y = ori_control[1]
        twist.angular.z = ori_control[2]

        self.cmd_pub.publish(twist)

        # Check if target reached - position < 1cm and orientation < 10° (0.175 rad)
        if pos_error_norm < self.pos_tol and ori_error_angle < self.ori_tol:
            print(f"\n✓ Target {target} reached (using initial transform)! Moving to next target...")
            print(f"  Target was at: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            print(f"  EE ended at:   [{self.current_position[0]:.3f}, {self.current_position[1]:.3f}, {self.current_position[2]:.3f}]")
            print(f"  Final position error: {pos_error_norm:.4f} m")
            print(f"  Final orientation error: {ori_error_angle:.4f} rad ({np.degrees(ori_error_angle):.2f}°)")

            # Stop the robot before calling service
            twist = Twist()
            self.cmd_pub.publish(twist)

            # Call appropriate service based on which target was reached
            if target == f'{self.team_id}_fertiliser_can':
                print("\n→ Calling attach service for fertiliser_can...")
                self.call_attach_service_async(is_bad_fruit=False)
            elif target == f'{self.team_id}_ebot_6':
                print("\n→ Calling detach service for fertiliser_can...")
                self.call_detach_service_async(is_bad_fruit=False)
            elif target in [f'{self.team_id}_bad_fruit_0', f'{self.team_id}_bad_fruit_1', f'{self.team_id}_bad_fruit_2']:
                print(f"\n→ Calling attach service for {target}...")
                self.call_attach_service_async(is_bad_fruit=True)
                # Set flag to create lift waypoint after attach completes
                self.needs_lift = True
            elif target == 'lift_waypoint':
                # Reached lift waypoint, now proceed to waypoint_p3
                print(f"\n✓ Lift complete! Proceeding to waypoint_p3...")
                self.current_index += 1
                self.error_integral = np.array([0.0, 0.0, 0.0])
                self.prev_error = np.array([0.0, 0.0, 0.0])
                self.ori_error_integral = np.array([0.0, 0.0, 0.0])
                self.prev_ori_error = np.array([0.0, 0.0, 0.0])
            elif target == 'waypoint_p3':
                print(f"\n→ Calling detach service at waypoint P3...")
                self.call_detach_service_async(is_bad_fruit=True)

def main():
    rclpy.init()
    node = MarkerFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()