#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import tf2_ros
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped
from scipy.spatial.transform import Rotation as R
from linkattacher_msgs.srv import AttachLink, DetachLink

class MarkerFollower(Node):
    def __init__(self):
        super().__init__("marker_follower")

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.cmd_pub = self.create_publisher(Twist, "/delta_twist_cmds", 10)
        self.sub = self.create_subscription(PoseStamped, '/tcp_pose_raw', self.pose_callback, 10)

        # Service clients for attach/detach
        self.attach_client = self.create_client(AttachLink, '/attach_link')
        self.detach_client = self.create_client(DetachLink, '/detach_link')

        self.current_position = None
        self.current_orientation = None  # Store as quaternion [x, y, z, w]
        self.timer = self.create_timer(0.1, self.control_loop)

        self.targets = ["cam_3", "cam_6"]
        self.current_index = 0
        self.startup_complete = False  # Flag to ensure all targets cached before starting
    
        # Cache for storing last known transforms for each target
        self.cached_transforms = {}  # Dictionary: {target_name: (position, quaternion)}

        self.pos_tol = 0.1  # 10cm position tolerance
        self.ori_tol = 0.175  # ~10 degrees in radians (0.174533 rad)
        self.vel_gain = 0.1
        self.ang_vel_gain = 0.2
        
        # PID terms for position
        self.error_integral = np.array([0.0, 0.0, 0.0])
        self.prev_error = np.array([0.0, 0.0, 0.0])
        self.ki = 0.01  # Integral gain
        self.kd = 0.05  # Derivative gain
        
        # PID terms for orientation
        self.ori_error_integral = np.array([0.0, 0.0, 0.0])
        self.prev_ori_error = np.array([0.0, 0.0, 0.0])
        self.ori_ki = 0.005
        self.ori_kd = 0.02

        self.print_counter = 0
        self.print_interval = 10  # Print every 10 iterations (1 second at 0.1s timer)

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

    def call_attach_service(self):
        """Call the attach_link service"""
        if not self.attach_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Attach service not available')
            return False
        
        request = AttachLink.Request()
        request.model1_name = 'fertiliser_can'
        request.link1_name = 'body'
        request.model2_name = 'ur5'
        request.link2_name = 'wrist_3_link'
        
        future = self.attach_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        
        if future.result() is not None:
            self.get_logger().info('✓ Successfully attached fertiliser_can to ur5')
            return True
        else:
            self.get_logger().error('Failed to call attach service')
            return False

    def call_detach_service(self):
        """Call the detach_link service"""
        if not self.detach_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Detach service not available')
            return False
        
        request = DetachLink.Request()
        request.model1_name = 'fertiliser_can'
        request.link1_name = 'body'
        request.model2_name = 'ur5'
        request.link2_name = 'wrist_3_link'
        
        future = self.detach_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        
        if future.result() is not None:
            self.get_logger().info('✓ Successfully detached fertiliser_can from ur5')
            return True
        else:
            self.get_logger().error('Failed to call detach service')
            return False

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
        # For unit quaternion: angle = 2 * arccos(w)
        angle = 2.0 * np.arccos(np.clip(q_error[3], -1.0, 1.0))
        
        # Extract axis-angle representation for angular velocity
        if angle < 1e-6:  # Very small error
            return np.zeros(3), 0.0
        
        # For small angles: axis ≈ [x, y, z] / sin(angle/2)
        sin_half_angle = np.sin(angle / 2.0)
        if abs(sin_half_angle) < 1e-6:
            return np.zeros(3), 0.0
        
        axis = q_error[0:3] / sin_half_angle
        
        # Angular velocity vector (axis-angle representation)
        axis_angle = axis * angle
        
        return axis_angle, angle

    def get_target_transform(self, target):
        """
        Get transform for target. If live transform is available, use and cache it.
        If not available, use cached transform if it exists.
        Returns: (target_pos, target_quat, using_cached) or (None, None, False) if never seen
        """
        try:
            # Try to get live transform
            tf = self.tf_buffer.lookup_transform("base_link", target, rclpy.time.Time())
            
            # Extract position and orientation
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
            
            # Update cache with latest transform
            self.cached_transforms[target] = (target_pos.copy(), target_quat.copy())
            self.get_logger().debug(
                f"Updated cache for {target}: pos=[{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]"
            )
            
            return target_pos, target_quat, False  # Using live transform
            
        except Exception as e:
            # Transform not available, check cache
            if target in self.cached_transforms:
                target_pos, target_quat = self.cached_transforms[target]
                self.get_logger().warn(
                    f"Marker {target} not visible, using cached transform",
                    throttle_duration_sec=2.0
                )
                return target_pos, target_quat, True  # Using cached transform
            else:
                # Never seen this marker
                return None, None, False

    def control_loop(self):
        # Wait until we have current position data
        if self.current_position is None or self.current_orientation is None:
            self.get_logger().warn("Waiting for /tcp_pose_raw data...", throttle_duration_sec=1.0)
            return
        
        # Startup phase: Cache all targets before starting movement
        if not self.startup_complete:
            all_cached = True
            for target in self.targets:
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
                    
                    if target not in self.cached_transforms:
                        self.cached_transforms[target] = (target_pos.copy(), target_quat.copy())
                        print(f"✓ Cached {target} at position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
                    else:
                        # Update cache continuously during startup
                        self.cached_transforms[target] = (target_pos.copy(), target_quat.copy())
                        
                except Exception as e:
                    all_cached = False
                    self.get_logger().warn(f"Waiting to see {target}...", throttle_duration_sec=1.0)
            
            if all_cached and len(self.cached_transforms) == len(self.targets):
                self.startup_complete = True
                print("\n" + "="*60)
                print("All targets cached! Starting movement...")
                print("="*60)
                for target in self.targets:
                    pos, quat = self.cached_transforms[target]
                    print(f"{target}: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], quat=[{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]")
                print("="*60 + "\n")
            else:
                # Don't move until all targets are cached
                twist = Twist()
                self.cmd_pub.publish(twist)
                return
            
        if self.current_index >= len(self.targets):
            twist = Twist()  # Stop movement
            self.cmd_pub.publish(twist)
            self.get_logger().info("All targets reached!", throttle_duration_sec=1.0)
            return

        target = self.targets[self.current_index]

        # Get target transform (live or cached)
        target_pos, target_quat, using_cached = self.get_target_transform(target)
        
        if target_pos is None:
            # Marker has never been seen, wait/retry
            twist = Twist()  # Stop movement
            self.cmd_pub.publish(twist)
            self.get_logger().warn(
                f"Waiting to see marker {target} for the first time...",
                throttle_duration_sec=1.0
            )
            return

        # Calculate position error
        pos_error = target_pos - self.current_position
        pos_error_norm = np.linalg.norm(pos_error)
        
        # Update position integral term (with anti-windup)
        self.error_integral += pos_error * 0.1  # dt = 0.1
        self.error_integral = np.clip(self.error_integral, -1.0, 1.0)
        
        # Calculate position derivative term
        pos_error_derivative = (pos_error - self.prev_error) / 0.1
        self.prev_error = pos_error.copy()
        
        # Calculate orientation error using quaternions
        ori_error_axis_angle, ori_error_angle = self.compute_orientation_error_quaternion(
            target_quat, self.current_orientation
        )
        
        # Update orientation integral term (with anti-windup)
        self.ori_error_integral += ori_error_axis_angle * 0.1
        self.ori_error_integral = np.clip(self.ori_error_integral, -1.0, 1.0)
        
        # Calculate orientation derivative term
        ori_error_derivative = (ori_error_axis_angle - self.prev_ori_error) / 0.1
        self.prev_ori_error = ori_error_axis_angle.copy()

        self.print_counter += 1
        if self.print_counter >= self.print_interval:
            self.print_counter = 0  # Reset counter
            cache_status = "[CACHED]" if using_cached else "[LIVE]"
            print(f"\n{cache_status} Target: {target}")
            print(f"Cached targets: {list(self.cached_transforms.keys())}")
            print(f"Target Position       -> x:{target_pos[0]:7.3f}, y:{target_pos[1]:7.3f}, z:{target_pos[2]:7.3f}")
            print(f"Current EE Position   -> x:{self.current_position[0]:7.3f}, y:{self.current_position[1]:7.3f}, z:{self.current_position[2]:7.3f}")
            print(f"Position Error        -> x:{pos_error[0]:7.3f}, y:{pos_error[1]:7.3f}, z:{pos_error[2]:7.3f}")
            print(f"Position Error Norm   -> {pos_error_norm:.4f} m")
            print(f"Orientation Error     -> x:{ori_error_axis_angle[0]:7.3f}, y:{ori_error_axis_angle[1]:7.3f}, z:{ori_error_axis_angle[2]:7.3f} rad")
            print(f"Orientation Error Angle-> {ori_error_angle:.4f} rad ({np.degrees(ori_error_angle):.2f}°)")

        # PID control for position
        twist = Twist()
        pos_control = (self.vel_gain * pos_error 
                      # + self.ki * self.error_integral
                      # + self.kd * pos_error_derivative
                      )
        
        twist.linear.x = pos_control[0]
        twist.linear.y = pos_control[1]
        twist.linear.z = pos_control[2]
        
        # PID control for orientation (using axis-angle representation)
        ori_control = (self.ang_vel_gain * ori_error_axis_angle
                      # + self.ori_ki * self.ori_error_integral
                      # + self.ori_kd * ori_error_derivative
                      )
        
        twist.angular.x = ori_control[0]
        twist.angular.y = ori_control[1]
        twist.angular.z = ori_control[2]

        self.cmd_pub.publish(twist)

        # Check if target reached - position < 10cm and orientation < 10° (0.175 rad)
        if pos_error_norm < self.pos_tol and ori_error_angle < self.ori_tol:
            cache_status = "cached" if using_cached else "live"
            print(f"\n✓ Target {target} reached (using {cache_status} transform)! Moving to next target...")
            print(f"  Target was at: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            print(f"  EE ended at:   [{self.current_position[0]:.3f}, {self.current_position[1]:.3f}, {self.current_position[2]:.3f}]")
            print(f"  Final position error: {pos_error_norm:.4f} m")
            print(f"  Final orientation error: {ori_error_angle:.4f} rad ({np.degrees(ori_error_angle):.2f}°)")

            # Call appropriate service based on which target was reached
            if target == "cam_3":
                print("\n→ Calling attach service for fertiliser_can...")
                self.call_attach_service()
            elif target == "cam_6":
                print("\n→ Calling detach service for fertiliser_can...")
                self.call_detach_service()

            self.current_index += 1
            
            # Show info about next target if exists
            if self.current_index < len(self.targets):
                next_target = self.targets[self.current_index]
                print(f"\n→ Now moving to target: {next_target}")
                if next_target in self.cached_transforms:
                    cached_pos, _ = self.cached_transforms[next_target]
                    print(f"  {next_target} cached at: [{cached_pos[0]:.3f}, {cached_pos[1]:.3f}, {cached_pos[2]:.3f}]")
                else:
                    print(f"  {next_target} not yet cached (will wait to see it)")
            # Reset integral and derivative terms for new target
            self.error_integral = np.array([0.0, 0.0, 0.0])
            self.prev_error = np.array([0.0, 0.0, 0.0])
            self.ori_error_integral = np.array([0.0, 0.0, 0.0])
            self.prev_ori_error = np.array([0.0, 0.0, 0.0])

def main():
    rclpy.init()
    node = MarkerFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()