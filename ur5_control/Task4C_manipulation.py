#!/usr/bin/env python3

# Team ID:          [ eYRC#5076 ]
# Author List:      [ Nagulapalli Shavaneeth Gourav, Pradeep J, Anand Vardhan, Raj Mohammad ]
# Filename:         Task4C_manipulation.py
# Functions:        [ pose_callback, gripper_service, lookup_tf, bad_fruit_callback,
#                     goal_pose_nav, control_loop ]
# Nodes:            Publishing Topics  - [ /delta_twist_cmds ]
#                   Subscribing Topics - [ /tcp_pose_raw ]

import rclpy
from rclpy.node import Node
import tf2_ros
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped
from linkattacher_msgs.srv import AttachLink, DetachLink
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import tf_transformations
from std_srvs.srv import Trigger

# Test command: ros2 service call /pick_and_place std_srvs/srv/Trigger {}

class ArmController(Node):
    def __init__(self):
        super().__init__("arm_controller_node")

        # TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publisher: end-effector velocity commands
        self.velocity_publisher = self.create_publisher(Twist, "/delta_twist_cmds", 10)
        
        # Callback groups
        self.reentrant_callback_group = ReentrantCallbackGroup()
        
        # Subscriber: current end-effector pose
        self.pose_subscriber = self.create_subscription(PoseStamped, '/tcp_pose_raw', self.pose_callback, 10,callback_group=self.reentrant_callback_group)
        self.control_rate = self.create_rate(1.2, self.get_clock())

        # Service clients for gripper attach/detach
        self.attach_client = self.create_client(AttachLink, '/attach_link')
        self.detach_client = self.create_client(DetachLink, '/detach_link')

        # Services and timers
        self.pick_place_service = self.create_service(Trigger, '/pick_and_place', self.control_loop,callback_group=self.reentrant_callback_group)

        # Current pose tracking
        self.current_position = None
        self.current_orientation = None 
        self.current_rotation_matrix = None

        # Team configuration
        self.team_id = "5076"
        self.bad_fruit_model_names = ['bad_fruit', 'bad_fruit', 'bad_fruit']

        # Target poses storage
        self.fertiliser_pose = None
        self.ebot_pose = None
        self.bad_fruit_poses = [None] * 3

        # Control flags
        self.are_target_poses_found = False
        self.are_fruit_poses_found = False

        # Control tolerances
        self.position_tolerance = 0.02     # meters
        self.orientation_tolerance = 0.01  # radians
        
        # Velocity limits
        self.max_linear_velocity = 0.1    # m/s
        self.min_linear_velocity = 0.015  # m/s
        self.max_angular_velocity = 1.0   # rad/s

        # PD control gains
        self.kp_position = 1.5
        self.kp_orientation = 3.0

    def pose_callback(self, msg):
        """Updates current end-effector position and orientation"""
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

    def publish_arm_velocity(self, twist_command):
        """Publishes velocity command to arm"""
        self.velocity_publisher.publish(twist_command)

    def gripper_service(self, command: str, model_name: str):
        """Attach or detach objects to/from gripper"""
        services = {
            'attach': (self.attach_client, AttachLink),
            'detach': (self.detach_client, DetachLink)
        }

        if command not in services:
            self.get_logger().error('Invalid command: use "attach" or "detach"')
            return False

        client, service_type = services[command]

        request = service_type.Request()
        request.model1_name = model_name
        request.link1_name = 'body'
        request.model2_name = 'ur5'
        request.link2_name = 'wrist_3_link'

        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error(f'{command} service not available')
            return False

        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        
        if future.result() is not None:
            self.get_logger().info(f'→ {command} successful')
            return True
        else:
            self.get_logger().error(f'{command} service call failed')
            return False
    
    def lookup_tf(self, frame_id, timeout_sec=2.0):
        """Look up transform from base_link to specified frame"""
        start_time = self.get_clock().now()
        
        while (self.get_clock().now() - start_time).nanoseconds < timeout_sec * 1e9:
            try:
                transform = self.tf_buffer.lookup_transform(
                    'base_link',
                    frame_id,
                    rclpy.time.Time()
                )
                self.get_logger().info(f'TF found for {frame_id}')
                
                return [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w,
                ]
            except tf2_ros.TransformException:
                self.get_logger().debug(f'Waiting for TF: {frame_id}')
                self.control_rate.sleep()

        self.get_logger().error(f'Timeout waiting for TF: {frame_id}')
        return None
    
    def bad_fruit_callback(self):
        """Pick bad fruits and place them in drop zone"""

        self.are_fruit_poses_found = False

        # Frame names for bad fruits
        bad_fruit_frames = [
            f'{self.team_id}_bad_fruit_1',
            f'{self.team_id}_bad_fruit_2',
            f'{self.team_id}_bad_fruit_3'
        ]

        # Look up TF for all bad fruits
        while not self.are_fruit_poses_found:
            self.control_rate.sleep()
            try:
                all_found = True
                for i in range(3):
                    fruit_pose = self.lookup_tf(bad_fruit_frames[i])
                    if fruit_pose is None:
                        all_found = False
                        self.get_logger().debug(f'TF for {bad_fruit_frames[i]} not found, retrying...')
                        break
                    fruit_pose[0] += 0.02  # Apply x-offset
                    self.bad_fruit_poses[i] = fruit_pose

                if all_found:
                    self.are_fruit_poses_found = True
                    self.get_logger().info('TF lookup successful - fruit poses stored')

            except (tf2_ros.LookupException, tf2_ros.TransformException) as ex:
                self.get_logger().debug(f'TF error: {ex}, retrying...')

        # Define home and drop poses
        home_pose = [0.1, 0.4, 0.4, 0.7, -0.7, 0.0, 0.0]
        drop_pose = [-0.8, 0.01, 0.3, 0.7, -0.7, 0.0, 0.0]

        # Pick and place each bad fruit
        for i in range(3):
            fruit_pose = self.bad_fruit_poses[i]

            # Move to home position
            self.goal_pose_nav(home_pose)

            # Approach fruit from above
            fruit_pose[2] += 0.1
            self.goal_pose_nav(fruit_pose)

            # Descend to fruit
            fruit_pose[2] -= 0.08
            self.goal_pose_nav(fruit_pose)

            # Attach fruit to gripper
            self.gripper_service("attach", self.bad_fruit_model_names[i])

            # Lift fruit
            fruit_pose[2] += 0.10
            self.goal_pose_nav(fruit_pose)
            
            # Move to drop zone
            self.goal_pose_nav(drop_pose)

            # Release fruit
            self.gripper_service("detach", self.bad_fruit_model_names[i])

        # Return to home position
        self.goal_pose_nav(home_pose)
        self.is_bad_fruit_task_active = False

    def goal_pose_nav(self, target_pose):

        while self.current_position is None or self.current_orientation is None:
            self.get_logger().warn('Waiting for pose data from /tcp_pose_raw...')
            self.control_rate.sleep()
        
        target_position = np.array(target_pose[:3])
        target_quaternion = np.array(target_pose[3:7])
        target_rotation_matrix = tf_transformations.quaternion_matrix(target_quaternion)[:3, :3]

        # Unit z-axis vector
        unit_z_axis = np.array([0.0, 0.0, 1.0])
        
        # Desired z-axis direction in world frame
        desired_z_axis = target_rotation_matrix @ unit_z_axis

        # Goal flags
        is_position_reached = False
        is_orientation_reached = False

        while rclpy.ok():
            try:
                # Get current rotation matrix
                rotation_matrix_4x4 = tf_transformations.quaternion_matrix(self.current_orientation)
                self.current_rotation_matrix = rotation_matrix_4x4[:3, :3]

                # --- Position Control ---
                position_error_base = target_position - self.current_position
                position_error_ee = self.current_rotation_matrix.T @ position_error_base
                
                # Proportional control
                linear_velocity_ee = self.kp_position * position_error_ee
                
                # Apply velocity limits
                linear_velocity_ee = np.clip(linear_velocity_ee, -self.max_linear_velocity, self.max_linear_velocity)
                linear_velocity_ee = np.where(
                    np.abs(linear_velocity_ee) < self.min_linear_velocity,
                    self.min_linear_velocity * np.sign(linear_velocity_ee),
                    linear_velocity_ee
                )
                
                position_error_norm = np.linalg.norm(position_error_ee)

                # --- Orientation Control (Z-axis alignment) ---
                current_z_axis = self.current_rotation_matrix @ unit_z_axis
                
                # Cross product gives rotation error
                z_axis_error = np.cross(current_z_axis, desired_z_axis)
                orientation_error_norm = np.linalg.norm(z_axis_error)

                twist_command = Twist()

                # PRIORITY 1: Correct orientation FIRST
                if not is_orientation_reached:
                    if orientation_error_norm > self.orientation_tolerance:
                        angular_velocity_world = self.kp_orientation * z_axis_error
                        angular_velocity_body = self.current_rotation_matrix.T @ angular_velocity_world
                        angular_velocity_body = np.clip(angular_velocity_body, -self.max_angular_velocity, self.max_angular_velocity)
                        
                        twist_command.angular.x = angular_velocity_body[0]
                        twist_command.angular.y = angular_velocity_body[1]
                        twist_command.angular.z = angular_velocity_body[2]
                    else:
                        is_orientation_reached = True
                        self.get_logger().info("Orientation reached")
                
                # PRIORITY 2: Correct position ONLY after orientation is done
                elif not is_position_reached:
                    if position_error_norm > self.position_tolerance:
                        twist_command.linear.x = linear_velocity_ee[0]
                        twist_command.linear.y = linear_velocity_ee[1]
                        twist_command.linear.z = linear_velocity_ee[2]
                    else:
                        is_position_reached = True
                        self.get_logger().info("Position reached")

                # Publish velocity command
                self.publish_arm_velocity(twist_command)
                self.control_rate.sleep()

                # Check if both goals reached
                if is_position_reached and is_orientation_reached:
                    self.get_logger().info(f"Goal pose reached at target: {target_pose}")
                    return True
                
            except Exception as e:
                self.get_logger().error(f"Exception in goal_pose_nav: {e}")
                self.control_rate.sleep()

    def control_loop(self, request, response):
        """Main pick and place control loop - picks fertilizer and places on ebot"""
        
        self.are_target_poses_found = False

        # Look up fertilizer and ebot transforms
        while not self.are_target_poses_found:
            self.control_rate.sleep()
            try:
                self.fertiliser_pose = self.lookup_tf(f'{self.team_id}_fertilizer_1')
                self.ebot_pose = self.lookup_tf(f'{self.team_id}_ebot_6')
                
                if self.fertiliser_pose is not None and self.ebot_pose is not None:
                    self.are_target_poses_found = True
                    self.get_logger().info('TF lookup successful – poses stored')
                else:
                    self.get_logger().debug('TF lookup returned None, retrying...')

            except (tf2_ros.LookupException, tf2_ros.TransformException) as ex:
                self.get_logger().debug(f'TF error: {ex}, retrying...')
        
        # Pick fertilizer can
        self.fertiliser_pose[1] += 0.01
        self.goal_pose_nav(self.fertiliser_pose)
        self.gripper_service("attach", "fertiliser_can")
        
        # Lift fertilizer can
        self.fertiliser_pose[1] += 0.2
        self.goal_pose_nav(self.fertiliser_pose)
        
        # Move to ebot
        self.ebot_pose[2] += 0.2
        self.goal_pose_nav(self.ebot_pose)
        
        # Place fertilizer on ebot
        self.gripper_service("detach", "fertiliser_can")
        
        # Move away from ebot
        self.ebot_pose[0] -= 0.3
        self.goal_pose_nav(self.ebot_pose)
        
        # Prepare response
        response.success = True
        response.message = "Pick and place done"
        self.get_logger().info('Pick and place completed successfully')
        
        # Activate bad fruit picking task
        self.bad_fruit_callback()
        
        return response
    
def main():
    rclpy.init()
    arm_controller = ArmController()
    executor = MultiThreadedExecutor()
    executor.add_node(arm_controller)
    executor.spin()
    arm_controller.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()