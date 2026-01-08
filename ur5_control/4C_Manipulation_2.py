#!/usr/bin/env python3

# Team ID:          [ eYRC#5076 ]
# Author List:      [ Nagulapalli Shavaneeth Gourav, Pradeep J, Anand Vardhan, Raj Mohammad ]
# Filename:         Task4C_manipulation.py
# Functions:        [ pose_callback, lookup_and_store_tf, goal_pose_nav, gripper_service,
#                     bad_fruit_sequence, execute_sequence ]
# Nodes:            Publishing Topics  - [ /delta_twist_cmds ]
#                   Subscribing Topics - [ /tcp_pose_raw ]

import rclpy
from rclpy.node import Node
import tf2_ros
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped
from linkattacher_msgs.srv import AttachLink, DetachLink
from std_srvs.srv import Trigger
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor


class ArmController(Node):
    def __init__(self):
        super().__init__("arm_controller_node")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.reentrant_callback_group = ReentrantCallbackGroup()

        self.velocity_publisher = self.create_publisher(Twist, "/delta_twist_cmds", 10)

        self.pose_subscriber = self.create_subscription(PoseStamped, '/tcp_pose_raw', self.pose_callback, 10,callback_group=self.reentrant_callback_group)

        self.attach_client = self.create_client(AttachLink, '/attach_link')
        self.detach_client = self.create_client(DetachLink, '/detach_link')

        self.pick_place_service = self.create_service(Trigger, '/pick_and_place', self.execute_sequence,callback_group=self.reentrant_callback_group)
        self.pick_bad_fruits_service = self.create_service(Trigger,'/pick_bad_fruits',self.bad_fruit_sequence,callback_group=self.reentrant_callback_group)

        self.control_rate = self.create_rate(10, self.get_clock())  # 10 Hz

        self.current_position = None
        self.current_orientation = None

        self.team_id = "5076"

        self.drop_location_pose = np.array([-0.8, 0.01, 0.3, 0.7, -0.7, 0.0, 0.0])
        self.bad_fruit_waypoint =np.array( [-0.4, 0.1, 0.4, 0.7, -0.7, 0.0, 0.0])
        self.home =np.array( [0.17, 0.0, 0.35, 0.0, 1.0, 0.0, 0.0 ])

        self.position_tolerance = 0.025     # 2.5 cm
        self.orientation_tolerance = 0.1  # ~10 degrees in radians

        self.kp_position = 0.5
        self.kp_orientation = 2.0
        self.prev_error = np.array([0.0, 0.0, 0.0])
        self.kd = 0.1
        self.prev_ori_error = np.array([0.0, 0.0, 0.0])
        self.ori_kd = 0.05

        self.max_linear_velocity = 0.1          # m/s
        self.max_angular_velocity = 0.5        # rad/s

        self.ebot_z_offset = 0.2                # meters
        self.bad_fruit_z_offset = 0.02          # meters
        self.lift_height = 0.1                  # meters

        self.tf_lookup_retries = 5
        self.tf_lookup_retry_delay = 0.5  # seconds

        self.is_fertilizer_task_active = False
        self.is_bad_fruit_task_active = False

        self.get_logger().info("Arm Controller Node Initialized")

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

    def lookup_and_store_tf(self, frame_id):

        for attempt in range(self.tf_lookup_retries):
            try:
                transform = self.tf_buffer.lookup_transform(
                    'base_link',
                    frame_id,
                    rclpy.time.Time()
                )
                
                pose = [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w,
                ]
                
                self.get_logger().info(f'TF found for {frame_id}')
                return pose
                
            except tf2_ros.TransformException as ex:
                self.get_logger().warn(
                    f'TF lookup failed for {frame_id} (attempt {attempt + 1}/{self.tf_lookup_retries}): {ex}'
                )
                if attempt < self.tf_lookup_retries - 1:
                    self.control_rate.sleep()
        
        self.get_logger().error(f'Failed to lookup TF for {frame_id} after {self.tf_lookup_retries} attempts')
        return None

    def goal_pose_nav(self, target_pose):

        # Wait for pose data
        while self.current_position is None or self.current_orientation is None:
            self.get_logger().warn('Waiting for pose data from /tcp_pose_raw...')
            self.control_rate.sleep()

        # Reset derivative tracking
        self.prev_error = np.zeros(3)
        self.prev_ori_error = np.zeros(3)

        target_position = np.array(target_pose[:3])
        target_quaternion = np.array(target_pose[3:7])
        target_quaternion = target_quaternion / np.linalg.norm(target_quaternion)

        self.get_logger().info(
            f'→ Navigating to: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]'
        )
        is_position_reached = False
        is_orientation_reached = False
        # Control loop
        while rclpy.ok():
            # Normalize current orientation
            current_quaternion = self.current_orientation / np.linalg.norm(self.current_orientation)


            position_error = target_position - self.current_position
            position_error_norm = np.linalg.norm(position_error)

            # Calculate derivative of position error
            position_error_derivative = (position_error - self.prev_error) / (1.0 / 10.0)  # dt = 0.1s at 10Hz
            self.prev_error = position_error.copy()

            # P control for position
            linear_velocity = self.kp_position * position_error + self.kd * position_error_derivative
            linear_velocity = np.clip(linear_velocity, -self.max_linear_velocity, self.max_linear_velocity)

            # Quaternion error: q_error = q_target * q_current_conjugate
            current_quat_conjugate = np.array([
                -current_quaternion[0],
                -current_quaternion[1],
                -current_quaternion[2],
                current_quaternion[3]
            ])

            # Quaternion multiplication
            x1, y1, z1, w1 = target_quaternion
            x2, y2, z2, w2 = current_quat_conjugate

            q_error = np.array([
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
                w1*w2 - x1*x2 - y1*y2 - z1*z2
            ])

            # Ensure shortest path
            if q_error[3] < 0:
                q_error = -q_error

            # Extract angle
            angle = 2.0 * np.arccos(np.clip(q_error[3], -1.0, 1.0))

            # Extract axis-angle for angular velocity
            if angle < 1e-6:
                orientation_error_axis_angle = np.zeros(3)
            else:
                sin_half_angle = np.sin(angle / 2.0)
                if abs(sin_half_angle) < 1e-6:
                    orientation_error_axis_angle = np.zeros(3)
                else:
                    axis = q_error[0:3] / sin_half_angle
                    orientation_error_axis_angle = axis * angle

            orientation_error_norm = angle

            # Calculate derivative of orientation error
            orientation_error_derivative = (orientation_error_axis_angle - self.prev_ori_error) / (1.0 / 10.0)
            self.prev_ori_error = orientation_error_axis_angle.copy()

            # P control for orientation
            angular_velocity = self.kp_orientation * orientation_error_axis_angle + self.ori_kd * orientation_error_derivative
            angular_velocity = np.clip(angular_velocity, -self.max_angular_velocity, self.max_angular_velocity)

            self.get_logger().info(f'Position error: {position_error}m | 'f'Orientation error: {orientation_error_norm:.4f}rad')

            twist_command = Twist()

            # Control orientation
            if orientation_error_norm > self.orientation_tolerance and is_orientation_reached == False:
                twist_command.angular.x = angular_velocity[0]
                twist_command.angular.y = angular_velocity[1]
                twist_command.angular.z = angular_velocity[2]
            elif is_orientation_reached == False:
                is_orientation_reached = True
                self.get_logger().info("✓ Orientation reached")

            # Control position
            if position_error_norm > self.position_tolerance and is_position_reached == False:
                twist_command.linear.x = linear_velocity[0]
                twist_command.linear.y = linear_velocity[1]
                twist_command.linear.z = linear_velocity[2]
            elif is_position_reached == False:
                is_position_reached = True
                self.get_logger().info("✓ Position reached")

            self.velocity_publisher.publish(twist_command)
            self.control_rate.sleep()

            # Check if both goals reached
            if is_position_reached and is_orientation_reached:
                # Stop the robot
                stop_twist = Twist()
                self.velocity_publisher.publish(stop_twist)
                self.get_logger().info("✓ Goal pose reached!")
                return True

        # return False

    def gripper_service(self, command, model_name):

        if command not in ['attach', 'detach']:
            self.get_logger().error('Invalid command: use "attach" or "detach"')
            return False

        # Select service client
        if command == 'attach':
            client = self.attach_client
            service_type = AttachLink
        else:
            client = self.detach_client
            service_type = DetachLink

        # Create request
        request = service_type.Request()
        request.model1_name = model_name
        request.link1_name = 'body'
        request.model2_name = 'ur5'
        request.link2_name = 'wrist_3_link'

        # Wait for service
        if not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(f'{command} service not available')
            return False

        # Call service
        future = client.call_async(request)
        # rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        # if future.result() is not None:
        #     self.get_logger().info(f'✓ {command} successful for {model_name}')
        #     return True
        # else:
        #     self.get_logger().error(f'✗ {command} service call failed for {model_name}')
        #     return False

    def bad_fruit_sequence(self, request, response):

        self.is_bad_fruit_task_active = True

        # Frame names for bad fruits
        bad_fruit_frames = [
            f'{self.team_id}_bad_fruit_0',
            f'{self.team_id}_bad_fruit_1',
            f'{self.team_id}_bad_fruit_2'
        ]
        
        print("hi")
        # Process each bad fruit
        for i in range(3):
            self.get_logger().info(f"\n→ Processing bad fruit {i}...")

            fruit_pose = self.lookup_and_store_tf(bad_fruit_frames[i])
            if fruit_pose is None:
                self.get_logger().error(f'Failed to lookup {bad_fruit_frames[i]}, continuing to next...')
                continue

            print("hello")

            # Apply z-offset
            fruit_pose[2] += self.bad_fruit_z_offset

            if not self.goal_pose_nav(fruit_pose):
                self.get_logger().error(f'Failed to navigate to {bad_fruit_frames[i]}, continuing to next...')
                continue

            if not self.gripper_service("attach", "bad_fruit"):
                self.get_logger().error(f'Failed to attach {bad_fruit_frames[i]}, continuing to next...')
                continue

            lift_pose = self.current_position.copy()
            lift_pose[2] += self.lift_height
            lift_pose_full = np.concatenate([lift_pose, self.current_orientation])

            self.get_logger().info(f'→ Lifting fruit by {self.lift_height}m...')

            if not self.goal_pose_nav(lift_pose_full):
                self.get_logger().error('Failed to lift fruit, continuing to next...')
                continue

            if not self.goal_pose_nav(self.drop_location_pose):
                self.get_logger().error('Failed to navigate to drop location, continuing to next...')
                continue

            if not self.gripper_service("detach", "bad_fruit"):
                self.get_logger().error(f'Failed to detach {bad_fruit_frames[i]}, continuing to next...')
                continue

            self.get_logger().info(f'✓ Bad fruit {i} processed successfully!')

        self.is_bad_fruit_task_active = False

        response.success = True
        response.message = "Bad fruit pick and place completed"
        return response

    def execute_sequence(self, request, response):

        self.is_fertilizer_task_active = True

        self.get_logger().info("\n→ Starting Fertilizer Sequence...")

        # Lookup fertilizer
        fertilizer_frame = f'{self.team_id}_fertiliser_can'
        fertilizer_pose = self.lookup_and_store_tf(fertilizer_frame)
        if fertilizer_pose is None:
            self.get_logger().error('Failed to lookup fertilizer, aborting task')
            response.success = False
            response.message = "Failed to lookup fertilizer"
            return response

        # Apply offset
        fertilizer_pose[1] += 0.02

        # Navigate to fertilizer
        if not self.goal_pose_nav(fertilizer_pose):
            self.get_logger().error('Failed to navigate to fertilizer, aborting task')
            response.success = False
            response.message = "Failed to navigate to fertilizer"
            return response

        # Attach fertilizer
        self.gripper_service("attach", "fertiliser_can")
            # self.get_logger().error('Failed to attach fertilizer, aborting task')
            # response.success = False
            # response.message = "Failed to attach fertilizer"
            # return response
        
        lift_fertilizer_pose = fertilizer_pose.copy()  # Make a copy
        lift_fertilizer_pose[1] += 0.2
        lift_fertilizer_pose[2] += 0.06
        
        if not self.goal_pose_nav(lift_fertilizer_pose):
            self.get_logger().error('Failed to lift fertilizer out of rack')
            response.success = False
            response.message = "Failed to lift fertilizer"
            return response
        
        if not self.goal_pose_nav(self.drop_location_pose):
            self.get_logger().error('Failed to navigate to drop location, continuing to next...')
            response.success = False
            response.message = "Failed to attach fertilizer"
            return response

        # # Lookup ebot
        # # ebot_frame = f'{self.team_id}_ebot_6'
        # # ebot_pose = self.lookup_and_store_tf(ebot_frame)
        # # if ebot_pose is None:
        # #     self.get_logger().error('Failed to lookup ebot, aborting task')
        # #     response.success = False
        # #     response.message = "Failed to lookup ebot"
        # #     return response

        # # # Apply z-offset
        # # ebot_pose[2] += self.ebot_z_offset

        # # # Navigate to ebot
        # # if not self.goal_pose_nav(ebot_pose):
        # #     self.get_logger().error('Failed to navigate to ebot, aborting task')
        # #     response.success = False
        # #     response.message = "Failed to navigate to ebot"
        # #     return response

        # Detach fertilizer
        self.gripper_service("detach", "fertiliser_can")
        self.goal_pose_nav([0.1, -0.2, 0.5, 0.7, -0.7, 0.0, 0.0])
        self.goal_pose_nav([0.1, 0.2, 0.5, 0.7, -0.7, 0.0, 0.0])
        #     self.get_logger().error('Failed to detach fertilizer, aborting task')
        #     response.success = False
        #     response.message = "Failed to detach fertilizer"
        #     return response

        # if not self.goal_pose_nav(lift_fertilizer_pose):
        #     self.get_logger().error('Failed to navigate to lift_fertilizer_pose, aborting task')
        
        # if not self.goal_pose_nav(self.home):
        #     self.get_logger().error('Failed to navigate to home, aborting task')

        # if not self.goal_pose_nav(self.home):
        #     self.get_logger().error('Failed to navigate to home, aborting task')

        self.is_fertilizer_task_active = False
        self.get_logger().info("✓ Fertilizer Sequence Completed!")

        # # Create dummy request/response for bad_fruit_sequence
        dummy_request = Trigger.Request()
        bad_fruit_response = Trigger.Response()
        bad_fruit_response = self.bad_fruit_sequence(dummy_request, bad_fruit_response)

        # # Check if bad fruit sequence was successful
        # if not bad_fruit_response.success:
        #     self.get_logger().warn('Bad fruit sequence completed with errors')

        response.success = True
        response.message = "Full task sequence completed"
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