#!/usr/bin/env python3

# Team ID:          [ eYRC#5076 ]
# Author List:		[ Nagulapalli Shavaneeth Gourav, Pradeep J, Anand Vardhan, Raj Mohammad ]
# Filename:		    Task5_manipulation.py
# Functions:
#			        []
# Nodes:		    Add your publishing and subscribing node
#			        Publishing Topics  - [ /delta_joint_cmds ]
#                   Subscribing Topics - [ /tcp_pose_raw, /net_wrench]

import rclpy
import tf2_ros
import time
import sys
import os
import PyKDL
import numpy as np
import tf_transformations
from std_msgs.msg import String
from rclpy.callback_groups import ReentrantCallbackGroup,MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import SetBool
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from ament_index_python.packages import get_package_share_directory
from urdf_parser_py.urdf import URDF
from control_msgs.msg import JointJog
from std_msgs.msg import Float32
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kdl_parser_py import treeFromUrdfModel

class ArmController(Node):
    def __init__(self):
        super().__init__("arm_controller_node")

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.rate = self.create_rate(200.0, self.get_clock())

        # Flags:
        self.service_flag = 0
        self.flag_force   = 0
        self.flag_fruit   = 0
        self.flag_fertilizer_drop     = 0
        self.flag_max_linear_velocity = 0

        # Publisher: publish joint velocity commands
        self.joint_velocity_publisher = self.create_publisher(JointJog, "/delta_joint_cmds", 10)

        # Subscribers:
        self.joint_state_subscriber = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.force_subscriber = self.create_subscription(Float32, '/net_wrench', self.force_callback, 10)

        # Services:
        self.magnet_client = self.create_client(SetBool, '/magnet')
        if self.service_flag:
            self.pick_place_service = self.create_service(SetBool,'pick_and_place',self.pick_and_place_callback, callback_group = MutuallyExclusiveCallbackGroup())
        else:
            self.auto_sequence_timer = self.create_timer(2.0, self.trigger_sequence_0, callback_group = MutuallyExclusiveCallbackGroup())

        self.fruit_cb_group = MutuallyExclusiveCallbackGroup()
        self.pose_sub = self.create_timer(0.5, self.pose_callback, callback_group = self.fruit_cb_group)
        self.callback_group = ReentrantCallbackGroup()
        self.bad_fruit_trigger = self.create_timer(0.5, self.bad_fruit_callback,callback_group = self.callback_group)
        self.drop_fertilizer_trigger = self.create_timer(0.5, self.drop_fertilizer_callback,callback_group = self.callback_group)

        # Load URDF and create KDL chain for Jacobian computation
        package_share_dir = get_package_share_directory('ur_simulation_gz')
        urdf_path = os.path.join(package_share_dir, 'ur5_arm.urdf')
        base_link = 'base_link'
        end_link = 'tool0'
        self.robot = URDF.from_xml_file(urdf_path)
        self.kdl_chain = self.load_kdl_chain(urdf_path, base_link, end_link)

        # Initialize positions and oreintations
        self.ebot_pose = None
        self.fertiliser_pose  = None
        self.joint_positions  = None
        self.current_position = None
        self.current_orientation     = None
        self.current_rotation_matrix = None

        # Location/Waypoints
        self.drop_pose          = [-0.81, 0.1, 0.3, 0.7, -0.7, 0.0, 0.0]
        self.bad_fruit_waypoint = [-0.6, 0.15, 0.5, 0.7, -0.7, 0.0, 0.0]
        self.home_location      = [0.12, -0.11, 0.445,  0.7, -0.7, 0.0, 0.0]

        # Error tolerances
        self.position_tolerance     = 0.05
        self.orientation_tolerance  = 0.1

        # Velocity thresholdings
        self.max_angular_velocity   = 1.0  # rad/s
        self.max_linear_velocity    = 0.2  # rad/s
        self.max_joint_velocity     = 1.0  # rad/s

        # Control loop parameters
        self.kp_position = 1.5
        self.kp_orientation = 3.0
        self.kd_position = 0.3
        self.kd_orientation = 0.5
        self.dt = 0.5
        self.previous_position_error = np.zeros(3)
        self.previous_orientation_error = np.zeros(3)

        # Velocity smoothing
        self.velocity_smoothing_factor = 0.3            # 0 = no smoothing, 1 = full smoothing
        self.previous_linear_velocity  = np.zeros(3)
        self.previous_angular_velocity = np.zeros(3)
        self.previous_joint_velocities = np.zeros(6)

    def trigger_sequence_0(self):
        """Auto-trigger Sequence 0 by calling the callback internally"""
        # Cancel timer
        self.auto_sequence_timer.cancel()
        self.get_logger().info('Auto-triggering Sequence 0...')
        
        # Create a dummy request with data = False (for Sequence 0)
        request = SetBool.Request()
        request.data = False
        response = SetBool.Response()
        
        # Call the pick_and_place_callback directly
        self.pick_and_place_callback(request, response)
        self.get_logger().info(f'Auto-Sequence 0 result: {response.message}')

    def pose_callback(self):
        self.current_pose = self.lookup_tf('tool0')
        self.current_position = np.array([self.current_pose[0],self.current_pose[1],self.current_pose[2]])
        self.current_orientation = np.array([self.current_pose[3],self.current_pose[4],self.current_pose[5],self.current_pose[6]])

    def joint_state_callback(self, msg):
        """Store current joint positions from /joint_states topic"""
        self.joint_positions = list(msg.position)

    def load_kdl_chain(self, urdf_path, base_link, end_link):
        """Load KDL chain from URDF file"""
        success, kdl_tree = treeFromUrdfModel(self.robot)
        if not success:
            raise RuntimeError("Failed to parse URDF into KDL tree.")
        chain = kdl_tree.getChain(base_link, end_link)
        return chain

    def compute_jacobian(self, chain, joint_positions):
        """Compute Jacobian matrix for current joint configuration"""
        nj = chain.getNrOfJoints()
        joint_array = PyKDL.JntArray(nj)
        for i in range(nj):
            joint_array[i] = joint_positions[i]
        
        jacobian = PyKDL.Jacobian(nj)
        solver = PyKDL.ChainJntToJacSolver(chain)
        solver.JntToJac(joint_array, jacobian)
        return jacobian

    def compute_joint_velocities(self, jacobian, twist_cmd):
        """Convert Cartesian twist to joint velocities using Jacobian"""
        # Convert KDL Jacobian to numpy array
        J = np.array([[jacobian[i, j] for j in range(jacobian.columns())] 
                    for i in range(6)])

        # compute damped pseudo-inverse of Jacobian 
        JT = J.T
        identity = np.eye(J.shape[0])
        J_pinv = JT @ np.linalg.inv(J @ JT + 0.01**2 * identity)    # Damping = 0.01
        
        return J_pinv @ twist_cmd

    def force_callback(self, msg):
        if self.flag_force:
            force_z = msg.data
            self.get_logger().info(f'Force in Z: {force_z}')

    def control_magnet(self, state):
        while not self.magnet_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Magnet service not available, waiting...')

        request = SetBool.Request()
        request.data = state  # True to activate, False to deactivate
        future = self.magnet_client.call_async(request)
        return future

    def lookup_tf(self, frame_id, timeout_sec=2.0):
        start = self.get_clock().now()
        while (self.get_clock().now() - start).nanoseconds < timeout_sec * 1e9:

            try:
                tf = self.tf_buffer.lookup_transform(
                    'base_link',
                    frame_id,
                    rclpy.time.Time()
                )
                return [
                    tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.translation.z,
                    tf.transform.rotation.x,
                    tf.transform.rotation.y,
                    tf.transform.rotation.z,
                    tf.transform.rotation.w,
                ]
            except tf2_ros.TransformException:
                self.get_logger().debug(f'Waiting for TF: {frame_id}')
                self.rate.sleep()

        self.get_logger().error(f'Timeout waiting for TF: {frame_id}')
        return None

    def goal_pose_nav(self,target_location):

        self.get_logger().info(f"Goal pose navigation to target: {target_location}")
        target_position = np.array(target_location[:3])
        target_quaternion = np.array(target_location[3:7])
        target_rotation_matrix = tf_transformations.quaternion_matrix(target_quaternion)[:3, :3]

        z_unit_vector = np.array([0.0, 0.0, 1.0])
        desired_z_axis = target_rotation_matrix  @ z_unit_vector

        position_reached = False
        orientation_reached = False

        self.previous_position_error = np.zeros(3)
        self.previous_orientation_error = np.zeros(3)
        self.previous_linear_velocity = np.zeros(3)
        self.previous_angular_velocity = np.zeros(3)

        while rclpy.ok():
            if self.joint_positions is None:
                self.get_logger().warn('Waiting for joint states to be available...')
                self.rate.sleep()
                continue
            try:
                self.current_rotation_matrix = tf_transformations.quaternion_matrix(self.current_orientation)[:3, :3]

                #Position error
                position_error = target_position - self.current_position
                position_derivative = (position_error - self.previous_position_error)/self.dt

                if self.flag_max_linear_velocity == 0:
                    self.max_linear_velocity = 0.1
                else:
                    self.max_linear_velocity = 0.05

                desired_velocity = self.kp_position * position_error + self.kd_position * position_derivative
                desired_velocity = np.clip(desired_velocity, -self.max_linear_velocity, self.max_linear_velocity)

                # Smooth linear velocity
                desired_velocity = (1 - self.velocity_smoothing_factor) * desired_velocity + self.velocity_smoothing_factor * self.previous_linear_velocity
                self.previous_linear_velocity = desired_velocity
                self.previous_position_error = position_error

                position_error_norm = np.linalg.norm(position_error)

                current_z_axis = self.current_rotation_matrix @ z_unit_vector
                orientation_error = np.cross(current_z_axis, desired_z_axis)
                orientation_derivative = (orientation_error - self.previous_orientation_error)/self.dt

                orientation_error_norm = np.linalg.norm(orientation_error)
                omega_world = self.kp_orientation * orientation_error + self.kd_orientation * orientation_derivative
                omega_world = np.clip(omega_world, -self.max_angular_velocity, self.max_angular_velocity)

                # Smooth angular velocity
                omega_world = (1 - self.velocity_smoothing_factor) * omega_world + self.velocity_smoothing_factor * self.previous_angular_velocity
                self.previous_angular_velocity = omega_world
                self.previous_orientation_error = orientation_error

                # Build twist command as numpy array
                twist_array = np.zeros(6)

                if position_error_norm > self.position_tolerance and position_reached == False: 
                    twist_array[0] = desired_velocity[0]  # linear x
                    twist_array[1] = desired_velocity[1]  # linear y
                    twist_array[2] = desired_velocity[2]  # linear z
                elif position_reached == False:
                    position_reached = True

                if orientation_error_norm > self.orientation_tolerance and orientation_reached == False:
                    twist_array[3] = omega_world[0]     # angular x
                    twist_array[4] = omega_world[1]     # angular y
                    twist_array[5] = omega_world[2]     # angular z
                elif orientation_reached == False:
                    orientation_reached = True

                # Compute Jacobian and convert twist to joint velocities
                jacobian = self.compute_jacobian(self.kdl_chain, self.joint_positions)
                joint_velocities = self.compute_joint_velocities(jacobian, twist_array)

                # Clip joint velocities
                joint_velocities = np.clip(joint_velocities, -self.max_joint_velocity, self.max_joint_velocity)

                # Smooth joint velocities
                joint_velocities = (1 - self.velocity_smoothing_factor) * joint_velocities + self.velocity_smoothing_factor * self.previous_joint_velocities
                self.previous_joint_velocities = joint_velocities
                
                # Publish joint velocities
                joint_cmd = JointJog()
                joint_cmd.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
                joint_cmd.velocities = joint_velocities.tolist()
                self.joint_velocity_publisher.publish(joint_cmd)
                self.rate.sleep()

                if position_reached == True and orientation_reached == True:
                    self.get_logger().info(f"Goal pose reached at target: {target_location}")
                    return True

            except Exception as e:
                self.get_logger().error(f"Exception in goal_pose_nav: {e}")
                self.rate.sleep()

    def pick_and_place_callback(self, request, response):

        self.flag_target_location = False

        while not self.flag_target_location:
            self.rate.sleep()
            try:
                self.fertiliser_pose = self.lookup_tf('5076_fertilizer_1')
                if request.data == False:
                    self.ebot_pose = self.lookup_tf('5076_ebot_6')
                if self.fertiliser_pose:
                    self.flag_target_location = True
                    self.get_logger().info('TF lookup successful — poses stored')
            except tf2_ros.LookupException:
                self.get_logger().debug('TF not available yet, retrying ...')
            except tf2_ros.TransformException as ex:
                self.get_logger().warn(f'TF error: {ex}')

        if request.data == False:  # Sequence 0

            self.fertiliser_pose[1] += 0.1
            self.goal_pose_nav(self.fertiliser_pose)
            self.flag_max_linear_velocity = 1
            self.fertiliser_pose[1] -= 0.1
            self.goal_pose_nav(self.fertiliser_pose)
            self.flag_max_linear_velocity = 0

            self.control_magnet(True)
            time.sleep(1.0)

            self.fertiliser_pose[1] += 0.2
            self.goal_pose_nav(self.fertiliser_pose)

            self.ebot_pose[2] += 0.2
            self.goal_pose_nav(self.ebot_pose)

            self.control_magnet(False)

            self.flag_fruit = 1

            response.success = True
            response.message = "Fertiliser can placed successfully"

        else:  # Sequence 1

            self.goal_pose_nav([0.1, -0.2, 0.5, 0.7, -0.7, 0.0, 0.0])

            self.fertiliser_pose[2] += 0.1
            self.goal_pose_nav(self.fertiliser_pose)
            self.fertiliser_pose[2] -= 0.1
            self.goal_pose_nav(self.fertiliser_pose)
            self.control_magnet(True)

            self.flag_fertilizer_drop = 1

            response.success = True
            response.message = "Sequence 1 completed successfully"

        return response

    def bad_fruit_callback(self):

        if self.flag_fruit == 0:
            return
        self.bad_fruit_trigger.cancel()
        self.flag_fruit_location = False

        fruit_frames = ['5076_bad_fruit_1','5076_bad_fruit_2','5076_bad_fruit_3']
        self.bad_fruits = [None] * 3

        while not self.flag_fruit_location:
            self.rate.sleep()
            try:
                for i in range(3):
                    pose = self.lookup_tf(fruit_frames[i])
                    self.bad_fruits[i] = pose
                self.flag_fruit_location = True
                self.get_logger().info('TF lookup successful — poses stored')
            except tf2_ros.LookupException:
                self.get_logger().debug('TF not available yet, retrying ...')
            except tf2_ros.TransformException as ex:
                self.get_logger().warn(f'TF error: {ex}')

        for i in range(3):
            fruit_pose = self.bad_fruits[i]

            fruit_pose[2] += 0.1
            self.goal_pose_nav(fruit_pose)
            self.flag_max_linear_velocity = 1
            fruit_pose[2] -= 0.08
            self.goal_pose_nav(fruit_pose)
            self.flag_max_linear_velocity = 0

            self.control_magnet(True)
            time.sleep(1.0)

            fruit_pose[2] += 0.1
            self.goal_pose_nav(fruit_pose)
            self.goal_pose_nav(self.bad_fruit_waypoint)

            self.goal_pose_nav(self.drop_pose)
            self.control_magnet(False)

            self.goal_pose_nav(self.bad_fruit_waypoint)

        self.flag_fruit = 0

    def drop_fertilizer_callback(self):

        if self.flag_fertilizer_drop == 0:
            return
        self.drop_fertilizer_trigger.cancel()

        self.goal_pose_nav([0.1, 0.2, 0.5, 0.7, -0.7, 0.0, 0.0])
        self.goal_pose_nav(self.bad_fruit_waypoint)

        self.goal_pose_nav(self.drop_pose)
        self.control_magnet(False)

        self.flag_fertilizer_drop = 0

def main():
    rclpy.init()
    robot_controller = ArmController()
    executor = MultiThreadedExecutor()
    executor.add_node(robot_controller)
    executor.spin()
    robot_controller.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
