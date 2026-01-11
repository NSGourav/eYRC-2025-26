#!/usr/bin/env python3

# Team ID:          [ eYRC#5076 ]
# Author List:		[ Nagulapalli Shavaneeth Gourav, Pradeep J, Anand Vardhan, Raj Mohammad ]
# Filename:		    Task4C_manipulation.py
# Functions:
#			        []
# Nodes:		    Add your publishing and subscribing node
#			        Publishing Topics  - [ /delta_twist_cmds ]
#                   Subscribing Topics - [ /tcp_pose_raw]

import rclpy
from rclpy.node import Node
import tf2_ros
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from linkattacher_msgs.srv import AttachLink, DetachLink
from rclpy.callback_groups import ReentrantCallbackGroup,MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import SetBool
import tf_transformations
import time

class ArmController(Node):
    def __init__(self):
        super().__init__("arm_controller_node")

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publisher: publish end-effector velocity commands
        self.cmd_pub = self.create_publisher(Twist, "/delta_twist_cmds", 10)
        self.callback_group = ReentrantCallbackGroup()
        self.fruit_cb_group=MutuallyExclusiveCallbackGroup()
        
        self.pose_sub=self.create_timer(0.5,self.pose_callback,callback_group=self.fruit_cb_group)
        self.rate = self.create_rate(1.2, self.get_clock())

        # Service clients for attach/detach
        self.attach_client = self.create_client(AttachLink, '/attach_link')
        self.detach_client = self.create_client(DetachLink, '/detach_link')

        self.current_position = None
        self.current_orientation = None 
        self.current_rotation_matrix = None
        self.flag_fruit=0

        self.pick_place_service = self.create_service(SetBool,'pick_and_place',self.pick_and_place_callback,callback_group=MutuallyExclusiveCallbackGroup())
        self.bad_fruit_service = self.create_timer(0.5, self.bad_fruit_callback,callback_group=self.callback_group)

        self.team_id = "5076"
        self.drop_pose = [-0.81, 0.1, 0.3, 0.7, -0.7, 0.0, 0.0]
        self.bad_fruit_waypoint = [-0.6, 0.15, 0.5, 0.7, -0.7, 0.0, 0.0]        

        self.fertiliser_pose = None
        self.ebot_pose = None
        self.home_location=[0.12, -0.11, 0.445,  0.7, -0.7, 0.0, 0.0]

        # Error positions
        self.position_tolerance = 0.02
        self.orientation_tolerance = 0.01  
        self.max_angular_velocity = 1.0
        self.max_linear_velocity = 0.1
        self.min_linear_velocity = 0.015
        
        # Control loop parameters
        self.kp_position = 1.5
        self.kp_orientation = 3.0

    def pose_callback(self):
        
        self.current_pose=self.lookup_tf('tool0')
        if self.current_pose is None:
            self.get_logger().warn('tool0 TF not available yet')
            return 
        self.current_position =np.array([self.current_pose[0],self.current_pose[1],self.current_pose[2]])
        self.current_orientation =np.array([self.current_pose[3],self.current_pose[4],self.current_pose[5],self.current_pose[6]])
    
    def gripper_service(self, command: str,model_n: str):
        services = {
            'attach': (self.attach_client, AttachLink),
            'detach': (self.detach_client, DetachLink)
        }

        if command not in services:
            self.get_logger().error('Invalid command: use "attach" or "detach"')
            return

        client, srv_type = services[command]

        request = srv_type.Request()
        request.model1_name = model_n
        request.link1_name  = 'body'
        request.model2_name = 'ur5'
        request.link2_name  = 'wrist_3_link'

        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error(f'{command} service not available')
            return

        self.service_future = client.call_async(request)
        self.get_logger().info(f'→ {command} request sent')
    
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

        while rclpy.ok():
            try:                
                self.current_rotation_matrix = tf_transformations.quaternion_matrix(self.current_orientation)[:3, :3]

                #Position error
                position_error = target_position - self.current_position
                desired_velocity = self.kp_position * position_error
                desired_velocity = np.clip(desired_velocity, -self.max_linear_velocity, self.max_linear_velocity)
                desired_velocity = np.where(
                    np.abs(desired_velocity) < self.min_linear_velocity,
                    self.min_linear_velocity * np.sign(desired_velocity),
                    desired_velocity
                )
                position_error_norm = np.linalg.norm(position_error)

                current_z_axis = self.current_rotation_matrix @ z_unit_vector
                orientation_error = np.cross(current_z_axis, desired_z_axis)
                orientation_error_norm = np.linalg.norm(orientation_error)
                omega_world = self.kp_orientation * orientation_error
                omega_world = np.clip(omega_world, -self.max_angular_velocity, self.max_angular_velocity)

                twist_cmd = Twist()

                if orientation_error_norm > self.orientation_tolerance and orientation_reached == False:
                    twist_cmd.angular.x = omega_world[0]
                    twist_cmd.angular.y = omega_world[1]
                    twist_cmd.angular.z = omega_world[2]
                elif orientation_reached == False:
                    orientation_reached = True

                if position_error_norm > self.position_tolerance and position_reached == False: 
                    twist_cmd.linear.x = desired_velocity[0]
                    twist_cmd.linear.y = desired_velocity[1]
                    twist_cmd.linear.z = desired_velocity[2]
                elif position_reached == False:
                    position_reached = True

                self.cmd_pub.publish(twist_cmd)
                self.rate.sleep()

                if position_reached == True and orientation_reached == True:
                    self.get_logger().info(f"Goal pose reached at target: {target_location}")
                    return True
                
            except Exception as e:
                self.get_logger().error(f"Exception in goal_pose_nav: {e}")
                self.rate.sleep()

    def bad_fruit_callback(self):

        if self.flag_fruit == 0:
            return
        self.bad_fruit_service.cancel()
        self.flag_fruit_location = False

        fruit_frames = ['5076_bad_fruit_0','5076_bad_fruit_1','5076_bad_fruit_2']
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
            fruit_pose[2] -= 0.08
            self.goal_pose_nav(fruit_pose)

            self.gripper_service("attach", 'bad_fruit')

            fruit_pose[2] += 0.1
            self.goal_pose_nav(fruit_pose)
            self.goal_pose_nav(self.bad_fruit_waypoint)

            self.goal_pose_nav(self.drop_pose)
            self.gripper_service("detach", 'bad_fruit')

            self.goal_pose_nav(self.bad_fruit_waypoint)

        self.flag_fruit = 0

    def pick_and_place_callback(self, request, response):
        
        self.flag_target_location = False
        
        while not self.flag_target_location:
            self.rate.sleep()
            try:
                self.fertiliser_pose = self.lookup_tf('5076_fertiliser_can')
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

            self.fertiliser_pose[1] += 0.02
            self.goal_pose_nav(self.fertiliser_pose)

            self.gripper_service("attach", "fertiliser_can")

            self.fertiliser_pose[1] += 0.2
            self.goal_pose_nav(self.fertiliser_pose)

            self.ebot_pose[2] += 0.2
            self.goal_pose_nav(self.ebot_pose)

            self.gripper_service("detach", "fertiliser_can")

            response.success = True
            response.message = "Fertiliser can placed successfully"

            self.flag_fruit = 1
            
        else:  # Sequence 1

            self.goal_pose_nav([0.1, -0.2, 0.5, 0.7, -0.7, 0.0, 0.0])

            self.fertiliser_pose[2] += 0.1
            self.goal_pose_nav(self.fertiliser_pose)
            self.fertiliser_pose[2] -= 0.08
            self.goal_pose_nav(self.fertiliser_pose)
            self.gripper_service("attach", "fertiliser_can")

            response.success = True
            response.message = "Sequence 1 completed successfully"

            self.goal_pose_nav([0.1, 0.2, 0.5, 0.7, -0.7, 0.0, 0.0])
            self.goal_pose_nav(self.bad_fruit_waypoint)

            self.goal_pose_nav(self.drop_pose)
            self.gripper_service("detach", "fertiliser_can")
        
        return response

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