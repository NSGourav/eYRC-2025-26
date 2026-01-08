#!/usr/bin/env python3

# Team ID:          [ eYRC#5076 ]
# Author List:		[ Nagulapalli Shavaneeth Gourav, Pradeep J, Anand Vardhan, Raj Mohammad ]
# Filename:		    Task4A_manipulation.py
# Functions:
#			        [pose_callback, force_callback, control_magnet, arm_vel_publish, lookup_tf, bad_fruit_callback, goal_pose_nav, control_loop]
# Nodes:		    Add your publishing and subscribing node
#			        Publishing Topics  - [ /delta_twist_cmds ]
#                   Subscribing Topics - [ /tcp_pose_raw], /net_wrench

import rclpy
from rclpy.node import Node
import tf2_ros
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped, TwistStamped
from std_msgs.msg import String
from rclpy.callback_groups import ReentrantCallbackGroup,MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import tf_transformations
from std_srvs.srv import Trigger
import time
from std_srvs.srv import SetBool
from std_msgs.msg import Float32

class ArmController(Node):
    def __init__(self):
        super().__init__("arm_controller_node")

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.cmd_pub = self.create_publisher(TwistStamped, "/delta_twist_cmds", 10)
        self.force_sub = self.create_subscription(Float32, '/net_wrench', self.force_callback, 10)
        self.magnet_client = self.create_client(SetBool, '/magnet')
        self.callback_group = ReentrantCallbackGroup()
        self.fruit_cb_group=MutuallyExclusiveCallbackGroup()
        
        self.pose_sub=self.create_timer(0.5,self.pose_callback,callback_group=self.fruit_cb_group)
        self.rate = self.create_rate(1.2, self.get_clock())

        self.flag_force=False
        
        self.current_position = None
        self.current_orientation = None 
        self.current_rotation_matrix = None
        self.flag_fruit=0
        self.detection_service = self.create_timer(0.5, self.control_loop,callback_group=self.callback_group)
        self.bad_fruit_service = self.create_timer(0.5, self.bad_fruit_callback,callback_group=self.callback_group)

        self.team_id = "5076"
        self.drop_pose = [-0.81, 0.01, 0.3, 0.7, -0.7, 0.0, 0.0]

        self.fertiliser_pose = None

        # Error positions
        self.pos_tol = 0.05  # Position tolerance
        self.ori_tol = 0.1  # Orientation tolerance
        self.omega_limit=1.0
        self.v_limit=0.2
        self.v_min = 0.05
        
        # Control loop parameters
        self.kp_position=1.5
        self.kp_orientation=3.0

    def pose_callback(self):
        self.current_pose=self.lookup_tf('tool0')
        self.current_position =np.array([self.current_pose[0],self.current_pose[1],self.current_pose[2]])
        self.current_orientation =np.array([self.current_pose[3],self.current_pose[4],self.current_pose[5],self.current_pose[6]])

    def arm_vel_publish(self,twist_cmd):

        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.twist.linear.x =  twist_cmd.twist.linear.x
        twist_msg.twist.linear.y = twist_cmd.twist.linear.y
        twist_msg.twist.linear.z = twist_cmd.twist.linear.z
        twist_msg.twist.angular.x = twist_cmd.twist.angular.x
        twist_msg.twist.angular.y = twist_cmd.twist.angular.y
        twist_msg.twist.angular.z = twist_cmd.twist.angular.z
        self.cmd_pub.publish(twist_msg)

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
    
    def bad_fruit_callback(self):
        if self.flag_fruit == 0:
            return
        self.bad_fruit_service.cancel()

        self.flag_fruit_loc = False

        fruit_frames = [
            '5076_bad_fruit_0',
            '5076_bad_fruit_1',
            '5076_bad_fruit_2'
        ]

        self.bad_fruits = [None] * 3

        # ---------------- TF lookup ----------------
        while not self.flag_fruit_loc:
            self.rate.sleep()
            try:
                for i in range(3):
                    pose = self.lookup_tf(fruit_frames[i])
                    # pose[0] += 0.02     # x-offset
                    self.bad_fruits[i] = pose

                self.flag_fruit_loc = True
                self.get_logger().info('TF lookup successful — poses stored')

            except tf2_ros.LookupException:
                self.get_logger().debug('TF not available yet, retrying ...')

            except tf2_ros.TransformException as ex:
                self.get_logger().warn(f'TF error: {ex}')

        # ---------------- Pick & place ----------------
        self.goal_pose_nav([0.1, -0.2, 0.5, 0.7, -0.7, 0.0, 0.0])
        self.goal_pose_nav([0.1, 0.2, 0.5, 0.7, -0.7, 0.0, 0.0])

        self.bad_fruit_waypoint = [-0.4, 0.2, 0.5, 0.7, -0.7, 0.0, 0.0]        

        for i in range(3):
            fruit_pose = self.bad_fruits[i]

            self.goal_pose_nav(self.bad_fruit_waypoint)

            fruit_pose[2] += 0.1
            self.goal_pose_nav(fruit_pose)
            fruit_pose[2] -= 0.08
            self.goal_pose_nav(fruit_pose)

            self.control_magnet(True)

            fruit_pose[2] += 0.1
            self.goal_pose_nav(fruit_pose)
            self.goal_pose_nav(self.bad_fruit_waypoint)

            self.goal_pose_nav(self.drop_pose)
            self.control_magnet(False)

        # self.goal_pose_nav(home_pose)
        self.flag_fruit=0

    def goal_pose_nav(self,target_location):

        self.get_logger().info(f"Goal pose navigation to target: {target_location}")
        target_pos = np.array(target_location[:3])
        target_quat = np.array(target_location[3:7])
        target_rotation_matrix = tf_transformations.quaternion_matrix(target_quat)[:3, :3]

        e3 = np.array([0.0, 0.0, 1.0])
        z_d = target_rotation_matrix  @ e3

        pose_flag=False
        ori_flag=False

        while rclpy.ok():
            try:                
                rot_matrix_4x4 = tf_transformations.quaternion_matrix(self.current_orientation)
                self.current_rotation_matrix = rot_matrix_4x4[:3, :3]

                #Position error
                e_p_base = target_pos - self.current_position
                v_ee = self.kp_position * e_p_base
                v_ee = np.clip(v_ee, -self.v_limit, self.v_limit)
                v_ee = np.where(
                    np.abs(v_ee) < self.v_min,
                    self.v_min * np.sign(v_ee),
                    v_ee
                )
                norm_ep = np.linalg.norm(e_p_base)

                z   = self.current_rotation_matrix @ e3
                z_err = np.cross(z, z_d)
                err_ori = np.linalg.norm(z_err)

                twist_cmd = Twist()

                if err_ori > self.ori_tol and ori_flag == False:
                    omega_world = self.kp_orientation * z_err
                    omega_world = np.clip(omega_world, -self.omega_limit, self.omega_limit)
                    twist_cmd.angular.x = omega_world[0]
                    twist_cmd.angular.y = omega_world[1]
                    twist_cmd.angular.z = omega_world[2]

                elif ori_flag == False:
                    ori_flag = True

                if (norm_ep>self.pos_tol and pose_flag==0): 
                    twist_cmd.linear.x = v_ee[0]
                    twist_cmd.linear.y = v_ee[1]
                    twist_cmd.linear.z = v_ee[2]

                elif pose_flag==0:
                    pose_flag=True
                
                self.arm_vel_publish(twist_cmd)
                self.rate.sleep()

                if pose_flag==1 and ori_flag==1:
                    self.get_logger().info(f"Goal pose reached at target: {target_location}")
                    return True
                
            except Exception as e:
                self.get_logger().error(f"Exception in goal_pose_nav: {e}")
                self.rate.sleep()

    def control_loop(self):
        self.detection_service.cancel()
        self.flag_target_loc=False

        while not self.flag_target_loc:
            self.rate.sleep()
            try:
                self.fertiliser_pose = self.lookup_tf('5076_fertiliser_can')
                if self.fertiliser_pose:
                    self.flag_target_loc = True
                    self.get_logger().info('TF lookup successful — poses stored')
                
            except tf2_ros.LookupException:
                self.get_logger().debug('TF not available yet, retrying ...')
            except tf2_ros.TransformException as ex:
                self.get_logger().warn(f'TF error: {ex}')
        
        self.fertiliser_pose[1]=self.fertiliser_pose[1]+0.01
        self.goal_pose_nav(self.fertiliser_pose)

        self.control_magnet(True)

        self.fertiliser_pose[1]=self.fertiliser_pose[1]+0.2
        self.goal_pose_nav(self.fertiliser_pose)

        self.goal_pose_nav(self.drop_pose)
        self.control_magnet(False)

        print('fertliser done')
        self.flag_fruit=1
    
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