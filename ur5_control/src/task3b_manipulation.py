#!/usr/bin/env python3

# Team ID:          [ eYRC#5076 ]
# Author List:		[ Nagulapalli Shavaneeth Gourav, Pradeep J, Anand Vardhan, Raj Mohammad ]
# Filename:		    Task2B_bad_fruit_perception.py
# Functions:
#			        [ pose_callback, gripper_service, call_detach_service_async, check_service_result,
#                        quaternion_multiply, quaternion_conjugate, compute_orientation_error_quaternion, get_target_transform, control_loop ]
# Nodes:		    Add your publishing and subscribing node
#			        Publishing Topics  - [ /delta_twist_cmds ]
#                   Subscribing Topics - [ /tcp_pose_raw]

import rclpy
from rclpy.node import Node
import tf2_ros
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String
from linkattacher_msgs.srv import AttachLink, DetachLink
from rclpy.callback_groups import ReentrantCallbackGroup,MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import tf_transformations
from std_srvs.srv import Trigger
import time
# Test command : ros2 service call /pick_and_place std_srvs/srv/Trigger {}\ 


class MarkerFollower(Node):
    def __init__(self):
        super().__init__("marker_follower")

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publisher: publish end-effector velocity commands
        self.cmd_pub = self.create_publisher(Twist, "/delta_twist_cmds", 10)
        self.callback_group = ReentrantCallbackGroup()
        self.fruit_cb_group=MutuallyExclusiveCallbackGroup()
        
        # Subscriber: get current end-effector pose
        self.sub = self.create_subscription(PoseStamped, '/tcp_pose_raw', self.pose_callback, 10,callback_group=self.callback_group)
        self.rate = self.create_rate(1.2, self.get_clock())

        # Service clients for attach/detach
        self.attach_client = self.create_client(AttachLink, '/attach_link')
        self.detach_client = self.create_client(DetachLink, '/detach_link')

        self.current_position = None
        self.current_orientation = None 
        self.current_rotation_matrix = None
        self.detection_service = self.create_service(Trigger,'/pick_and_place', self.control_loop,callback_group=self.callback_group)
        self.bad_fruit_pick=self.create_timer(0.5,self.bad_fruit_callback,callback_group=self.fruit_cb_group)
        self.team_id = "5076"
        self.flag_fruit=0

        self.fertiliser_pose = None
        self.ebot_pose = None
        self.home_location=[]

        # Error positions
        self.pos_tol = 0.02  # 2.5 cm position tolerance
        self.ori_tol = 0.01  # ~10 degrees in radians
        
        # Control loop parameters

        self.kp_position=1.5
        self.kd_position=0.1
        self.prev_error_pos=[0.0,0.0,0.0]

        self.kp_orientation=3.0
        self.kd_orientation=0.1
        self.prev_error_orientation=[0.0,0.0,0.0]

        

    def pose_callback(self, msg):
        """ Updates the current position and orientation of the end-effector """
        self.current_position = np.array([msg.pose.position.x,msg.pose.position.y,msg.pose.position.z])
        self.current_orientation = np.array([msg.pose.orientation.x,msg.pose.orientation.y,msg.pose.orientation.z,msg.pose.orientation.w])


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
                self.get_logger().info(f'TF found for {frame_id}')
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

        if self.flag_fruit==1:
            self.flag_fruit_loc=False

            while not self.flag_fruit_loc:
                self.rate.sleep()
                try:
                    self.bad_fruit_1  = self.lookup_tf('5076_bad_fruit_0')
                    self.bad_fruit_1[0]=self.bad_fruit_1[0]+0.02

                    self.bad_fruit_2  = self.lookup_tf('5076_bad_fruit_1')
                    self.bad_fruit_2[0]=self.bad_fruit_2[0]+0.02

                    self.bad_fruit_3  = self.lookup_tf('5076_bad_fruit_2')
                    self.bad_fruit_3[0]=self.bad_fruit_3[0]+0.02



                    self.flag_fruit_loc = True
                    self.get_logger().info('TF lookup successful — poses stored')


                except tf2_ros.LookupException:
                    self.get_logger().debug('TF not available yet, retrying ...')

                except tf2_ros.TransformException as ex:
                    self.get_logger().warn(f'TF error: {ex}')


            self.goal_pose_nav([0.1,0.4,0.4,0.7,-0.7,0.0,0.0])
            self.bad_fruit_1[2]=self.bad_fruit_1[2]+0.1
            self.goal_pose_nav(self.bad_fruit_1)
            self.bad_fruit_1[2]=self.bad_fruit_1[2]-0.07
            self.goal_pose_nav(self.bad_fruit_1)
            self.gripper_service("attach","bad_fruit_00")
            self.bad_fruit_1[2]=self.bad_fruit_1[2]+0.2
            self.goal_pose_nav([-0.8,0.01,0.3,0.7,-0.7,0.0,0.0])
            self.gripper_service("detach","bad_fruit_00")


            self.goal_pose_nav([0.1,0.4,0.4,0.7,-0.7,0.0,0.0])
            self.bad_fruit_2[2]=self.bad_fruit_2[2]+0.1
            self.goal_pose_nav(self.bad_fruit_2)
            self.bad_fruit_2[2]=self.bad_fruit_2[2]-0.07
            self.goal_pose_nav(self.bad_fruit_2)
            self.gripper_service("attach","bad_fruit_12")
            self.bad_fruit_2[2]=self.bad_fruit_2[2]+0.2
            self.goal_pose_nav([-0.8,0.01,0.3,0.7,-0.7,0.0,0.0])
            self.gripper_service("detach","bad_fruit_12")

            self.goal_pose_nav([0.1,0.4,0.4,0.7,-0.7,0.0,0.0])
            self.bad_fruit_3[2]=self.bad_fruit_3[2]+0.1
            self.goal_pose_nav(self.bad_fruit_3)
            self.bad_fruit_3[2]=self.bad_fruit_3[2]-0.07
            self.goal_pose_nav(self.bad_fruit_3)
            self.gripper_service("attach","bad_fruit_01")
            self.bad_fruit_3[2]=self.bad_fruit_3[2]+0.2
            self.goal_pose_nav([-0.8,0.01,0.3,0.7,-0.7,0.0,0.0])
            self.gripper_service("detach","bad_fruit_01")

            self.flag_fruit=0
         # self.goal_pose_nav(self.bad_fruit_1)

    def goal_pose_nav(self,target_var):
        pose_flag=0
        ori_flag=0
        target_pos = np.array([target_var[0],target_var[1],target_var[2]])
        target_ori = np.array([target_var[3],target_var[4],target_var[5],target_var[6]])
        rot_matrix_4x4 = tf_transformations.quaternion_matrix(target_ori)
        target_rotation_matrix = rot_matrix_4x4[:3, :3]
        e3 = np.array([0.0, 0.0, 1.0])
  
        z_d = target_rotation_matrix  @ e3

        while True:
            try:                
                #Position error
                rot_matrix_4x4 = tf_transformations.quaternion_matrix(self.current_orientation)
                self.current_rotation_matrix = rot_matrix_4x4[:3, :3]

                e_p_base = target_pos - self.current_position
                e_p_ee = self.current_rotation_matrix.T @ e_p_base
                v_ee = self.kp_position * e_p_ee
                v_ee = np.clip(v_ee, -0.1, 0.1)
                v_min = 0.015
                v_ee = np.where(
                    np.abs(v_ee) < v_min,
                    v_min * np.sign(v_ee),
                    v_ee
                )
                norm_ep = np.linalg.norm(e_p_ee)


                # Current and desired z axes (world frame)
                z   = self.current_rotation_matrix @ e3

                # Z-axis alignment error
                z_err = np.cross(z, z_d)
                err_ori = np.linalg.norm(z_err)

                twist_cmd = Twist()

                if err_ori > self.ori_tol and ori_flag == 0:
                    omega_world = self.kp_orientation * z_err
                    omega_body = self.current_rotation_matrix.T @ omega_world
                    omega_body = np.clip(omega_body, -1.0, 1.0)
                    twist_cmd.angular.x = omega_body[0]
                    twist_cmd.angular.y = omega_body[1]
                    twist_cmd.angular.z = omega_body[2]
                elif ori_flag == 0:
                    ori_flag = 1
                    twist_cmd.angular.x = 0.0
                    twist_cmd.angular.y = 0.0
                    twist_cmd.angular.z = 0.0


                if (norm_ep>self.pos_tol and pose_flag==0): 
                    twist_cmd.linear.x = v_ee[0]
                    twist_cmd.linear.y = v_ee[1]
                    twist_cmd.linear.z = v_ee[2]

                elif pose_flag==0:
                    pose_flag=1

                
                self.cmd_pub.publish(twist_cmd)
                self.rate.sleep()

                if pose_flag==1 and ori_flag==1:
                    self.get_logger().info(f"Goal pose reached at target: {target_var}")
                    return True
                
            except Exception as e:
                self.get_logger().error(f"Exception in goal_pose_nav: {e}")
                self.rate.sleep()



    def control_loop(self,request,response):

        self.flag_target_loc=False

        while not self.flag_target_loc:
            self.rate.sleep()
            try:
                self.fertiliser_pose = self.lookup_tf('5076_fertiliser_can')
                self.ebot_pose = self.lookup_tf('5076_ebot_6')
                if self.fertiliser_pose and self.ebot_pose:
                    self.flag_target_loc = True
                self.get_logger().info('TF lookup successful — poses stored')

            except tf2_ros.LookupException:
                self.get_logger().debug('TF not available yet, retrying ...')

            except tf2_ros.TransformException as ex:
                self.get_logger().warn(f'TF error: {ex}')
        
        self.fertiliser_pose[1]=self.fertiliser_pose[1]+0.01
        self.goal_pose_nav(self.fertiliser_pose)
        self.gripper_service("attach","fertiliser_can")
        self.fertiliser_pose[1]=self.fertiliser_pose[1]+0.2
        self.goal_pose_nav(self.fertiliser_pose)
        self.ebot_pose[2]=self.ebot_pose[2]+0.3
        self.goal_pose_nav(self.ebot_pose)
        self.gripper_service("detach","fertiliser_can")

        response.success = True
        response.message = "Pick and place done"
        print('response sent')
        self.flag_fruit=1
        return response
    
def main():
    rclpy.init()
    robot_controller = MarkerFollower()
    executor = MultiThreadedExecutor()
    executor.add_node(robot_controller)
    executor.spin()
    robot_controller.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()