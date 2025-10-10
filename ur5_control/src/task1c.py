#!/usr/bin/env python3
import rclpy
import PyKDL as kdl
from urdf_parser_py.urdf import URDF
import numpy as np
import sys
import os
import time
import importlib.util
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from ament_index_python.packages import get_package_share_directory

# Get the absolute path of the file inside the package
package_path = get_package_share_directory("pymoveit2")
kdl_file_path = os.path.join(package_path, "examples", "kdl_parser_py.py")

# Dynamically import the file
spec = importlib.util.spec_from_file_location("kdl_parser_py", kdl_file_path)
kdl_parser_py = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kdl_parser_py)
treeFromUrdfModel = kdl_parser_py.treeFromUrdfModel

class JointServoingNode(Node):
    def __init__(self):
        super().__init__('joint_servoing_node')

        # --- Load URDF and Build KDL Chain ---
        urdf_path = os.path.join(package_path, "examples", "ur5_arm.urdf")
        robot_model = URDF.from_xml_file(urdf_path)
        ok, tree = treeFromUrdfModel(robot_model)
        if not ok:
            self.get_logger().error("Failed to build KDL tree from URDF")
            return

        base_link = "base_link"
        ee_link = "wrist_3_link"
        self.chain = tree.getChain(base_link, ee_link)
        self.num_joints = self.chain.getNrOfJoints()

        # --- Define Multiple Waypoints (XYZ + Quaternion XYZW) ---
        self.waypoints = [
            {"pos": [-0.214, -0.532, 0.557], "quat": [0.707, 0.028, 0.034, 0.707]},
            {"pos": [ -0.159, 0.501, 0.415 ], "quat": [0.029, 0.997, 0.045,0.033]},
            {"pos": [ -0.706, 0.010, 0.182] , "quat":  [-0.684, 0.726, 0.05, 0.008]}, # -0.806
        ]

        # --- Compute IK for Each Waypoint ---
        ik_solver = kdl.ChainIkSolverPos_LMA(self.chain)
        current_guess = kdl.JntArray(self.num_joints)
        home_pos = [0, -np.pi/2, np.pi/2, 0, np.pi/2, 0]
        for i in range(self.num_joints):
            current_guess[i] = home_pos[i]

        self.q_goals = []
        for idx, wp in enumerate(self.waypoints):
            pos, quat = wp["pos"], wp["quat"]   # np.array(wp["quat"], dtype=float)
            # quat = quat / np.linalg.norm(quat)
            rotation = kdl.Rotation.Quaternion(*quat)
            frame = kdl.Frame(rotation, kdl.Vector(*pos))
            result = kdl.JntArray(self.num_joints)
            ret = ik_solver.CartToJnt(current_guess, frame, result)
            if ret >= 0:
                q_goal = np.array([result[i] for i in range(self.num_joints)])
                self.q_goals.append(q_goal)
                self.get_logger().info(f"IK solution {idx+1}: {q_goal}")
                current_guess = result
            else:
                self.get_logger().error(f"Failed IK for waypoint {idx+1}")

        # --- Initialize ROS Interfaces ---
        self.current_joints = np.zeros(self.num_joints)
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint",
            "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]

        self.create_subscription(JointState, "/joint_states", self.joint_state_callback, 10)
        self.vel_pub = self.create_publisher(Float64MultiArray, "/delta_joint_cmds", 10)

        self.timer = self.create_timer(0.01, self.control_loop)  # 100 Hz
        self.kp = 0.7
        self.kd = 0.02  # Derivative gain
        self.prev_delta_q = np.zeros(self.num_joints)
        self.dt = 0.01
        self.current_goal_idx = 0
        self.q_goal = self.q_goals[self.current_goal_idx]
        self.reached_goal = False
        self.pause_time = 1.0       # pause 1 second at each waypoint
        self.last_reach_time = None

    def joint_state_callback(self, msg: JointState):
        name_to_pos = dict(zip(msg.name, msg.position))
        self.current_joints = np.array([name_to_pos.get(j, 0.0) for j in self.joint_names])

    def control_loop(self):
        # If all waypoints completed
        if self.current_goal_idx >= len(self.q_goals):
            if not hasattr(self, "done_logged"):
                self.get_logger().info("All waypoints reached. Stopping servoing.")
                self.done_logged = True
            return

        delta_q = self.q_goal - self.current_joints
        vel_cmd = self.kp * delta_q + self.kd * (delta_q - self.prev_delta_q) / self.dt
        vel_cmd = np.clip(vel_cmd, -0.5, 0.5)   # Velocity thresholding added here

        # --- Compute current end-effector pose via FK ---
        fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        current_kdl_joints = kdl.JntArray(self.num_joints)
        for i in range(self.num_joints):
            current_kdl_joints[i] = self.current_joints[i]

        current_frame = kdl.Frame()
        fk_solver.JntToCart(current_kdl_joints, current_frame)

        current_pos = np.array([current_frame.p.x(), current_frame.p.y(), current_frame.p.z()])
        current_quat = np.array(current_frame.M.GetQuaternion())

        # --- Target pose for this waypoint ---
        wp = self.waypoints[self.current_goal_idx]
        target_pos = np.array(wp["pos"])
        target_quat = np.array(wp["quat"])

        # --- Compute pose errors ---
        pos_error = np.linalg.norm(target_pos - current_pos)
        dot = np.abs(np.dot(current_quat, target_quat))
        dot = np.clip(dot, -1.0, 1.0)
        ori_error = 2 * np.arccos(dot)

        # --- Use pose-based tolerance instead of joint-space tolerance ---
        if pos_error < 0.04 and ori_error < np.deg2rad(10):
            if not self.reached_goal:
                self.get_logger().info(f"Reached waypoint {self.current_goal_idx + 1}")
                self.reached_goal = True
                self.last_reach_time = time.time()
            # Wait before moving to next
            if time.time() - self.last_reach_time > self.pause_time:
                self.current_goal_idx += 1
                if self.current_goal_idx < len(self.q_goals):
                    self.q_goal = self.q_goals[self.current_goal_idx]
                    self.reached_goal = False
                    self.get_logger().info(f"➡️ Moving to waypoint {self.current_goal_idx + 1}")
            vel_cmd = np.zeros_like(vel_cmd)
        
        self.prev_delta_q = delta_q.copy()
        msg = Float64MultiArray()
        msg.data = list(vel_cmd)
        self.vel_pub.publish(msg)

        # Log every ~1 second
        self.log_counter = getattr(self, "log_counter", 0) + 1
        if self.log_counter % 50 == 0:
            self.get_logger().info(f"Publishing velocity: {vel_cmd}")


def main():
    rclpy.init()
    node = JointServoingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
