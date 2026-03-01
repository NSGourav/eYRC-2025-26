#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import PyKDL
import importlib.util
from urdf_parser_py.urdf import URDF
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Float32
from scipy.spatial.transform import Rotation
from ament_index_python.packages import get_package_share_directory
import os

# package_path = get_package_share_directory("pymoveit2")
# kdl_file_path = os.path.join(package_path, "examples", "kdl_parser_py.py")
# kdl_file_path = "/home/gourav/eyantra_ws/src/pymoveit2/examples/kdl_parser_py.py"

package_path = os.path.join(os.environ["HOME"], "eyantra_ws", "src", "pymoveit2")
kdl_file_path = os.path.join(package_path, "examples", "kdl_parser_py.py")


# Dynamically import the file
spec = importlib.util.spec_from_file_location("kdl_parser_py", kdl_file_path)
kdl_parser_py = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kdl_parser_py)
treeFromUrdfModel = kdl_parser_py.treeFromUrdfModel

class JointServoingNode(Node):
    def __init__(self):
        super().__init__('joint_servoing_node')

        # --- Robot Setup ---
        urdf_path = os.path.join(package_path, "examples", "ur5_arm.urdf")
        base_link = 'base_link'
        end_link = 'tool0'

        self.robot = URDF.from_xml_file(urdf_path)
        self.kdl_chain = self.load_kdl_chain(urdf_path, base_link, end_link)

        # Solvers
        self.fk_solver = PyKDL.ChainFkSolverPos_recursive(self.kdl_chain)

        # Publishers & Subscribers
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.cmd_pub = self.create_publisher(Float64MultiArray, '/delta_joint_cmds', 10)
        self.state_pub = self.create_publisher(Float32, '/servo_status', 10)

        # Data holders
        self.joint_positions = None
        self.current_waypoint_index = 0
        self.loop_count = 0
        self.initial_pos = None
        self.initial_rot = None
        self.stable_count = 0
        self.align_orientation_first = True
        self.orientation_done = False


        # Waypoints (positions + quaternions)
        self.waypoints = [
            {"pos": [-0.214, -0.532, 0.552], "quat": [0.707, 0.028, 0.034, 0.707]}, #{"pos": [-0.214, -0.532, 0.557], "quat": [0.707, 0.028, 0.034, 0.707]},
            {"pos": [0.170, -0.109,  0.545], "quat": [0.500, 0.497, 0.503, 0.499]},
            {"pos": [-0.159, 0.501, 0.415],  "quat": [0.029, 0.997, 0.045, 0.033]},
            {"pos": [-0.806, 0.010, 0.182],  "quat": [-0.684, 0.726, 0.05, 0.008]}
        ]

        self.create_timer(0.01, self.control_loop)  # 100 Hz
    # --- KDL Utilities ---
    def load_kdl_chain(self, urdf_path, base_link, end_link):
        success, kdl_tree = treeFromUrdfModel(self.robot)
        if not success:
            raise RuntimeError("Failed to parse URDF into KDL tree.")
        return kdl_tree.getChain(base_link, end_link)

    def compute_jacobian(self, chain, joint_positions):
        nj = chain.getNrOfJoints()
        joint_array = PyKDL.JntArray(nj)
        for i in range(nj):
            joint_array[i] = joint_positions[i]
        jacobian = PyKDL.Jacobian(nj)
        solver = PyKDL.ChainJntToJacSolver(chain)
        solver.JntToJac(joint_array, jacobian)
        return jacobian

    def damped_pseudo_inverse(self, J, damping=0.01):
        JT = J.T
        identity = np.eye(J.shape[0])
        return JT @ np.linalg.inv(J @ JT + damping**2 * identity)

    # --- Callbacks ---
    def joint_callback(self, msg):
        self.joint_positions = msg.position

    # --- FK ---
    def compute_fk(self, joint_positions):
        nj = self.kdl_chain.getNrOfJoints()
        q = PyKDL.JntArray(nj)
        for i in range(nj):
            q[i] = joint_positions[i]
        frame = PyKDL.Frame()
        self.fk_solver.JntToCart(q, frame)
        pos = np.array([frame.p.x(), frame.p.y(), frame.p.z()])
        rot = np.array([[frame.M[i, j] for j in range(3)] for i in range(3)])
        return pos, rot
    
    # --- Control Loop ---
    # def control_loop(self):       # Only jacobian pseudo inverse method
        self.loop_count += 1
        
        # Get current and target poses
        current_pos, current_rot = self.compute_fk(self.joint_positions)
        waypoint = self.waypoints[self.current_waypoint_index]
        goal_pos = np.array(waypoint["pos"])
        goal_rot = Rotation.from_quat(waypoint["quat"]).as_matrix()

        # --- Compute Errors ---
        pos_error = (goal_pos - current_pos)
        rot_error = (np.cross(current_rot[:, 0], goal_rot[:, 0]) +
                           np.cross(current_rot[:, 1], goal_rot[:, 1]) +
                           np.cross(current_rot[:, 2], goal_rot[:, 2]))
        
        twist_error = np.hstack((pos_error, rot_error))

        # --- If goal reached, move to next waypoint ---
        if np.linalg.norm(pos_error) < 0.05 and np.linalg.norm(rot_error) < 0.01:
            self.get_logger().info(f"Reached waypoint {self.current_waypoint_index + 1}")
            self.current_waypoint_index += 1
            return

        # --- Compute Jacobian and joint velocities ---
        jacobian = self.compute_jacobian(self.kdl_chain, self.joint_positions)
        J = np.array([[jacobian[i, j] for j in range(jacobian.columns())] for i in range(6)])
        J_pinv = self.damped_pseudo_inverse(J)
        Kp = 1.5
        joint_velocities = Kp * (J_pinv @ twist_error)
        max_vel = 0.5  # rad/s per joint
        joint_velocities = np.clip(joint_velocities, -max_vel, max_vel)

        if self.loop_count % 200 == 0:
            print(f"Current waypoint: {self.current_waypoint_index}")
            print(f"Current joint angles: {self.joint_positions}")
            print(f"Joint velocities: {joint_velocities}")
            print(f"pos_error: {pos_error}, rot_error: {rot_error}")
            
        # --- Publish commands ---
        cmd_msg = Float64MultiArray()
        cmd_msg.data = joint_velocities.tolist()
        self.cmd_pub.publish(cmd_msg)

        state_msg = Float32()
        state_msg.data = 1.0
        self.state_pub.publish(state_msg)

    # def control_loop(self):       # Orientation-first control
        self.loop_count += 1

        if self.joint_positions is None:
            return  # wait for first joint state

        # --- Current FK ---
        current_pos, current_rot = self.compute_fk(self.joint_positions)
        waypoint = self.waypoints[self.current_waypoint_index]
        goal_pos = np.array(waypoint["pos"])
        goal_rot = Rotation.from_quat(waypoint["quat"]).as_matrix()

        # --- Compute full Jacobian ---
        jacobian = self.compute_jacobian(self.kdl_chain, self.joint_positions)
        J_full = np.array([[jacobian[i, j] for j in range(jacobian.columns())] for i in range(6)])

        # --- Compute position and rotation errors ---
        pos_error = goal_pos - current_pos
        rot_error_full = (np.cross(current_rot[:, 0], goal_rot[:, 0]) +
                        np.cross(current_rot[:, 1], goal_rot[:, 1]) +
                        np.cross(current_rot[:, 2], goal_rot[:, 2]))

        # --- Tolerances ---
        pos_tol = 0.1  # meters
        angle_tol = np.deg2rad(5)  # 5 degrees in radians

        # --- Orientation-first phase ---
        if not self.orientation_done:
            J_rot = J_full[3:6, :]

            # Weighted pseudo-inverse (orientation only)
            W = np.diag([1, 5, 5, 0.5, 1, 1])
            lambda_ = 0.02
            J_weighted = J_rot @ np.linalg.inv(W)
            J_rot_pinv = np.linalg.inv(J_weighted.T @ J_weighted + lambda_**2 * np.eye(J_weighted.shape[1])) @ J_weighted.T

            delta_joint = 1.5 * (J_rot_pinv @ rot_error_full.reshape(3, 1)).flatten()
            joint_velocities = delta_joint

            # Check orientation convergence
            if np.linalg.norm(rot_error_full) < angle_tol:
                self.orientation_done = True
                self.get_logger().info(f"Orientation aligned for waypoint {self.current_waypoint_index + 1}")

        else:
            # --- Full position + orientation correction (no joint weighting) ---
            J_pos = J_full[:3, :]
            J_rot = J_full[3:6, :]

            # Orientation correction
            dq_orient = np.linalg.pinv(J_rot) @ rot_error_full.reshape(3, 1)

            # Position correction (faster movement)
            pos_gain = 1.0  # proportional gain for position
            dq_pos = pos_gain * np.linalg.pinv(J_pos) @ pos_error.reshape(3, 1)

            # Combine: orientation high priority, position lower but faster than before
            delta_joint = dq_orient + dq_pos
            joint_velocities = delta_joint.flatten()

            # Check if waypoint reached (with tolerances)
            if np.linalg.norm(pos_error) < pos_tol and np.linalg.norm(rot_error_full) < angle_tol:
                self.get_logger().info(f"Reached waypoint {self.current_waypoint_index + 1}")
                self.current_waypoint_index += 1
                self.orientation_done = False
                return

        # --- Clip joint velocities ---
        max_vel = 0.5
        joint_velocities = np.clip(joint_velocities, -max_vel, max_vel)

        # --- Debug prints ---
        if self.loop_count % 200 == 0:
            print(f"Current waypoint: {self.current_waypoint_index}")
            print(f"Joint angles: {self.joint_positions}")
            print(f"Joint velocities: {joint_velocities}")
            print(f"Pos error: {pos_error}, Rot error: {rot_error_full}")

        # --- Publish commands ---
        cmd_msg = Float64MultiArray()
        cmd_msg.data = joint_velocities.tolist()
        self.cmd_pub.publish(cmd_msg)

        state_msg = Float32()
        state_msg.data = 1.0
        self.state_pub.publish(state_msg)

    # def control_loop(self):       # Priority-based control with high weight in orientation control
        self.loop_count += 1
        if self.joint_positions is None:
            return

        # --- Current state ---
        current_pos, current_rot = self.compute_fk(self.joint_positions)
        waypoint = self.waypoints[self.current_waypoint_index]
        goal_pos = np.array(waypoint["pos"])
        goal_rot = Rotation.from_quat(waypoint["quat"]).as_matrix()

        # --- Compute Jacobians ---
        jacobian = self.compute_jacobian(self.kdl_chain, self.joint_positions)
        J_full = np.array([[jacobian[i, j] for j in range(jacobian.columns())] for i in range(6)])
        J_pos = J_full[0:3, :]
        J_rot = J_full[3:6, :]

        # --- Compute errors ---
        pos_error = goal_pos - current_pos
        rot_error = (np.cross(current_rot[:, 0], goal_rot[:, 0]) +
                    np.cross(current_rot[:, 1], goal_rot[:, 1]) +
                    np.cross(current_rot[:, 2], goal_rot[:, 2]))

        # --- Weighted orientation control (minimum rotation energy) ---
        n_joints = J_rot.shape[1]
        # W = np.eye(n_joints)

        # # Optional: penalize joints near limits to avoid self-collision
        # for i in range(n_joints):
        #     margin = abs(self.joint_positions[i]) / np.pi
        #     W[i, i] = 1.0 + 2.0 * margin  # higher weight = less movement

        W = np.diag([1, 100, 100, 0.1, 1, 1])

        # Compute weighted pseudo-inverse for orientation
        JWJ = J_rot @ np.linalg.inv(W) @ J_rot.T
        if np.linalg.matrix_rank(JWJ) < 3:
            JWJ += np.eye(3) * 1e-4  # numerical stability
        J_rot_pinv = np.linalg.inv(W) @ J_rot.T @ np.linalg.inv(JWJ)

        # --- Null-space control for position ---
        U_p, S_p, Vh_p = np.linalg.svd(J_pos, full_matrices=False)
        S_p_inv = np.array([1/s if s > 0.01 else 0 for s in S_p])
        J_pos_pinv = Vh_p.T @ np.diag(S_p_inv) @ U_p.T

        # --- Priority-based control ---
        K_rot = 1.8
        K_pos = 0.4

        q_dot_rot = K_rot * (J_rot_pinv @ rot_error)
        null_space = np.eye(n_joints) - J_rot_pinv @ J_rot
        q_dot_pos = K_pos * (J_pos_pinv @ pos_error)

        joint_velocities = q_dot_rot + null_space @ q_dot_pos
        joint_velocities = np.clip(joint_velocities, -0.5, 0.5)

        # --- Check for completion ---
        pos_tol = 0.15  # meters
        rot_tol_deg = 5.0
        rot_tol = np.deg2rad(rot_tol_deg)

        if np.linalg.norm(pos_error) < pos_tol and np.linalg.norm(rot_error) < rot_tol:
            self.get_logger().info(
                f"Reached waypoint {self.current_waypoint_index + 1} "
                f"(pos_err={np.linalg.norm(pos_error):.3f}, rot_err={np.rad2deg(np.linalg.norm(rot_error)):.2f}°)"
            )
            self.current_waypoint_index += 1

                # --- Debug info ---
        if self.loop_count % 200 == 0:
            print(f"Current waypoint: {self.current_waypoint_index}")
            print(f"Joint angles: {self.joint_positions}")
            print(f"Joint velocities: {joint_velocities}")
            print(f"Pos error: {pos_error}, Rot error: {rot_error}")

        # --- Publish commands ---
        cmd_msg = Float64MultiArray()
        cmd_msg.data = joint_velocities.tolist()
        self.cmd_pub.publish(cmd_msg)

        state_msg = Float32()
        state_msg.data = 1.0
        self.state_pub.publish(state_msg)

    def control_loop(self):         # Priority-based control with different phases
        self.loop_count += 1
        if self.joint_positions is None:
            return

        # --- Current state ---
        current_pos, current_rot = self.compute_fk(self.joint_positions)
        waypoint = self.waypoints[self.current_waypoint_index]
        goal_pos = np.array(waypoint["pos"])
        goal_rot = Rotation.from_quat(waypoint["quat"]).as_matrix()

        # --- Jacobians ---
        jacobian = self.compute_jacobian(self.kdl_chain, self.joint_positions)
        J_full = np.array([[jacobian[i, j] for j in range(jacobian.columns())] for i in range(6)])
        J_pos = J_full[0:3, :]
        J_rot = J_full[3:6, :]

        # --- Errors ---
        pos_error = goal_pos - current_pos
        rot_error = (np.cross(current_rot[:, 0], goal_rot[:, 0]) +
                    np.cross(current_rot[:, 1], goal_rot[:, 1]) +
                    np.cross(current_rot[:, 2], goal_rot[:, 2]))

        # --- Thresholds / params ---
        pos_tol = 0.05
        rot_tol = np.deg2rad(10.0)
        orient_transition_thresh = np.deg2rad(15.0)
        orientation_error_norm = np.linalg.norm(rot_error)
        n_joints = J_rot.shape[1]
        max_vel = 0.5

        # --- Phase / gains ---
        if orientation_error_norm > orient_transition_thresh:
            weighted_phase = True
            K_rot = 5.0
            K_pos = 0.5
        else:
            weighted_phase = False
            K_rot = 1.2
            K_pos = 1.0

        # --- Orientation pseudo-inverse ---
        # Use moderate weights and Tikhonov regularization for stability.
        lambda_reg = 1e-4
        if weighted_phase:
            W = np.diag([1.0, 100.0, 100.0, 0.1, 1.0, 1.0])  # penalties only used here
            invW = np.linalg.inv(W)
            JWJ = J_rot @ invW @ J_rot.T
            JWJ += np.eye(3) * lambda_reg
            J_rot_pinv = invW @ J_rot.T @ np.linalg.inv(JWJ)
        else:
            # unweighted stable pseudo-inverse
            J_rot_pinv = np.linalg.pinv(J_rot)

        # --- Position pseudo-inverse (always unweighted) ---
        U_p, S_p, Vh_p = np.linalg.svd(J_pos, full_matrices=False)
        S_p_inv = np.array([1/s if s > 0.01 else 0.0 for s in S_p])
        J_pos_pinv = Vh_p.T @ np.diag(S_p_inv) @ U_p.T

        # --- Compute raw commands ---
        q_dot_rot_raw = (J_rot_pinv @ rot_error).flatten()
        q_dot_pos_raw = (J_pos_pinv @ pos_error).flatten()

        # --- Scale orientation command to avoid global saturation ---
        q_dot_rot = K_rot * q_dot_rot_raw

        # --- Make position secondary small during weighted phase (null-space) ---
        if weighted_phase:
            pos_scale = 0.8  # keep position drift slow while orienting
        else:
            pos_scale = 1.0

        pos_max_component = np.max(np.abs(q_dot_pos_raw)) + 1e-8
        pos_scale_final = pos_scale * min(1.0, (max_vel / (K_pos * pos_max_component)))
        q_dot_pos = K_pos * pos_scale_final * q_dot_pos_raw

        # --- Combine with strict null-space in weighted phase ---
        if weighted_phase:
            null_space = np.eye(n_joints) - J_rot_pinv @ J_rot
            joint_velocities = q_dot_rot + null_space @ q_dot_pos
        else:
            # equal priority additive
            joint_velocities = q_dot_rot + q_dot_pos

        # --- Final safety clip (small effect because of pre-scaling) ---
        joint_velocities = np.clip(joint_velocities, -max_vel, max_vel)

        # --- Completion check ---
        if np.linalg.norm(pos_error) < pos_tol and orientation_error_norm < rot_tol:
            self.get_logger().info(
                f"Reached waypoint {self.current_waypoint_index + 1} "
                f"(pos_err={np.linalg.norm(pos_error):.3f}, rot_err={np.rad2deg(orientation_error_norm):.2f}°)"
            )
            self.current_waypoint_index += 1

        # --- Debug ---
        if self.loop_count % 200 == 0:
            mode = "Orientation-weighted" if weighted_phase else "Equal-priority"
            print(f"Mode: {mode}")
            print(f"Current waypoint: {self.current_waypoint_index}")
            # print(f"Joint angles: {self.joint_positions}")
            print(f"Joint velocities: {joint_velocities}")
            print(f"Pos error: {pos_error}, Rot error: {rot_error}")
            print(f"Orientation error norm: {np.rad2deg(orientation_error_norm):.2f}°")

        # --- Publish ---
        cmd_msg = Float64MultiArray()
        cmd_msg.data = joint_velocities.tolist()
        self.cmd_pub.publish(cmd_msg)

        state_msg = Float32()
        state_msg.data = 1.0
        self.state_pub.publish(state_msg)


def main(args=None):
    rclpy.init(args=args)
    node = JointServoingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()