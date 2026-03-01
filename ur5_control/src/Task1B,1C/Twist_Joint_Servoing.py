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

# Additional imports for collision / TF / visualization utilities
import trimesh
import fcl
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PoseStamped, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray

package_path = get_package_share_directory("pymoveit2")
kdl_file_path = os.path.join(package_path, "examples", "kdl_parser_py.py")

# Dynamically import the file
spec = importlib.util.spec_from_file_location("kdl_parser_py", kdl_file_path)
kdl_parser_py = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kdl_parser_py)
treeFromUrdfModel = kdl_parser_py.treeFromUrdfModel

# ADJACENT_PAIRS to avoid checking trivial adjacent collisions (optional)
ADJACENT_PAIRS = {
    ("base_link_inertia", "shoulder_link"),
    ("base_link_inertia", "upper_arm_link"),
    ("base_link", "shoulder_link"),
    ("shoulder_link", "upper_arm_link"),
    ("upper_arm_link", "forearm_link"),
    ("forearm_link", "wrist_1_link"),
    ("wrist_1_link", "wrist_2_link"),
    ("wrist_2_link", "wrist_3_link"),
}

# Visualization plane constants
PLANE_SIZE = 1.0
PLANE_THICKNESS = 0.01
PLANE_ALPHA = 0.3


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

        # TF buffer & listener (used by helper get_link_pose)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Visualization publisher (optional)
        self.marker_pub = self.create_publisher(MarkerArray, '/visualization_marker_array', 10)

        # Data holders
        self.joint_positions = None
        self.current_waypoint_index = 0
        self.loop_count = 0
        self.initial_pos = None
        self.initial_rot = None
        self.stable_count = 0
        self.align_orientation_first = True
        self.orientation_done = False

        # Singularity / safety state (1 normal, 2 scaling, 3 halt)
        self.state = 1.0

        # Waypoints (positions + quaternions)
        self.waypoints = [
            {"pos": [-0.214, -0.532, 0.557], "quat": [0.707, 0.028, 0.034, 0.707]},
            {"pos": [0.120, -0.109,  0.445], "quat": [0.500, 0.497, 0.503, 0.499]},
            {"pos": [-0.159, 0.501, 0.415],  "quat": [0.029, 0.997, 0.045, 0.033]},
            {"pos": [-0.806, 0.010, 0.182],  "quat": [-0.684, 0.726, 0.05, 0.008]}
        ]

        # Workspace limits used to create boundary planes (tweak these)
        self.WORKSPACE_LIMITS = {
            'x_min': -0.90, 'x_max': 0.9,
            'y_min': -0.90, 'y_max': 0.9,
            'z_min': -0.5,   'z_max': 1.0
        }

        # Collision-related containers
        self.link_geometries = self.load_collision_geometries()
        self.link_names = list(self.link_geometries.keys())
        self.robot_collision_objects = {}
        self.create_robot_collision_objects()  # populate robot_collision_objects

        # Create boundary planes
        self.boundary_planes = self.create_boundary_planes()

        # Timer running control loop at 100 Hz
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

    # --- Collision / geometry helpers (new) ---
    def resolve_package_uri(self, uri):
        """Resolve package:// URIs in URDF meshes."""
        if uri.startswith("package://"):
            package_name, rel_path = uri[len("package://"):].split("/", 1)
            package_path_local = get_package_share_directory(package_name)
            return os.path.join(package_path_local, rel_path)
        return uri

    def load_collision_geometries(self):
        """Extract collision geometry objects from URDF (Mesh objects)."""
        geoms = {}
        for link in self.robot.links:
            if link.collision and link.collision.geometry:
                geoms[link.name] = link.collision.geometry
        return geoms

    def create_fcl_object(self, geometry, pose_matrix):
        """Create an FCL collision object from a URDF geometry (Mesh expected)."""
        # Only handle Mesh geometry here. You can extend for boxes/cylinders/spheres.
        from urdf_parser_py.urdf import Mesh
        if isinstance(geometry, Mesh):
            mesh_path = self.resolve_package_uri(geometry.filename)
            try:
                mesh = trimesh.load_mesh(mesh_path, force='mesh')
            except Exception as e:
                self.get_logger().warn(f"Failed to load mesh {mesh_path}: {e}")
                return None

            # Convert to arrays for FCL BVHModel
            vertices = np.asarray(mesh.vertices, dtype=float)
            # trimesh faces may be in Nx3
            triangles = np.asarray(mesh.faces, dtype=int)
            model = fcl.BVHModel()
            model.beginModel(len(vertices), len(triangles))
            model.addSubModel(vertices, triangles)
            model.endModel()
            transform = fcl.Transform(pose_matrix[0:3, 0:3], pose_matrix[0:3, 3])
            return fcl.CollisionObject(model, transform)
        else:
            return None

    def create_robot_collision_objects(self):
        """Create FCL collision objects for robot links once during initialization."""
        identity_pose = np.eye(4)
        for link_name, geometry in self.link_geometries.items():
            obj = self.create_fcl_object(geometry, identity_pose)
            if obj is not None:
                self.robot_collision_objects[link_name] = obj
                self.get_logger().info(f"Created collision object for link: {link_name}")
        self.get_logger().info(f"Created {len(self.robot_collision_objects)} robot collision objects")

    def update_collision_object_pose(self, collision_object, pose_matrix):
        """Update the transform of an existing FCL collision object."""
        new_transform = fcl.Transform(pose_matrix[0:3, 0:3], pose_matrix[0:3, 3])
        collision_object.setTransform(new_transform)

    def create_boundary_planes(self):
        """Create 6 boundary planes using FCL Plane primitives."""
        planes = []
        plane_configs = [
            ([-1, 0, 0], self.WORKSPACE_LIMITS['x_min']),  # Left wall
            ([1, 0, 0], self.WORKSPACE_LIMITS['x_max']),   # Right wall
            ([0, -1, 0], self.WORKSPACE_LIMITS['y_min']),  # Front wall
            ([0, 1, 0], self.WORKSPACE_LIMITS['y_max']),   # Back wall
            ([0, 0, -1], self.WORKSPACE_LIMITS['z_min']),  # Floor
            ([0, 0, 1], self.WORKSPACE_LIMITS['z_max']),   # Ceiling
        ]
        for normal, offset in plane_configs:
            n = np.array(normal, dtype=float)
            # fcl.Plane takes (normal, unsigned_distance). We use abs(offset).
            plane = fcl.Plane(n, abs(offset))
            transform = fcl.Transform()  # default identity
            plane_obj = fcl.CollisionObject(plane, transform)
            planes.append(plane_obj)
        self.get_logger().info(f"Created {len(planes)} boundary planes")
        return planes

    def publish_plane_markers(self):
        """Publish visualization markers for workspace boundary planes (optional)."""
        marker_array = MarkerArray()
        planes = [
            ([self.WORKSPACE_LIMITS['x_min'], 0, PLANE_SIZE/2.0], [PLANE_THICKNESS, PLANE_SIZE, PLANE_SIZE], [1, 0, 0, PLANE_ALPHA], "Left Wall"),
            ([self.WORKSPACE_LIMITS['x_max'], 0, PLANE_SIZE/2.0], [PLANE_THICKNESS, PLANE_SIZE, PLANE_SIZE], [1, 0, 0, PLANE_ALPHA], "Right Wall"),
            ([0, self.WORKSPACE_LIMITS['y_min'], PLANE_SIZE/2.0], [PLANE_SIZE, PLANE_THICKNESS, PLANE_SIZE], [0, 1, 0, PLANE_ALPHA], "Front Wall"),
            ([0, self.WORKSPACE_LIMITS['y_max'], PLANE_SIZE/2.0], [PLANE_SIZE, PLANE_THICKNESS, PLANE_SIZE], [0, 1, 0, PLANE_ALPHA], "Back Wall"),
            ([0, 0, self.WORKSPACE_LIMITS['z_min']],              [PLANE_SIZE, PLANE_SIZE, PLANE_THICKNESS], [0, 0, 1, PLANE_ALPHA], "Floor"),
            ([0, 0, self.WORKSPACE_LIMITS['z_max']],              [PLANE_SIZE, PLANE_SIZE, PLANE_THICKNESS], [0, 0, 1, PLANE_ALPHA], "Ceiling")
        ]
        for i, (pos, scale, color, name) in enumerate(planes):
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "boundary_planes"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = float(pos[0])
            marker.pose.position.y = float(pos[1])
            marker.pose.position.z = float(pos[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = float(scale[0])
            marker.scale.y = float(scale[1])
            marker.scale.z = float(scale[2])
            marker.color.r = float(color[0])
            marker.color.g = float(color[1])
            marker.color.b = float(color[2])
            marker.color.a = float(color[3])
            marker_array.markers.append(marker)
        # publish non-blocking
        self.marker_pub.publish(marker_array)

    def get_link_pose(self, target_frame, source_frame='base_link'):
        """Return 4x4 transform matrix of target_frame relative to source_frame using TF."""
        try:
            trans: TransformStamped = self.tf_buffer.lookup_transform(source_frame, target_frame, rclpy.time.Time())
            translation = [trans.transform.translation.x,
                           trans.transform.translation.y,
                           trans.transform.translation.z]
            rotation = [trans.transform.rotation.x,
                        trans.transform.rotation.y,
                        trans.transform.rotation.z,
                        trans.transform.rotation.w]
            r = Rotation.from_quat(rotation)
            matrix = np.eye(4)
            matrix[:3, :3] = r.as_matrix()
            matrix[0:3, 3] = translation
            return matrix
        except Exception as e:
            # TF may not be immediately available; that's ok
            return None

    def check_singularity(self, J, hard_stop_threshold=75.0, lower_threshold=50.0):
        """Return state code based on Jacobian condition number:
           3.0 -> halt, 2.0 -> scale, 1.0 -> normal.
           thresholds tuned for typical UR5; adjust as needed.
        """
        try:
            _, singular_vals, _ = np.linalg.svd(J)
            # avoid divide by zero
            if np.min(singular_vals) <= 1e-12:
                cond_number = float('inf')
            else:
                cond_number = np.max(singular_vals) / np.min(singular_vals)
        except Exception:
            cond_number = float('inf')

        if cond_number > hard_stop_threshold:
            self.state = 3.0
            return 3.0
        elif cond_number > lower_threshold:
            self.state = 2.0
            return 2.0
        else:
            self.state = 1.0
            return 1.0

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

    # --- Testing helper (simple predicted-collision check) ---
    def testing_predicted_collision(self, joint_velocities, delta_t=0.1):
        """
        Predicts joint positions after delta_t and does a fast collision check
        against the boundary planes using the robot collision objects.
        Returns True if a predicted collision is found.
        """
        if self.joint_positions is None:
            return False
        current_positions = list(self.joint_positions)
        predicted_positions = [p + delta_t * v for p, v in zip(current_positions, joint_velocities)]

        # FK for predicted positions: compute transforms for all segments
        fk_solver = PyKDL.ChainFkSolverPos_recursive(self.kdl_chain)
        joint_array = PyKDL.JntArray(self.kdl_chain.getNrOfJoints())
        for i in range(min(len(predicted_positions), joint_array.rows())):
            joint_array[i] = predicted_positions[i]

        # collect predicted transforms for segments and update corresponding collision objects
        predicted_global_poses = {}
        for seg_idx in range(self.kdl_chain.getNrOfSegments()):
            frame = PyKDL.Frame()
            fk_solver.JntToCart(joint_array, frame, seg_idx + 1)
            pose = np.eye(4)
            for r in range(3):
                for c in range(3):
                    pose[r, c] = frame.M[r, c]
                pose[r, 3] = frame.p[r]
            link_name = self.kdl_chain.getSegment(seg_idx).getName()
            predicted_global_poses[link_name] = pose

        # Update collision object transforms for links that exist
        valid_collision_objects = []
        valid_links = []
        for link_name, obj in self.robot_collision_objects.items():
            if link_name in predicted_global_poses:
                self.update_collision_object_pose(obj, predicted_global_poses[link_name])
                valid_collision_objects.append(obj)
                valid_links.append(link_name)

        # Self-collision check (skip adjacent pairs)
        for i in range(len(valid_collision_objects)):
            for j in range(i + 1, len(valid_collision_objects)):
                l1 = valid_links[i]
                l2 = valid_links[j]
                if (l1, l2) in ADJACENT_PAIRS or (l2, l1) in ADJACENT_PAIRS:
                    continue
                req = fcl.CollisionRequest()
                res = fcl.CollisionResult()
                fcl.collide(valid_collision_objects[i], valid_collision_objects[j], req, res)
                if res.is_collision:
                    self.get_logger().warn(f"Predicted self-collision between {l1} and {l2}")
                    return True

        # Collision with boundary planes
        for i, link_obj in enumerate(valid_collision_objects):
            link_name = valid_links[i]
            if 'base' in link_name:
                continue
            for j, plane in enumerate(self.boundary_planes):
                req = fcl.CollisionRequest()
                res = fcl.CollisionResult()
                fcl.collide(link_obj, plane, req, res)
                if res.is_collision:
                    self.get_logger().warn(f"Predicted collision between {link_name} and plane {j}")
                    return True

        return False

    # --- Control Loop ---
    def control_loop(self):
        self.loop_count += 1

        if self.joint_positions is None:
            self.get_logger().info_once('Waiting for joint states...')
            return

        # --- 1. Get current and target poses ---
        current_pos, current_rot = self.compute_fk(self.joint_positions)
        waypoint = self.waypoints[self.current_waypoint_index]
        goal_pos = np.array(waypoint["pos"])
        pos_error = (goal_pos - current_pos)

        # Normalize and convert goal quaternion ([x,y,z,w])
        q = np.array(waypoint["quat"], dtype=float)
        q /= np.linalg.norm(q)
        goal_rot = Rotation.from_quat(q).as_matrix()

        # --- 2. Compute relative rotation (goal wrt current) ---
        R_rel = goal_rot @ current_rot.T
        rotvec_full = Rotation.from_matrix(R_rel).as_rotvec()
        angle_full = np.linalg.norm(rotvec_full)
        if angle_full < 1e-6:
            rotvec_full = np.zeros(3)

        # --- 3. Stepwise alignment (Z-axis first) ---
        current_z = current_rot[:, 2]
        goal_z = goal_rot[:, 2]
        z_dot = np.clip(np.dot(current_z, goal_z), -1.0, 1.0)
        z_angle = np.arccos(z_dot)

        if self.align_orientation_first and not self.orientation_done:
            pos_error = np.zeros(3)  # freeze translation
            if z_angle < 0.02:
                self.get_logger().info("Z-axis alignment done. Switching to full orientation + position control.")
                self.orientation_done = True
                rot_error = np.zeros(3)
            else:
                axis = np.cross(current_z, goal_z)
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 1e-8:
                    axis = axis / axis_norm
                    rot_error = axis * z_angle
                else:
                    perp = np.array([1.0, 0.0, 0.0])
                    if abs(np.dot(perp, current_z)) > 0.999:
                        perp = np.array([0.0, 1.0, 0.0])
                    axis = np.cross(current_z, perp)
                    axis = axis / np.linalg.norm(axis)
                    rot_error = axis * z_angle
        else:
            rot_error = rotvec_full

        # --- 4. Convert errors to end-effector frame ---
        # Transform both translational and rotational errors from base to EE frame
        pos_error_local = current_rot.T @ pos_error
        rot_error_local = current_rot.T @ rot_error
        twist_error = np.hstack((pos_error_local, rot_error_local))

        # --- 5. Check goal reached ---
        if np.linalg.norm(pos_error) < 0.05 and np.linalg.norm(rot_error) < 0.01:
            self.get_logger().info(f"Reached waypoint {self.current_waypoint_index + 1}")
            self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.waypoints)
            self.orientation_done = False
            return

        # --- 6. Compute Jacobian & joint velocities ---
        jacobian = self.compute_jacobian(self.kdl_chain, self.joint_positions)
        J = np.array([[jacobian[i, j] for j in range(jacobian.columns())] for i in range(6)])
        singularity_state = self.check_singularity(J)
        J_pinv = self.damped_pseudo_inverse(J)

        # --- 7. Compute joint velocities ---
        Kp = 1.5
        joint_velocities = Kp * (J_pinv @ twist_error)

        # --- 8. Collision prediction ---
        collision_predicted = self.testing_predicted_collision(joint_velocities, delta_t=0.1)
        if collision_predicted:
            self.get_logger().warn("Predicted collision detected. Halting motion.")
            joint_velocities = np.zeros_like(joint_velocities)

        # --- 9. Clip and publish ---
        max_vel = 0.5
        joint_velocities = np.clip(joint_velocities, -max_vel, max_vel)

        if self.loop_count % 100 == 0:
            self.get_logger().info(f"Current waypoint: {self.current_waypoint_index}")
            self.get_logger().info(f"Current joint angles: {self.joint_positions}")
            self.get_logger().info(f"Joint velocities: {joint_velocities}")
            self.get_logger().info(f"pos_error_local: {pos_error_local}, rot_error_local: {rot_error_local}")

        cmd_msg = Float64MultiArray()
        cmd_msg.data = joint_velocities.tolist()
        self.cmd_pub.publish(cmd_msg)

        state_msg = Float32()
        state_msg.data = float(self.state)
        self.state_pub.publish(state_msg)

        if self.loop_count % 100 == 0:
            try:
                self.publish_plane_markers()
            except Exception:
                pass

def main(args=None):
    rclpy.init(args=args)
    node = JointServoingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
