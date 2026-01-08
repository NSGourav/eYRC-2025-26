#!/usr/bin/env python3
"""
Shape detector using Hough line detection
Cleaned and simplified version with unnecessary code removed
"""
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import tf2_ros
from tf2_ros import TransformException
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class ShapeDetection:
    """Data structure for a detected shape"""
    local_position: np.ndarray
    world_position: np.ndarray  # Actual position (for duplicate checking)
    goal_position: np.ndarray   # Navigation goal (with safety offset for visualization)
    shape_type: str
    distance: float
    plant_id: Optional[int]


class HoughLineDetector(Node):
    # Detection constants
    SHAPE_STATUS_MAP = {
        "Triangle": "FERTILIZER_REQUIRED",
        "Square": "BAD_HEALTH",
        "Pentagon": "DOCK_STATION"
    }

    # Angle tolerances for shape detection (degrees)
    TRIANGLE_ANGLE_40 = (140, 10)  # (target, tolerance) - increased tolerance
    TRIANGLE_ANGLE_130 = (130, 10)
    SQUARE_ANGLE = (90, 10)  # increased tolerance
    PENTAGON_ANGLES = [(90, 8), (140, 8)]

    # Distance thresholds
    CORNER_GAP_MAX = 0.06  # Max gap between connected lines - increased for better detection
    MAX_TRIANGLE_LENGTH = 0.25
    MAX_PENTAGON_SQUARE_LENGTH = 0.35

    # Safety offset to prevent robot collision with shape
    SAFETY_OFFSET_Y = 0.25  # 20cm offset in y-direction (perpendicular to robot's approach)
    SAFETY_OFFSET_X_SQUARE = 0.10  # 10cm offset in x-direction for Square (push away from robot)

    def __init__(self):
        super().__init__('hough_line_detector')

        # Scan filtering parameters
        self.min_range = 0.1
        self.max_range = 3.0
        self.median_filter_size = 5

        # Hough parameters
        self.rho_resolution = 0.005
        self.theta_resolution = np.pi / 360
        self.hough_threshold = 12
        self.min_line_length = 0.08
        self.max_line_gap = 0.08  # Increased to 5cm to merge broken segments

        # Minimum line length to filter noise (3cm)
        self.min_valid_line_length = 0.02

        # Image conversion
        self.pixels_per_meter = 300
        self.image_size = 800

        # Line processing
        self.merge_angle_tolerance = 10
        self.merge_distance_tolerance = 0.05  # 5cm - merge collinear lines with small gaps

        # Detection tracking
        self.detected_shapes: List[ShapeDetection] = []
        self.detection_range = 1.0  # Minimum 1m separation between shapes
        self.pending_shape: Optional[Tuple[str, int]] = None  # (shape_status, plant_id)

        # Robot state
        self.position_x = 0.0
        self.position_y = 0.0
        self.yaw = 0.0

        # Laser scanner offset from base_link (cached from TF)
        self.laser_offset_x = 0.0
        self.laser_offset_y = 0.0
        self.laser_offset_cached = False

        # TF buffer for getting laser offset
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Detection flow control
        self.detection_paused = False

        # ROS2 parameters
        self.declare_parameter('enable_visualization', False)
        self.declare_parameter('debug_mode', False)
        self.enable_visualization = self.get_parameter('enable_visualization').value
        self.debug_mode = self.get_parameter('debug_mode').value

        # Set logging level based on debug mode
        if self.debug_mode:
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)

        # ROS2 communication
        self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.shape_pub = self.create_publisher(String, '/detection_status', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/shape_markers', 10)
        self.shape_pose_pub = self.create_publisher(String, '/shape_pose', 10)  # Priority navigation

        # Visualization setup
        if self.enable_visualization:
            plt.ion()
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(18, 9))
            self.get_logger().info('Hough Line Detector started with visualization ENABLED')
        else:
            self.fig = None
            self.ax1 = None
            self.ax2 = None
            self.get_logger().info('Hough Line Detector started with visualization DISABLED')

    def odom_callback(self, msg: Odometry):
        """Update robot position from odometry"""
        self.position_x = msg.pose.pose.position.x
        self.position_y = msg.pose.pose.position.y

        # Extract yaw from quaternion
        quat = msg.pose.pose.orientation
        siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def scan_callback(self, msg: LaserScan):
        """Main callback for laser scan processing"""
        try:
            if self.detection_paused:
                return

            # Cache laser offset from TF (only done once)
            self._cache_laser_offset(msg.header.frame_id)

            # Get robot pose at scan timestamp using odometry
            scan_position_x, scan_position_y, scan_yaw = self._get_scan_pose(msg)

            # Process scan data
            all_points = self.scan_to_points(msg)
            if len(all_points) < 20:
                return

            filtered_points = self.filter_by_distance(all_points, self.max_range)
            if len(filtered_points) < 10:
                return

            # Detect lines using Hough transform
            binary_image = self.points_to_image(filtered_points)
            raw_lines = self.detect_lines(binary_image)
            if len(raw_lines) == 0:
                return

            # Group nearby lines and find best shape candidate
            line_groups = self.group_by_proximity(raw_lines, 0.15)
            best_group = self._find_best_group(line_groups)

            if not best_group:
                return

            # Determine shape type and position
            shape_type = self.detect_shape(best_group)
            shape_position = self._calculate_shape_position(best_group, shape_type)
            shape_distance = np.linalg.norm(shape_position)

            # Log detection summary
            if shape_type:
                self.get_logger().info(
                    f'Shape candidate: {shape_type} | '
                    f'Lines: {len(best_group)} | '
                    f'Local pos: ({shape_position[0]:.3f}, {shape_position[1]:.3f}) | '
                    f'Distance: {shape_distance:.3f}m'
                )

            # Process valid detections
            if shape_type in ['Pentagon', 'Square', 'Triangle']:
                # Log robot pose being used for transformation
                self.get_logger().info(
                    f'Robot pose for transform: x={scan_position_x:.3f}, y={scan_position_y:.3f}, '
                    f'yaw={math.degrees(scan_yaw):.1f}°'
                )

                is_new, world_pos = self.is_new_detection(
                    shape_position, scan_position_x, scan_position_y, scan_yaw
                )

                if is_new:
                    plant_id = self.assign_plant_id(world_pos, scan_position_x, scan_position_y)
                    shape_status = self.SHAPE_STATUS_MAP.get(shape_type)

                    # Calculate goal position with safety offset for navigation/visualization
                    if shape_type in ['Square', 'Triangle']:
                        goal_world_pos = self.local_to_world(
                            shape_position, scan_position_x, scan_position_y, scan_yaw,
                            apply_safety=True, shape_type=shape_type
                        )
                    else:
                        # Pentagon (dock) - no offset needed
                        goal_world_pos = world_pos

                    # Store detection with both actual and goal positions
                    detection = ShapeDetection(
                        local_position=shape_position,
                        world_position=world_pos,      # Actual position (for duplicate checking)
                        goal_position=goal_world_pos,  # Goal position (for navigation & visualization)
                        shape_type=shape_type,
                        distance=shape_distance,
                        plant_id=plant_id
                    )
                    self.detected_shapes.append(detection)

                    # Log detection info
                    if shape_type in ['Square', 'Triangle']:
                        self.get_logger().info(
                            f'✓ NEW {shape_type} DETECTED:\n'
                            f'  Local:  ({shape_position[0]:.3f}, {shape_position[1]:.3f}) m\n'
                            f'  Actual: ({world_pos[0]:.3f}, {world_pos[1]:.3f}) m\n'
                            f'  Goal:   ({goal_world_pos[0]:.3f}, {goal_world_pos[1]:.3f}) m (with {self.SAFETY_OFFSET_Y}m safety offset)\n'
                            f'  Robot:  ({scan_position_x:.3f}, {scan_position_y:.3f}, {math.degrees(scan_yaw):.1f}°)\n'
                            f'  Plant ID: {plant_id}\n'
                            f'  Total detections: {len(self.detected_shapes)}'
                        )
                    else:
                        self.get_logger().info(
                            f'✓ NEW {shape_type} DETECTED:\n'
                            f'  Local:  ({shape_position[0]:.3f}, {shape_position[1]:.3f}) m\n'
                            f'  World:  ({world_pos[0]:.3f}, {world_pos[1]:.3f}) m\n'
                            f'  Robot:  ({scan_position_x:.3f}, {scan_position_y:.3f}, {math.degrees(scan_yaw):.1f}°)\n'
                            f'  Plant ID: {plant_id}\n'
                            f'  Total detections: {len(self.detected_shapes)}'
                        )

                    # Publish shape pose for priority navigation (Square and Triangle only)
                    if shape_type in ['Square', 'Triangle']:
                        pose_msg = String()
                        pose_msg.data = f"{shape_type},{goal_world_pos[0]:.3f},{goal_world_pos[1]:.3f}"
                        self.shape_pose_pub.publish(pose_msg)
                        self.get_logger().info(f"Published shape pose: {pose_msg.data}")

                        self.pending_shape = (shape_status, plant_id)
                    else:
                        self.get_logger().info(f"{shape_type} detected (dock station - no intermediate goal)")

                    # Publish updated markers to RViz
                    self.publish_shape_markers()
                else:
                    self.get_logger().info(
                        f'✗ DUPLICATE {shape_type} ignored (within 1.0m of existing detection)\n'
                        f'  Current world pos: ({world_pos[0]:.3f}, {world_pos[1]:.3f})'
                    )
                    shape_type = f'{shape_type} (Already Detected)'

            self.visualize(filtered_points, binary_image, best_group, shape_type, shape_position)

        except Exception as e:
            self.get_logger().error(f"Error in scan_callback: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def _cache_laser_offset(self, scan_frame: str):
        """Cache the static transform from base_link to laser scanner (only called once)"""
        if self.laser_offset_cached:
            return

        try:
            # Get static transform from base_link to laser frame
            transform = self.tf_buffer.lookup_transform(
                'ebot_base_link',
                scan_frame,
                Time(),
                timeout=Duration(seconds=1.0)
            )

            self.laser_offset_x = transform.transform.translation.x
            self.laser_offset_y = transform.transform.translation.y
            self.laser_offset_cached = True

            self.get_logger().info(
                f'Laser offset cached: x={self.laser_offset_x:.3f}m, y={self.laser_offset_y:.3f}m '
                f'(from {scan_frame} to ebot_base_link)'
            )

        except TransformException as ex:
            self.get_logger().warn(f'Could not get laser offset, assuming 0: {ex}', throttle_duration_sec=5.0)
            self.laser_offset_x = 0.0
            self.laser_offset_y = 0.0
            # Don't set cached=True, will retry next time

    def _get_scan_pose(self, msg: LaserScan) -> Tuple[float, float, float]:
        """Get robot pose - using odometry for consistency"""
        # Use current odometry directly - more reliable than TF for this application
        # TF can have timing issues and extrapolation problems
        return self.position_x, self.position_y, self.yaw

    def _find_best_group(self, line_groups: List[List[Dict]]) -> Optional[List[Dict]]:
        """Find the best line group based on scoring"""
        best_group = None
        best_score = -1

        for group in line_groups:
            processed = self.process_lines(group)
            score = self.score_group(processed)
            if score > best_score:
                best_score = score
                best_group = processed

        return best_group

    def _calculate_shape_position(self, lines: List[Dict], shape_type: str) -> np.ndarray:
        """Calculate shape position based on type - returns the nearest corner/point to robot"""
        all_endpoints = np.array([[l['start'], l['end']] for l in lines]).reshape(-1, 2)

        if shape_type == 'Triangle':
            # For triangle: find the corner vertex (point where two lines meet)
            if len(lines) == 2:
                # 2-line triangle: use the corner between the two lines
                corner = self.compute_corner(lines[0], lines[1])
                self.get_logger().debug(f'Triangle position (2-line corner): ({corner[0]:.3f}, {corner[1]:.3f})')
                return corner
            elif len(lines) >= 4:
                # 4-line triangle: use corner between lines 1 and 2 (middle vertex)
                corner = self.compute_corner(lines[1], lines[2])
                self.get_logger().debug(f'Triangle position (4-line corner): ({corner[0]:.3f}, {corner[1]:.3f})')
                return corner

        elif shape_type == 'Square':
            # For square: use the nearest corner to robot (which is at origin in local frame)
            if len(lines) >= 3:
                corner = self.compute_corner(lines[1], lines[2])
                self.get_logger().debug(f'Square position (corner): ({corner[0]:.3f}, {corner[1]:.3f})')
                return corner

        # Default: use centroid of all line endpoints
        centroid = np.mean(all_endpoints, axis=0)
        self.get_logger().debug(f'Shape position (centroid): ({centroid[0]:.3f}, {centroid[1]:.3f})')
        return centroid

    def detect_shape(self, lines: List[Dict]) -> Optional[str]:
        """Detect shape type from line segments"""
        num_lines = len(lines)
        self.get_logger().debug(f'Analyzing {num_lines} lines for shape detection')

        # PRIORITY 1: Check 3-line shapes FIRST (Pentagon/Square) to avoid misclassification
        if num_lines == 3:
            angles = [180 - self.calc_angle(lines[i], lines[i+1]) for i in range(num_lines-1)]
            lengths = [l['length'] for l in lines]

            # Check corner gaps
            gaps = []
            for i in range(num_lines-1):
                gap = math.sqrt((lines[i]['end'][0] - lines[i+1]['start'][0])**2 +
                               (lines[i]['end'][1] - lines[i+1]['start'][1])**2)
                gaps.append(gap)

            self.get_logger().info(f'3-line candidate: angles={[f"{a:.1f}" for a in angles]}°, '
                                  f'lengths={[f"{l:.3f}" for l in lengths]}m, gaps={[f"{g:.3f}" for g in gaps]}m')

            # Pentagon: 90° and 140° angles
            if (abs(angles[0] - 90) < 7 and abs(angles[1] - 140) < 7 and
                all(g < self.CORNER_GAP_MAX for g in gaps)):
                self.get_logger().info(f'✓ Pentagon detected (3 lines, angles={[f"{a:.1f}" for a in angles]}°)')
                return 'Pentagon'

            # Square: all 90° angles
            if (all(abs(a - 90) < 10 for a in angles) and
                all(g < self.CORNER_GAP_MAX for g in gaps) and
                lines[1]['length'] < self.MAX_PENTAGON_SQUARE_LENGTH and
                lines[2]['length'] < self.MAX_PENTAGON_SQUARE_LENGTH):
                self.get_logger().info(f'✓ Square detected (3 lines, angles={[f"{a:.1f}" for a in angles]}°)')
                return 'Square'

        if num_lines == 2:
            angle = self.calc_angle(lines[0], lines[1])
            angle_inv = 180 - angle
            corner_gap = math.sqrt((lines[0]['end'][0] - lines[1]['start'][0])**2 +
                                  (lines[0]['end'][1] - lines[1]['start'][1])**2)

            L1 = lines[0]['length']
            L2 = lines[1]['length']

            self.get_logger().info(f'2-line: angle={angle_inv:.1f}°, L1={L1:.3f}m, L2={L2:.3f}m, gap={corner_gap:.3f}m')

            # Triangle criteria
            TRIANGLE_ANGLE = 130.0
            TRIANGLE_ANGLE_TOL = 8.0
            TRIANGLE_L2_TARGET = 0.22  # 22cm
            TRIANGLE_L2_TOL = 0.02     # ±2cm

            angle_ok = abs(angle_inv - TRIANGLE_ANGLE) < TRIANGLE_ANGLE_TOL
            length_ok = abs(L2 - TRIANGLE_L2_TARGET) < TRIANGLE_L2_TOL
            gap_ok = corner_gap < self.CORNER_GAP_MAX

            if angle_ok and length_ok and gap_ok:
                self.get_logger().info(f'✓ Triangle detected: angle={angle_inv:.1f}°, L2={L2:.3f}m')
                return 'Triangle'

        self.get_logger().debug(f'{num_lines} lines did not match any shape pattern')
        return None

    def assign_plant_id(self, shape_world_pos: np.ndarray, robot_x: float, robot_y: float) -> Optional[int]:
        """
        Assign plant_id (0-8) based on lane and segment

        Simplified version - returns None if position doesn't match expected ranges.
        For production, consider loading this from a config file.
        """
        # Define lane boundaries (simplified - could be config file)
        lanes = {
            1: {'x_range': (0.2, 0.35), 'right_id': 0, 'left_segments': [((-4.67, -3.374), 1), ((-3.374, -2.020), 2), ((-2.020, -0.720), 3), ((-0.720, 0.552), 4)]},
            2: {'x_range': (-1.45, -1.3), 'right_segments': [((-4.757, -3.381), 1), ((-3.381, -2.057), 2), ((-2.057, -0.711), 3), ((-0.711, 0.553), 4)], 'left_segments': [((-4.757, -3.381), 5), ((-3.381, -2.057), 6), ((-2.057, -0.711), 7), ((-0.711, 0.553), 8)]},
            3: {'x_range': (-3.5, -3.35), 'right_segments': [((-4.716, -3.456), 5), ((-3.456, -2.079), 6), ((-2.079, -0.723), 7), ((-0.723, 0.527), 8)]}
        }

        # Find current lane
        current_lane = None
        for lane_num, lane_data in lanes.items():
            x_min, x_max = lane_data['x_range']
            if x_min <= robot_x <= x_max:
                current_lane = lane_num
                break

        if current_lane is None:
            self.get_logger().warn(f"Robot x={robot_x:.3f} not in any lane")
            return None

        is_left_side = shape_world_pos[0] < robot_x

        # Lane 1: Dock on right, segments on left
        if current_lane == 1:
            if not is_left_side:
                self.get_logger().info("Plant ID: 0 (Dock - Lane 1 Right)")
                return 0
            else:
                for (y_min, y_max), plant_id in lanes[1]['left_segments']:
                    if y_min <= robot_y <= y_max:
                        self.get_logger().info(f"Plant ID: {plant_id} (Lane 1 Left, y:[{y_min}, {y_max}])")
                        return plant_id

        # Lane 2: Both sides
        elif current_lane == 2:
            segments = lanes[2]['right_segments'] if not is_left_side else lanes[2]['left_segments']
            for (y_min, y_max), plant_id in segments:
                if y_min <= robot_y <= y_max:
                    side = "Right" if not is_left_side else "Left"
                    self.get_logger().info(f"Plant ID: {plant_id} (Lane 2 {side}, y:[{y_min}, {y_max}])")
                    return plant_id

        # Lane 3: Right side only
        elif current_lane == 3 and not is_left_side:
            for (y_min, y_max), plant_id in lanes[3]['right_segments']:
                if y_min <= robot_y <= y_max:
                    self.get_logger().info(f"Plant ID: {plant_id} (Lane 3 Right, y:[{y_min}, {y_max}])")
                    return plant_id

        self.get_logger().warn(f"Could not assign plant_id - robot at ({robot_x:.3f}, {robot_y:.3f}) in Lane {current_lane}")
        return None

    def apply_safety_offset(self, local_pos: np.ndarray, shape_type: str = None) -> np.ndarray:
        """Apply safety offset to shape position to prevent collision

        Y-offset: Shifts goal toward robot's centerline by SAFETY_OFFSET_Y
        - If shape is on left (negative y), add positive offset (move right toward robot)
        - If shape is on right (positive y), add negative offset (move left toward robot)

        X-offset: For Square only, push forward by SAFETY_OFFSET_X_SQUARE (away from robot)
        - Adds to X to move goal further away from robot
        """
        offset_y = -np.sign(local_pos[1]) * self.SAFETY_OFFSET_Y if local_pos[1] != 0 else 0.0
        offset_x = 0.0

        # For Square: add X offset away from robot (positive X direction - push forward)
        if shape_type == 'Square':
            offset_x = self.SAFETY_OFFSET_X_SQUARE

        adjusted_pos = np.array([local_pos[0] + offset_x, local_pos[1] + offset_y])

        self.get_logger().debug(
            f'Safety offset applied ({shape_type}): ({local_pos[0]:.3f}, {local_pos[1]:.3f}) -> '
            f'({adjusted_pos[0]:.3f}, {adjusted_pos[1]:.3f}) [offset_x={offset_x:.3f}m, offset_y={offset_y:.3f}m]'
        )

        return adjusted_pos

    def local_to_world(self, local_pos: np.ndarray, robot_x: float = None,
                      robot_y: float = None, robot_yaw: float = None,
                      apply_safety: bool = False, shape_type: str = None) -> np.ndarray:
        """Convert local laser frame coordinates to world coordinates

        Steps:
        1. local_pos is in laser scanner frame
        2. Optionally apply safety offset
        3. Add laser offset to get position in base_link frame
        4. Rotate and translate to get world frame position
        """
        robot_x = robot_x if robot_x is not None else self.position_x
        robot_y = robot_y if robot_y is not None else self.position_y
        robot_yaw = robot_yaw if robot_yaw is not None else self.yaw

        # Make a copy to avoid mutating the original
        local_pos = np.copy(local_pos)

        # Apply safety offset if requested (for navigation goals)
        if apply_safety:
            local_pos = self.apply_safety_offset(local_pos, shape_type=shape_type)

        # Step 1: Transform from laser frame to base_link frame
        # Add the laser offset (laser is ahead/to-side of base_link)
        base_link_x = local_pos[0] + self.laser_offset_x
        base_link_y = local_pos[1] + self.laser_offset_y

        # Step 2: Transform from base_link frame to world frame
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)

        world_x = robot_x + (base_link_x * cos_yaw - base_link_y * sin_yaw)
        world_y = robot_y + (base_link_x * sin_yaw + base_link_y * cos_yaw)

        self.get_logger().debug(
            f'Transform: laser({local_pos[0]:.3f}, {local_pos[1]:.3f}) -> '
            f'base_link({base_link_x:.3f}, {base_link_y:.3f}) -> '
            f'world({world_x:.3f}, {world_y:.3f}) | '
            f'robot@({robot_x:.3f}, {robot_y:.3f}, yaw={math.degrees(robot_yaw):.1f}°) | '
            f'laser_offset=({self.laser_offset_x:.3f}, {self.laser_offset_y:.3f})'
        )

        return np.array([world_x, world_y])

    def compute_corner(self, line1: Dict, line2: Dict) -> np.ndarray:
        """Compute intersection point (corner) of two connected lines"""
        endpoints1 = [line1['start'], line1['end']]
        endpoints2 = [line2['start'], line2['end']]

        min_dist = float('inf')
        corner = None

        for p1 in endpoints1:
            for p2 in endpoints2:
                d = np.linalg.norm(p1 - p2)
                if d < min_dist:
                    min_dist = d
                    corner = (p1 + p2) / 2

        return corner

    def is_new_detection(self, local_position: np.ndarray, robot_x: float = None,
                        robot_y: float = None, robot_yaw: float = None) -> Tuple[bool, np.ndarray]:
        """Check if shape at local position is new based on world coordinates"""
        world_position = self.local_to_world(local_position, robot_x, robot_y, robot_yaw)

        for detected in self.detected_shapes:
            if np.linalg.norm(world_position - detected.world_position) < self.detection_range:
                return False, world_position

        return True, world_position

    # ========================================================================
    # SCAN PROCESSING METHODS
    # ========================================================================

    def filter_by_distance(self, points: np.ndarray, max_dist: float) -> np.ndarray:
        """Filter points by maximum distance"""
        return points[np.linalg.norm(points, axis=1) <= max_dist]

    def scan_to_points(self, msg: LaserScan) -> np.ndarray:
        """Convert laser scan to 2D points"""
        ranges = np.array(msg.ranges, dtype=float)
        ranges[np.isnan(ranges) | np.isinf(ranges)] = self.max_range + 1
        filtered = self.median_filter(ranges)

        points = []
        for i, r in enumerate(filtered):
            if self.min_range <= r <= self.max_range:
                angle = msg.angle_min + i * msg.angle_increment
                points.append([r * np.cos(angle), r * np.sin(angle)])

        return np.array(points) if points else np.array([]).reshape(0, 2)

    def median_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply median filter to range data"""
        filtered = np.copy(data)
        half = self.median_filter_size // 2

        for i in range(len(data)):
            window = data[max(0, i-half):min(len(data), i+half+1)]
            valid = window[window <= self.max_range]
            if len(valid) > 0:
                filtered[i] = np.median(valid)

        return filtered

    def points_to_image(self, points: np.ndarray) -> np.ndarray:
        """Convert points to binary image"""
        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        center = self.image_size // 2

        for pt in points:
            px = int(pt[0] * self.pixels_per_meter + center)
            py = int(-pt[1] * self.pixels_per_meter + center)
            if 0 <= px < self.image_size and 0 <= py < self.image_size:
                cv2.circle(image, (px, py), 2, 255, -1)

        return image

    def detect_lines(self, binary_image: np.ndarray) -> List[Dict]:
        """Detect lines using Hough transform"""
        lines_cv = cv2.HoughLinesP(
            binary_image,
            rho=self.rho_resolution * self.pixels_per_meter,
            theta=self.theta_resolution,
            threshold=self.hough_threshold,
            minLineLength=int(self.min_line_length * self.pixels_per_meter),
            maxLineGap=int(self.max_line_gap * self.pixels_per_meter)
        )

        if lines_cv is None:
            return []

        lines = []
        center = self.image_size // 2
        ppm = self.pixels_per_meter

        for line in lines_cv:
            x1, y1, x2, y2 = line[0]
            x1_m, y1_m = (x1-center)/ppm, -(y1-center)/ppm
            x2_m, y2_m = (x2-center)/ppm, -(y2-center)/ppm
            length = np.hypot(x2_m-x1_m, y2_m-y1_m)

            if length >= self.min_line_length:
                lines.append({
                    'start': np.array([x1_m, y1_m]),
                    'end': np.array([x2_m, y2_m]),
                    'length': length
                })

        return lines

    # ========================================================================
    # LINE PROCESSING METHODS (Grouping, Merging, Scoring)
    # ========================================================================

    def group_by_proximity(self, lines: List[Dict], threshold: float) -> List[List[Dict]]:
        """Group lines that are close to each other"""
        if not lines:
            return []

        groups = []
        used = set()

        for i, line in enumerate(lines):
            if i in used:
                continue

            group = [line]
            used.add(i)

            changed = True
            while changed:
                changed = False
                for j in range(len(lines)):
                    if j in used:
                        continue

                    for g_line in group:
                        endpoints_g = [g_line['start'], g_line['end']]
                        endpoints_j = [lines[j]['start'], lines[j]['end']]
                        min_dist = min(np.linalg.norm(p1-p2)
                                     for p1 in endpoints_g for p2 in endpoints_j)

                        if min_dist < threshold:
                            group.append(lines[j])
                            used.add(j)
                            changed = True
                            break

            groups.append(group)

        return groups

    def score_group(self, lines: List[Dict]) -> float:
        """Score a line group based on quality metrics"""
        if not lines:
            return 0

        score = 10.0 if 2 <= len(lines) <= 4 else (2.0 if len(lines) == 1 else 1.0)

        all_pts = np.array([[l['start'], l['end']] for l in lines]).reshape(-1, 2)
        bbox_size = np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0))

        if 0.15 < bbox_size < 0.8:
            score += 5.0 / (bbox_size + 0.1)

        if np.mean([l['length'] for l in lines]) < 0.3:
            score += 5.0

        score += max(0, 5.0 - np.linalg.norm(all_pts.mean(axis=0)))

        return score

    def process_lines(self, lines: List[Dict]) -> List[Dict]:
        """Merge collinear segments and remove overlaps"""
        if len(lines) <= 1:
            return lines

        # Filter out noise - lines shorter than 3cm
        lines = [l for l in lines if l['length'] >= self.min_valid_line_length]
        if len(lines) <= 1:
            return lines

        # Merge collinear lines
        merged = []
        used = set()

        for i, line in enumerate(lines):
            if i in used:
                continue

            group = [line]
            used.add(i)

            changed = True
            while changed:
                changed = False
                for j in range(len(lines)):
                    if j in used:
                        continue

                    for g_line in group:
                        if self.are_similar(g_line, lines[j]):
                            group.append(lines[j])
                            used.add(j)
                            changed = True
                            break

            merged.append(self.merge_group(group))

        # Remove overlaps
        sorted_lines = sorted(merged, key=lambda l: l['length'], reverse=True)
        filtered = []

        for line in sorted_lines:
            if not any(self.check_overlap(line, kept) for kept in filtered):
                filtered.append(line)

        return self.order_lines(filtered)

    def normalize_dir(self, line: Dict) -> Tuple[Optional[np.ndarray], float]:
        """Get normalized direction vector"""
        d = line['end'] - line['start']
        length = np.linalg.norm(d)
        return (d/length, length) if length > 1e-6 else (None, length)

    def are_similar(self, line1: Dict, line2: Dict) -> bool:
        """Check if lines should be merged"""
        dir1, len1 = self.normalize_dir(line1)
        dir2, len2 = self.normalize_dir(line2)

        if dir1 is None or dir2 is None:
            return False

        # Check angle
        if np.degrees(np.arccos(np.clip(np.abs(np.dot(dir1, dir2)), 0, 1))) > self.merge_angle_tolerance:
            return False

        # Check endpoint proximity
        eps1 = [line1['start'], line1['end']]
        eps2 = [line2['start'], line2['end']]
        min_dist = min(np.linalg.norm(p1-p2) for p1 in eps1 for p2 in eps2)

        # Check perpendicular distance
        mid2 = (line2['start'] + line2['end']) / 2
        to_mid = mid2 - line1['start']
        perp_dist = np.linalg.norm(to_mid - np.dot(to_mid, dir1) * dir1)

        # Merge if: endpoints are close OR lines are collinear with small gap
        return min_dist < self.merge_distance_tolerance or (perp_dist < 0.08 and min_dist < 0.40)

    def merge_group(self, group: List[Dict]) -> Dict:
        """Merge multiple lines into one"""
        all_pts = np.array([[l['start'], l['end']] for l in group]).reshape(-1, 2)

        max_dist = 0
        best_pair = (all_pts[0], all_pts[1])

        for i in range(len(all_pts)):
            for j in range(i+1, len(all_pts)):
                dist = np.linalg.norm(all_pts[i] - all_pts[j])
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (all_pts[i], all_pts[j])

        return {'start': best_pair[0], 'end': best_pair[1], 'length': max_dist}

    def check_overlap(self, line1: Dict, line2: Dict) -> bool:
        """Check if two lines overlap"""
        dir1, len1 = self.normalize_dir(line1)
        dir2, len2 = self.normalize_dir(line2)

        if dir1 is None or dir2 is None:
            return False

        # Check parallelism
        if np.abs(np.dot(dir1, dir2)) < 0.95:
            return False

        # Check perpendicular distance
        mid1 = (line1['start'] + line1['end']) / 2
        mid2 = (line2['start'] + line2['end']) / 2
        mid_vec = mid2 - mid1
        perp_dist = np.linalg.norm(mid_vec - np.dot(mid_vec, dir1) * dir1)

        if perp_dist < 0.03:
            # Check projection overlap
            ref = line1['start']
            projs = [np.dot(line1['start']-ref, dir1), np.dot(line1['end']-ref, dir1),
                    np.dot(line2['start']-ref, dir1), np.dot(line2['end']-ref, dir1)]
            min1, max1 = min(projs[0], projs[1]), max(projs[0], projs[1])
            min2, max2 = min(projs[2], projs[3]), max(projs[2], projs[3])

            if min(max1, max2) - max(min1, min2) > 0.02:
                return True

        # Check endpoint proximity
        eps1 = [line1['start'], line1['end']]
        eps2 = [line2['start'], line2['end']]
        close_count = sum(1 for p1 in eps1 for p2 in eps2 if np.linalg.norm(p1-p2) < 0.05)

        return close_count >= 3

    def calc_angle(self, line1: Dict, line2: Dict) -> float:
        """Calculate angle between two lines in degrees"""
        dir1, len1 = self.normalize_dir(line1)
        dir2, len2 = self.normalize_dir(line2)

        if dir1 is None or dir2 is None:
            return 0.0

        dot = np.clip(np.dot(dir1, dir2), -1.0, 1.0)
        return np.degrees(np.arccos(np.abs(dot)))

    def order_lines(self, lines: List[Dict]) -> List[Dict]:
        """Order lines by connectivity"""
        if len(lines) <= 2:
            return lines

        ordered = [lines[0]]
        remaining = set(range(1, len(lines)))

        while remaining:
            last = ordered[-1]
            best_idx = None
            best_dist = float('inf')

            for idx in remaining:
                distances = [
                    np.linalg.norm(last['end'] - lines[idx]['start']),
                    np.linalg.norm(last['end'] - lines[idx]['end']),
                    np.linalg.norm(last['start'] - lines[idx]['start']),
                    np.linalg.norm(last['start'] - lines[idx]['end'])
                ]

                min_dist = min(distances)
                if min_dist < best_dist:
                    best_dist = min_dist
                    best_idx = idx

            if best_idx is None or best_dist > 0.20:
                break

            ordered.append(lines[best_idx])
            remaining.remove(best_idx)

        return ordered

    # ========================================================================
    # RVIZ MARKER PUBLISHING
    # ========================================================================

    def publish_shape_markers(self):
        """Publish RViz markers for all detected shapes"""
        marker_array = MarkerArray()

        for idx, detection in enumerate(self.detected_shapes):
            # Create a marker for the shape
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "detected_shapes"
            marker.id = idx
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            # Set position (use stored goal position)
            marker.pose.position.x = float(detection.goal_position[0])
            marker.pose.position.y = float(detection.goal_position[1])
            marker.pose.position.z = 0.1  # Slightly above ground

            # Set orientation (upright)
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            # Set scale based on shape type
            marker.scale.x = 0.15  # diameter
            marker.scale.y = 0.15
            marker.scale.z = 0.2   # height

            # Set color based on shape type
            if detection.shape_type == "Triangle":
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0  # Yellow
                marker.color.a = 0.8
            elif detection.shape_type == "Square":
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0  # Red
                marker.color.a = 0.8
            elif detection.shape_type == "Pentagon":
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0  # Blue
                marker.color.a = 0.8
            else:
                marker.color.r = 0.5
                marker.color.g = 0.5
                marker.color.b = 0.5  # Gray
                marker.color.a = 0.8

            marker.lifetime.sec = 0  # 0 means forever

            marker_array.markers.append(marker)

            # Add text label with plant ID
            text_marker = Marker()
            text_marker.header.frame_id = "odom"
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = "shape_labels"
            text_marker.id = idx + 1000  # Offset to avoid ID collision
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            # Position text above the marker (using goal position)
            text_marker.pose.position.x = float(detection.goal_position[0])
            text_marker.pose.position.y = float(detection.goal_position[1])
            text_marker.pose.position.z = 0.4

            text_marker.pose.orientation.w = 1.0

            # Set text
            plant_id_str = str(detection.plant_id) if detection.plant_id is not None else "?"
            text_marker.text = f"{detection.shape_type}\nID: {plant_id_str}"

            text_marker.scale.z = 0.12  # Text height

            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0

            text_marker.lifetime.sec = 0

            marker_array.markers.append(text_marker)

        # Publish the marker array
        self.marker_pub.publish(marker_array)

    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    def visualize(self, points: np.ndarray, binary_image: np.ndarray,
                 lines: List[Dict], shape_type: str = 'Unknown',
                 shape_position: Optional[np.ndarray] = None):
        """Visualize detection results"""
        if not self.enable_visualization or self.fig is None:
            return

        self.ax1.clear()
        self.ax2.clear()

        self.ax1.imshow(binary_image, cmap='gray')
        self.ax1.set_title('Binary Image', fontsize=12, fontweight='bold')
        self.ax1.axis('off')

        self.ax2.scatter(points[:, 0], points[:, 1], c='lightgray', s=2, alpha=0.4, label='Points')

        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
        for i, line in enumerate(lines):
            color = colors[i % len(colors)]
            # Draw line
            self.ax2.plot([line['start'][0], line['end'][0]],
                         [line['start'][1], line['end'][1]],
                         color=color, linewidth=4, alpha=0.9,
                         label=f'L{i+1}: {line["length"]:.2f}m')
            # Draw endpoints
            self.ax2.scatter([line['start'][0], line['end'][0]],
                           [line['start'][1], line['end'][1]],
                           c=color, s=80, marker='o', edgecolors='black', linewidths=1, zorder=5)
            # Label endpoints
            self.ax2.text(line['start'][0], line['start'][1], f'{i}s', fontsize=8, color='white',
                         ha='center', va='center', weight='bold', zorder=6)
            self.ax2.text(line['end'][0], line['end'][1], f'{i}e', fontsize=8, color='white',
                         ha='center', va='center', weight='bold', zorder=6)

        if shape_position is not None:
            self.ax2.scatter(shape_position[0], shape_position[1],
                           c='red', s=200, marker='X',
                           edgecolors='black', linewidths=2,
                           label=f'Position ({shape_position[0]:.2f}, {shape_position[1]:.2f})',
                           zorder=10)

        self.ax2.set_xlim(-0.5, 2.5)
        self.ax2.set_ylim(-1.5, 1.5)
        self.ax2.set_aspect('equal')
        self.ax2.set_xlabel('X (m)', fontsize=10)
        self.ax2.set_ylabel('Y (m)', fontsize=10)

        title_color = 'green' if shape_type in ['Square', 'Pentagon', 'Triangle'] else 'orange'
        self.ax2.set_title(f'Shape: {shape_type} | {len(lines)} lines',
                          fontsize=14, fontweight='bold', color=title_color)

        self.ax2.legend(fontsize=8, loc='upper right')
        self.ax2.grid(True, alpha=0.3)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main(args=None):
    rclpy.init(args=args)
    node = HoughLineDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
