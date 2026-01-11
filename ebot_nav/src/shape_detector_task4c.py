#!/usr/bin/env python3
"""Optimized line detector for laser scan data with shape detection"""

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
from typing import List, Dict, Tuple, Optional
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
    TRIANGLE_ANGLE_45 = (45, 10)  # (target, tolerance) for 2-line triangle
    TRIANGLE_INTERIOR_ANGLE = 130.0  # Interior angle for triangle detection
    TRIANGLE_ANGLE_TOLERANCE = 10.0  # Tolerance for interior angle (degrees)
    SQUARE_ANGLE = (90, 8)  # Square angle tolerance
    PENTAGON_ANGLES = [(90, 6), (140, 6)]  # Pentagon specific angles

    # Distance thresholds
    CORNER_GAP_MAX = 0.06  # Max gap between connected lines
    MAX_TRIANGLE_LENGTH = 0.25
    MAX_PENTAGON_SQUARE_LENGTH = 0.35

    # Triangle edge length constraints
    TRIANGLE_L2_TARGET = 0.20  # Expected L2 length for triangle (meters)
    TRIANGLE_L2_TOLERANCE = 0.05  # L2 length tolerance (meters)

    # Safety offset to prevent robot collision with shape
    SAFETY_OFFSET_Y_SQUARE = 0.28  # 25cm offset in y-direction
    SAFETY_OFFSET_X_SQUARE = 0.10  # 10cm offset in x-direction for Square

    SAFETY_OFFSET_Y_TRIANGLE = 0.60  # 25cm offset in y-direction  
    SAFETY_OFFSET_X_TRIANGLE = 0.25  # 15cm offset in x-direction for Triangle

    def __init__(self):
        super().__init__('hough_line_detector')

        # Scan filtering parameters
        self.min_range = 0.1
        self.max_range = 5.0
        self.median_filter_size = 10  # Reduced from 25 to 10 to preserve more points and reduce gaps

        # Hough parameters
        self.rho_resolution = 0.005
        self.theta_resolution = np.pi / 360
        self.hough_threshold = 11      # Lowered from 12 to 8 to detect weaker/shorter lines
        self.min_line_length = 0.12  # Lowered from 0.15 to 0.10 (10cm) to catch shorter segments
        self.max_line_gap = 0.08     # Increased from 0.05 to 0.10 (10cm) to bridge larger gaps

        # Minimum valid line length to filter noise (2cm)
        self.min_valid_line_length = 0.02

        # Image conversion
        self.pixels_per_meter = 300
        self.image_size = 800

        # Detection tracking
        self.detected_shapes: List[ShapeDetection] = []
        self.detection_range = 1.0  # Minimum 1m separation between shapes
        self.pending_shape: Optional[Tuple[str, int]] = None  # (shape_status, plant_id)

        # Pentagon gating: Only detect Square/Triangle AFTER Pentagon is found
        self.pentagon_detected = False
        self.pentagon_position: Optional[np.ndarray] = None  # Store pentagon location
        self.pentagon_exclusion_radius = 1.5  # Don't detect triangles within 1.5m of pentagon

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

        # Gating notice
        self.get_logger().info('WAITING FOR PENTAGON - Square/Triangle detection DISABLED until Pentagon is found')

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
        """Process each laser scan"""
        try:
            if self.detection_paused:
                return

            # Cache laser offset from TF (only done once)
            self._cache_laser_offset(msg.header.frame_id)

            # Get robot pose at scan timestamp using odometry
            scan_position_x, scan_position_y, scan_yaw = self._get_scan_pose(msg)

            # Process scan data (angle-filtered in scan_to_points)
            all_points = self.scan_to_points(msg)
            if len(all_points) < 20:
                return

            # Note: all_points already filtered by angle in scan_to_points()
            filtered_points = self.filter_by_distance(all_points, self.max_range)
            if len(filtered_points) < 10:
                return

            # Detect lines using Hough transform
            binary_image = self.points_to_image(filtered_points)
            raw_lines = self.detect_lines(binary_image)
            self.get_logger().debug(f'[DETECTION] Raw lines detected: {len(raw_lines)}')
            if len(raw_lines) == 0:
                return

            # Separate lines by valid angles
            valid_lines, invalid_lines = self.separate_lines_by_angle(raw_lines)
            self.get_logger().debug(
                f'[DETECTION] Valid lines: {len(valid_lines)}, Invalid lines: {len(invalid_lines)}'
            )
            if len(valid_lines) == 0:
                return

            # Merge invalid-angle lines to nearby valid-angle lines if collinear
            recovered_lines = self.merge_invalid_to_valid_lines(valid_lines, invalid_lines)
            self.get_logger().debug(f'[DETECTION] After recovery: {len(recovered_lines)} lines')

            # Merge collinear/similar lines
            merged_lines = self.merge_collinear_lines(recovered_lines)
            self.get_logger().debug(f'[DETECTION] After merging: {len(merged_lines)} lines')

            # Group nearby lines and find best shape candidate
            line_groups = self.group_by_proximity(merged_lines, 0.15)
            self.get_logger().debug(f'[DETECTION] Line groups: {len(line_groups)}, sizes: {[len(g) for g in line_groups]}')
            best_group = self._find_best_group(line_groups)

            if not best_group:
                self.get_logger().debug('[DETECTION] No valid line group found')
                self.visualize(filtered_points, binary_image, merged_lines,
                             shape_type='No Shape', shape_position=None)
                return

            self.get_logger().debug(f'[DETECTION] Best group has {len(best_group)} lines')

            # Determine shape type and position
            shape_type = self.detect_shape(best_group)
            self.get_logger().debug(f'[DETECTION] Shape detected: {shape_type}')

            shape_position = self._calculate_shape_position(best_group, shape_type)
            shape_distance = np.linalg.norm(shape_position)

            # Process valid detections
            if shape_type in ['Pentagon', 'Square', 'Triangle']:

                is_new, world_pos = self.is_new_detection(
                    shape_position, scan_position_x, scan_position_y, scan_yaw
                )

                # GATING LOGIC: Pentagon must be detected first before Square/Triangle
                if shape_type == 'Pentagon':
                    # Always process Pentagon - it's the trigger for other detections
                    if is_new:
                        if not self.pentagon_detected:
                            self.pentagon_detected = True
                            self.pentagon_position = world_pos
                            self.get_logger().info(
                                f'🔓 PENTAGON DETECTED - Enabling Square/Triangle detection! '
                                f'Position: ({world_pos[0]:.2f}, {world_pos[1]:.2f})'
                            )

                        plant_id = self.assign_plant_id(world_pos, scan_position_x, scan_position_y)
                        shape_status = self.SHAPE_STATUS_MAP.get(shape_type)
                        goal_world_pos = world_pos  # Pentagon (dock) - no offset needed

                        detection = ShapeDetection(
                            local_position=shape_position,
                            world_position=world_pos,
                            goal_position=goal_world_pos,
                            shape_type=shape_type,
                            distance=shape_distance,
                            plant_id=plant_id
                        )
                        self.detected_shapes.append(detection)

                        self.get_logger().info(
                            f'✓ {shape_type} detected | Plant ID: {plant_id} | '
                            f'World: ({world_pos[0]:.2f}, {world_pos[1]:.2f}) | '
                            f'Total: {len(self.detected_shapes)}'
                        )

                        self.publish_shape_markers()
                    else:
                        self.get_logger().debug(
                            f'Duplicate {shape_type} ignored at ({world_pos[0]:.2f}, {world_pos[1]:.2f})'
                        )
                        shape_type = f'{shape_type} (Duplicate)'

                elif shape_type in ['Square', 'Triangle']:
                    # Gate: Only process Square/Triangle AFTER Pentagon is detected
                    if not self.pentagon_detected:
                        self.get_logger().debug(
                            f'[GATE] {shape_type} detected but Pentagon not found yet - IGNORING'
                        )
                        return

                    # Additional filter: Don't detect Triangles near Pentagon (false positives)
                    if shape_type == 'Triangle' and self.pentagon_position is not None:
                        distance_to_pentagon = np.linalg.norm(world_pos - self.pentagon_position)
                        if distance_to_pentagon < self.pentagon_exclusion_radius:
                            self.get_logger().debug(
                                f'[GATE] Triangle too close to Pentagon ({distance_to_pentagon:.2f}m < '
                                f'{self.pentagon_exclusion_radius}m) - IGNORING (likely false positive)'
                            )
                            return

                    # Proceed with detection
                    if is_new:
                        plant_id = self.assign_plant_id(world_pos, scan_position_x, scan_position_y)
                        shape_status = self.SHAPE_STATUS_MAP.get(shape_type)

                        # Calculate goal position with safety offset
                        goal_world_pos = self.local_to_world(
                            shape_position, scan_position_x, scan_position_y, scan_yaw,
                            apply_safety=True, shape_type=shape_type
                        )

                        detection = ShapeDetection(
                            local_position=shape_position,
                            world_position=world_pos,
                            goal_position=goal_world_pos,
                            shape_type=shape_type,
                            distance=shape_distance,
                            plant_id=plant_id
                        )
                        self.detected_shapes.append(detection)

                        self.get_logger().info(
                            f'✓ {shape_type} detected | Plant ID: {plant_id} | '
                            f'World: ({world_pos[0]:.2f}, {world_pos[1]:.2f}) | '
                            f'Total: {len(self.detected_shapes)}'
                        )

                        # Publish shape pose for priority navigation
                        pose_msg = String()
                        pose_msg.data = f"{shape_type},{goal_world_pos[0]:.3f},{goal_world_pos[1]:.3f}"
                        self.shape_pose_pub.publish(pose_msg)
                        self.pending_shape = (shape_status, plant_id)

                        self.publish_shape_markers()
                    else:
                        self.get_logger().debug(
                            f'Duplicate {shape_type} ignored at ({world_pos[0]:.2f}, {world_pos[1]:.2f})'
                        )
                        shape_type = f'{shape_type} (Duplicate)'

            self.visualize(filtered_points, binary_image, best_group, shape_type, shape_position)

        except Exception as e:
            self.get_logger().error(f"Error in scan_callback: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())


    # ========================================================================
    # SCAN PROCESSING METHODS
    # ========================================================================

    def filter_by_distance(self, points: np.ndarray, max_dist: float) -> np.ndarray:
        """Filter points by maximum distance"""
        return points[np.linalg.norm(points, axis=1) <= max_dist]

    def scan_to_points(self, msg: LaserScan) -> np.ndarray:
        """Convert laser scan to 2D points, filtering by angle and position"""
        ranges = np.array(msg.ranges, dtype=float)
        ranges[np.isnan(ranges) | np.isinf(ranges)] = self.max_range + 1
        filtered = self.median_filter(ranges)

        # Angle range for valid scan rays (in radians)
        # Keep rays from -90° to +90° (left and right sides)
        MIN_SCAN_ANGLE = -np.pi / 2  # -90 degrees
        MAX_SCAN_ANGLE = np.pi / 2   # +90 degrees

        # Position limits to stay within lane (reject points outside lane boundaries)
        MAX_FORWARD_DISTANCE = 2.0  # Max X distance (2m forward)
        MIN_FORWARD_DISTANCE = 0.1  # Min X distance (10cm, ignore very close)
        MAX_LATERAL_DISTANCE = 1.8  # Max |Y| distance (1.8m left/right)
        MIN_LATERAL_DISTANCE = 0.1  # Min |Y| distance (10cm, ignore center)

        points = []
        for i, r in enumerate(filtered):
            if self.min_range <= r <= self.max_range:
                angle = msg.angle_min + i * msg.angle_increment

                # Filter by scan angle - keep only forward hemisphere
                if MIN_SCAN_ANGLE <= angle <= MAX_SCAN_ANGLE:
                    x = r * np.cos(angle)
                    y = r * np.sin(angle)

                    # Additional filtering by position (X and Y coordinates)
                    # Reject points outside lane boundaries
                    if (MIN_FORWARD_DISTANCE <= x <= MAX_FORWARD_DISTANCE and
                        MIN_LATERAL_DISTANCE <= abs(y) <= MAX_LATERAL_DISTANCE):
                        points.append([x, y])

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

    def separate_lines_by_angle(self, lines: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Separate lines into valid-angle and invalid-angle groups"""
        valid_angles = [0, 30, 45, 60, 90, 120, 130, 135, 150, 180]
        angle_tolerance = 5

        valid_lines = []
        invalid_lines = []
        angle_log = []

        for line in lines:
            angle_deg = np.degrees(np.arctan2(line['end'][1] - line['start'][1],
                                               line['end'][0] - line['start'][0]))
            if angle_deg < 0:
                angle_deg += 180

            # Check if angle matches any valid angle
            is_valid = any(abs(angle_deg - valid_angle) <= angle_tolerance
                          for valid_angle in valid_angles)

            (valid_lines if is_valid else invalid_lines).append(line)
            angle_log.append(f"{angle_deg:.1f}°{'✓' if is_valid else '✗'}")

        self.get_logger().debug(f'[ANGLES] Detected angles: {", ".join(angle_log)}')
        return valid_lines, invalid_lines

    def merge_invalid_to_valid_lines(self, valid_lines: List[Dict], invalid_lines: List[Dict]) -> List[Dict]:
        """Merge invalid-angle lines to nearby valid-angle lines if collinear"""
        if not invalid_lines:
            return valid_lines

        for invalid_line in invalid_lines:
            best_match = None
            best_score = float('inf')

            dir_invalid, _ = self.normalize_dir(invalid_line)
            if dir_invalid is None:
                continue

            mid_invalid = (invalid_line['start'] + invalid_line['end']) * 0.5

            for valid_line in valid_lines:
                dir_valid, _ = self.normalize_dir(valid_line)
                if dir_valid is None:
                    continue

                # Quick angle check
                angle = np.degrees(np.arccos(np.clip(np.abs(np.dot(dir_valid, dir_invalid)), 0, 1)))
                if angle > 12:  # Increased from 10 to 12 degrees
                    continue

                # Quick distance checks
                mid_valid = (valid_line['start'] + valid_line['end']) * 0.5
                if np.linalg.norm(mid_valid - mid_invalid) > 0.40:  # Increased from 0.30 to 0.40 (40cm)
                    continue

                # Check perpendicular distance
                to_mid = mid_invalid - valid_line['start']
                perp_dist = np.linalg.norm(to_mid - np.dot(to_mid, dir_valid) * dir_valid)
                if perp_dist > 0.10:  # Increased from 0.08 to 0.10 (10cm)
                    continue

                # Score based on proximity
                score = perp_dist
                if score < best_score:
                    best_score = score
                    best_match = valid_line

            # Merge if match found
            if best_match is not None:
                all_points = np.vstack([best_match['start'], best_match['end'],
                                       invalid_line['start'], invalid_line['end']])
                dists = np.linalg.norm(all_points[:, None] - all_points[None, :], axis=2)
                i, j = np.unravel_index(dists.argmax(), dists.shape)

                best_match['start'] = all_points[i]
                best_match['end'] = all_points[j]
                best_match['length'] = dists[i, j]

        return valid_lines

    def merge_collinear_lines(self, lines: List[Dict]) -> List[Dict]:
        """Merge collinear and nearby lines - improved with noise filtering and overlap removal"""
        initial_count = len(lines)
        if len(lines) <= 1:
            return lines

        # Step 1: Filter out noise - lines shorter than min_valid_line_length
        lines = [l for l in lines if l['length'] >= self.min_valid_line_length]
        after_filter = len(lines)
        if len(lines) <= 1:
            return lines

        # Step 2: Merge collinear lines
        angle_tolerance = 5  # degrees - relaxed from 3 to 5 for better merging
        endpoint_distance = 0.20  # meters - increased from 0.12 to 0.20 (20cm) to bridge larger gaps
        perpendicular_tolerance = 0.08  # meters - increased from 0.05 to 0.08 (8cm) for more tolerance

        merged = []
        used = set()

        for i, line1 in enumerate(lines):
            if i in used:
                continue

            # Start a group with this line
            group = [line1]
            used.add(i)

            # Find all similar lines that are truly connected
            changed = True
            while changed:
                changed = False
                for j, line2 in enumerate(lines):
                    if j in used:
                        continue

                    # Check if line2 should be merged with any line in the group
                    for group_line in group:
                        if self.should_merge_conservative(group_line, line2, angle_tolerance,
                                                          endpoint_distance, perpendicular_tolerance):
                            group.append(line2)
                            used.add(j)
                            changed = True
                            break

            # Merge the group into a single line
            merged_line = self.merge_line_group(group)
            if len(group) > 1:
                self.get_logger().debug(f'[MERGE] Merged {len(group)} fragments into one line')
            merged.append(merged_line)

        # Step 3: Remove overlapping lines (keep longer ones)
        sorted_lines = sorted(merged, key=lambda l: l['length'], reverse=True)
        filtered = []

        for line in sorted_lines:
            if not any(self.check_overlap(line, kept) for kept in filtered):
                filtered.append(line)

        # Log merging statistics
        self.get_logger().debug(
            f'[MERGE] Line merging: {initial_count} → {after_filter} (after noise filter) → '
            f'{len(merged)} (after merge) → {len(filtered)} (after overlap removal)'
        )

        return filtered

    def should_merge_conservative(self, line1: Dict, line2: Dict, angle_tol: float,
                                  endpoint_dist_tol: float, perp_tol: float) -> bool:
        """Check if two lines should be merged"""
        dir1, _ = self.normalize_dir(line1)
        dir2, _ = self.normalize_dir(line2)
        if dir1 is None or dir2 is None:
            return False

        # Quick angle check
        if np.degrees(np.arccos(np.clip(np.abs(np.dot(dir1, dir2)), 0, 1))) > angle_tol:
            return False

        # Check center distance
        mid1 = (line1['start'] + line1['end']) * 0.5
        mid2 = (line2['start'] + line2['end']) * 0.5
        if np.linalg.norm(mid1 - mid2) > 0.50:  # Increased from 0.35 to 0.50 (50cm)
            return False

        # Check endpoint proximity
        eps1 = np.array([line1['start'], line1['end']])
        eps2 = np.array([line2['start'], line2['end']])
        dists = np.linalg.norm(eps1[:, None] - eps2[None, :], axis=2)
        if dists.min() > endpoint_dist_tol:
            return False

        # Check collinearity
        to_mid = mid2 - line1['start']
        return np.linalg.norm(to_mid - np.dot(to_mid, dir1) * dir1) < perp_tol

    def merge_line_group(self, group: List[Dict]) -> Dict:
        """Merge a group of lines by finding furthest endpoints"""
        all_points = np.array([[l['start'], l['end']] for l in group]).reshape(-1, 2)
        dists = np.linalg.norm(all_points[:, None] - all_points[None, :], axis=2)
        i, j = np.unravel_index(dists.argmax(), dists.shape)

        return {
            'start': all_points[i],
            'end': all_points[j],
            'length': dists[i, j]
        }

    def normalize_dir(self, line: Dict) -> Tuple[Optional[np.ndarray], float]:
        """Get normalized direction vector"""
        d = line['end'] - line['start']
        length = np.linalg.norm(d)
        return (d/length, length) if length > 1e-6 else (None, length)

    def check_overlap(self, line1: Dict, line2: Dict) -> bool:
        """Check if two lines overlap (for duplicate removal)"""
        dir1, len1 = self.normalize_dir(line1)
        dir2, len2 = self.normalize_dir(line2)

        if dir1 is None or dir2 is None:
            return False

        # Check if lines are parallel
        if np.abs(np.dot(dir1, dir2)) < 0.95:
            return False

        # Check perpendicular distance (collinearity)
        mid1 = (line1['start'] + line1['end']) / 2
        mid2 = (line2['start'] + line2['end']) / 2
        mid_vec = mid2 - mid1
        perp_dist = np.linalg.norm(mid_vec - np.dot(mid_vec, dir1) * dir1)

        if perp_dist < 0.03:  # Within 3cm perpendicular distance
            # Check projection overlap on the line
            ref = line1['start']
            projs = [
                np.dot(line1['start'] - ref, dir1), np.dot(line1['end'] - ref, dir1),
                np.dot(line2['start'] - ref, dir1), np.dot(line2['end'] - ref, dir1)
            ]
            min1, max1 = min(projs[0], projs[1]), max(projs[0], projs[1])
            min2, max2 = min(projs[2], projs[3]), max(projs[2], projs[3])

            # Check if projections overlap
            if min(max1, max2) - max(min1, min2) > 0.02:  # 2cm overlap
                return True

        # Check if many endpoints are close (alternative overlap detection)
        eps1 = [line1['start'], line1['end']]
        eps2 = [line2['start'], line2['end']]
        close_count = sum(1 for p1 in eps1 for p2 in eps2 if np.linalg.norm(p1 - p2) < 0.05)

        return close_count >= 3

    def calc_angle(self, line1: Dict, line2: Dict) -> float:
        """Calculate angle between two lines in degrees"""
        dir1, _ = self.normalize_dir(line1)
        dir2, _ = self.normalize_dir(line2)
        if dir1 is None or dir2 is None:
            return 0.0
        return np.degrees(np.arccos(np.clip(np.abs(np.dot(dir1, dir2)), -1.0, 1.0)))

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
    # SHAPE DETECTION METHODS
    # ========================================================================

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
        return self.position_x, self.position_y, self.yaw

    def _find_best_group(self, line_groups: List[List[Dict]]) -> Optional[List[Dict]]:
        """Find the best line group based on scoring"""
        best_group = None
        best_score = -1

        for group in line_groups:
            score = self.score_group(group)
            if score > best_score:
                best_score = score
                best_group = group

        return best_group

    def _calculate_shape_position(self, lines: List[Dict], shape_type: str) -> np.ndarray:
        """Calculate shape position based on type"""
        if shape_type == 'Triangle':
            if len(lines) == 2:
                return self.compute_corner(lines[0], lines[1])
            elif len(lines) >= 4:
                return self.compute_corner(lines[1], lines[2])
        elif shape_type == 'Square' and len(lines) >= 3:
            return self.compute_corner(lines[1], lines[2])

        # Default: centroid
        all_endpoints = np.array([[l['start'], l['end']] for l in lines]).reshape(-1, 2)
        return np.mean(all_endpoints, axis=0)

    def detect_shape(self, lines: List[Dict]) -> Optional[str]:
        """Detect shape type from line segments"""
        lengths = [l['length'] for l in lines]
        self.get_logger().debug(
            f'[SHAPE] Analyzing {len(lines)} lines, lengths: {[f"{l:.3f}m" for l in lengths]}'
        )

        # Check 3-line shapes first (Pentagon/Square)
        if len(lines) == 3:
            angles = [180 - self.calc_angle(lines[i], lines[i+1]) for i in range(2)]
            gaps = [np.linalg.norm(lines[i]['end'] - lines[i+1]['start']) for i in range(2)]

            self.get_logger().debug(
                f'[SHAPE] 3-line group: angles={[f"{a:.1f}°" for a in angles]}, '
                f'gaps={[f"{g:.3f}m" for g in gaps]}'
            )

            # Pentagon: 90° and 140° angles
            if (abs(angles[0] - 90) < 7 and abs(angles[1] - 140) < 7 and
                all(g < self.CORNER_GAP_MAX for g in gaps)):
                self.get_logger().info('✓ Pentagon detected')
                return 'Pentagon'

            # Square: all 90° angles
            if (all(abs(a - 90) < 10 for a in angles) and
                all(g < self.CORNER_GAP_MAX for g in gaps) and
                lines[1]['length'] < self.MAX_PENTAGON_SQUARE_LENGTH and
                lines[2]['length'] < self.MAX_PENTAGON_SQUARE_LENGTH):
                self.get_logger().info('✓ Square detected')
                return 'Square'

        # Triangle: Check consecutive line pairs
        if len(lines) >= 2:
            if self.detect_triangle_pattern(lines):
                return 'Triangle'

        self.get_logger().debug('[SHAPE] No shape detected')
        return None

    def detect_triangle_pattern(self, lines: List[Dict]) -> bool:
        """Detect triangle from consecutive line pairs: ~45° angle, L2 ~20cm"""
        self.get_logger().debug(f'[TRIANGLE] Checking {len(lines)} lines for triangle pattern')

        for i in range(len(lines) - 1):
            angle_inv = 180 - self.calc_angle(lines[i], lines[i + 1])
            corner_gap = np.linalg.norm(lines[i]['end'] - lines[i + 1]['start'])
            L1 = lines[i]['length']
            L2 = lines[i + 1]['length']

            # Calculate deviations from expected values
            angle_diff = abs(angle_inv - self.TRIANGLE_INTERIOR_ANGLE)
            length_diff = abs(L2 - self.TRIANGLE_L2_TARGET)

            # Check each condition
            angle_ok = angle_diff < self.TRIANGLE_ANGLE_TOLERANCE
            length_ok = length_diff < self.TRIANGLE_L2_TOLERANCE
            gap_ok = corner_gap < self.CORNER_GAP_MAX

            self.get_logger().debug(
                f'[TRIANGLE] Pair {i}-{i+1}: '
                f'angle={angle_inv:.1f}° (target={self.TRIANGLE_INTERIOR_ANGLE}°, diff={angle_diff:.1f}°, {"✓" if angle_ok else "✗"}), '
                f'L1={L1:.3f}m, L2={L2:.3f}m (target={self.TRIANGLE_L2_TARGET}m, diff={length_diff:.3f}m, {"✓" if length_ok else "✗"}), '
                f'gap={corner_gap:.3f}m (max={self.CORNER_GAP_MAX}m, {"✓" if gap_ok else "✗"})'
            )

            if angle_ok and length_ok and gap_ok:
                self.get_logger().info(
                    f'✓ Triangle detected: angle={angle_inv:.1f}°, L1={L1:.3f}m, L2={L2:.3f}m'
                )
                return True

        self.get_logger().debug('[TRIANGLE] No valid triangle pattern found')
        return False

    def assign_plant_id(self, shape_world_pos: np.ndarray, robot_x: float, robot_y: float) -> Optional[int]:
        """
        Assign plant_id (0-8) based on lane and segment
        """
        # Define lane boundaries (expanded to include robot waypoint positions)
        lanes = {
            1: {'x_range': (0.20, 0.60), 'right_id': 0, 'left_segments': [((-4.67, -3.374), 1), ((-3.374, -2.020), 2), ((-2.020, -0.720), 3), ((-0.720, 0.552), 4)]},
            2: {'x_range': (-1.70, -1.20), 'right_segments': [((-4.757, -3.381), 1), ((-3.381, -2.057), 2), ((-2.057, -0.711), 3), ((-0.711, 0.553), 4)], 'left_segments': [((-4.757, -3.381), 5), ((-3.381, -2.057), 6), ((-2.057, -0.711), 7), ((-0.711, 0.553), 8)]},
            3: {'x_range': (-3.70, -3.20), 'right_segments': [((-4.716, -3.456), 5), ((-3.456, -2.079), 6), ((-2.079, -0.723), 7), ((-0.723, 0.527), 8)]}
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
                return 0
            else:
                for (y_min, y_max), plant_id in lanes[1]['left_segments']:
                    if y_min <= robot_y <= y_max:
                        return plant_id

        # Lane 2: Both sides
        elif current_lane == 2:
            segments = lanes[2]['right_segments'] if not is_left_side else lanes[2]['left_segments']
            for (y_min, y_max), plant_id in segments:
                if y_min <= robot_y <= y_max:
                    return plant_id

        # Lane 3: Right side only
        elif current_lane == 3 and not is_left_side:
            for (y_min, y_max), plant_id in lanes[3]['right_segments']:
                if y_min <= robot_y <= y_max:
                    return plant_id

        self.get_logger().warn(f"Could not assign plant_id")
        return None

    def apply_safety_offset(self, local_pos: np.ndarray, shape_type: str = None) -> np.ndarray:
        """Apply safety offset to shape position to prevent collision

        Y-offset: Shifts goal toward robot's centerline by SAFETY_OFFSET_Y
        X-offset: For Square only, push forward by SAFETY_OFFSET_X_SQUARE
        """
        if shape_type == 'Square':
            offset_x = self.SAFETY_OFFSET_X_SQUARE
            offset_y = -np.sign(local_pos[1]) * self.SAFETY_OFFSET_Y_SQUARE if local_pos[1] != 0 else 0.0

        elif shape_type == 'Triangle':
            offset_x = self.SAFETY_OFFSET_X_TRIANGLE
            offset_y = -np.sign(local_pos[1]) * self.SAFETY_OFFSET_Y_TRIANGLE if local_pos[1] != 0 else 0.0       

        adjusted_pos = np.array([local_pos[0] + offset_x, local_pos[1] + offset_y])
        return adjusted_pos

    def local_to_world(self, local_pos: np.ndarray, robot_x: float = None,
                      robot_y: float = None, robot_yaw: float = None,
                      apply_safety: bool = False, shape_type: str = None) -> np.ndarray:
        """Convert local laser frame coordinates to world coordinates"""
        robot_x = robot_x if robot_x is not None else self.position_x
        robot_y = robot_y if robot_y is not None else self.position_y
        robot_yaw = robot_yaw if robot_yaw is not None else self.yaw

        # Make a copy to avoid mutating the original
        local_pos = np.copy(local_pos)

        # Apply safety offset if requested (for navigation goals)
        if apply_safety:
            local_pos = self.apply_safety_offset(local_pos, shape_type=shape_type)

        # Step 1: Transform from laser frame to base_link frame
        base_link_x = local_pos[0] + self.laser_offset_x
        base_link_y = local_pos[1] + self.laser_offset_y

        # Step 2: Transform from base_link frame to world frame
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)

        world_x = robot_x + (base_link_x * cos_yaw - base_link_y * sin_yaw)
        world_y = robot_y + (base_link_x * sin_yaw + base_link_y * cos_yaw)

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

        n = len(lines)
        score = 10.0 if 2 <= n <= 4 else (2.0 if n == 1 else 1.0)

        all_pts = np.array([[l['start'], l['end']] for l in lines]).reshape(-1, 2)
        bbox_size = np.linalg.norm(all_pts.ptp(axis=0))

        if 0.15 < bbox_size < 0.8:
            score += 5.0 / (bbox_size + 0.1)

        if np.mean([l['length'] for l in lines]) < 0.3:
            score += 5.0

        score += max(0, 5.0 - np.linalg.norm(all_pts.mean(axis=0)))

        return score

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
        """Visualize detection results - just the lines"""
        if not self.enable_visualization or self.fig is None:
            return

        self.ax1.clear()
        self.ax2.clear()

        # Left panel: Binary image
        self.ax1.imshow(binary_image, cmap='gray')
        self.ax1.set_title('Binary Image', fontsize=12, fontweight='bold')
        self.ax1.axis('off')

        # Right panel: Point cloud and lines
        self.ax2.scatter(points[:, 0], points[:, 1], c='lightgray', s=1, alpha=0.3)

        # Draw detected lines only
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
        for i, line in enumerate(lines):
            color = colors[i % len(colors)]
            self.ax2.plot([line['start'][0], line['end'][0]],
                         [line['start'][1], line['end'][1]],
                         color=color, linewidth=3, alpha=0.9)

        # Set plot limits and appearance
        self.ax2.set_xlim(-0.5, 2.5)
        self.ax2.set_ylim(-1.5, 1.5)
        self.ax2.set_aspect('equal')
        self.ax2.set_xlabel('X (m)', fontsize=10)
        self.ax2.set_ylabel('Y (m)', fontsize=10)
        self.ax2.set_title(f'Detected Lines: {len(lines)}', fontsize=12, fontweight='bold')
        self.ax2.grid(True, alpha=0.3)

        # Update display
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
