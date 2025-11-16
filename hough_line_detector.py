#!/usr/bin/env python3
"""
Optimized Hough Transform Line Detector
Pure line detection - no classification (comes later)

Goal: Detect clean, accurate line segments from LiDAR scans
Strategy:
1. Isolate target cluster (nearest compact object)
2. Convert to high-quality binary image
3. Apply optimized Hough Transform
4. Aggressive merging to form long, clean lines
5. Remove overlapping/redundant segments

Output: List of distinct line segments with high accuracy
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import cv2
import matplotlib.pyplot as plt


class HoughLineDetector(Node):
    def __init__(self):
        super().__init__('hough_line_detector')

        # === SCAN FILTERING ===
        self.min_range = 0.1  # Minimum valid range (m)
        self.max_range = 3.0  # Maximum valid range (m)
        self.median_filter_size = 5  # Larger window for better noise reduction

        # === HOUGH TRANSFORM PARAMETERS ===
        # Tuned for optimal line detection
        self.rho_resolution = 0.005  # 5mm distance resolution (finer)
        self.theta_resolution = np.pi / 360  # 0.5° angular resolution (finer)
        self.hough_threshold = 12  # Minimum votes for line
        self.min_line_length = 0.08  # 8cm minimum
        self.max_line_gap = 0.03  # 3cm maximum gap

        # === IMAGE CONVERSION ===
        self.pixels_per_meter = 300  # Higher resolution for better accuracy
        self.image_size = 800  # Larger image

        # === LINE MERGING ===
        self.merge_angle_tolerance = 10  # degrees
        self.merge_distance_tolerance = 0.12  # meters
        self.overlap_threshold = 0.5

        # === DETECTION TRACKING ===
        # Track detected shapes to avoid duplicate detections
        self.detected_shapes = []  # List of (position, shape_type, distance_from_robot)
        self.detection_range = 0.25  # 30cm - shapes within this range are considered same detection

        self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)

        # Visualization
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(18, 9))

        self.get_logger().info('Hough Line Detector started - Optimized for accuracy')

    def scan_callback(self, msg):
        """Main processing pipeline"""
        # Step 1: Filter scan and convert to points
        all_points = self.scan_to_points(msg)
        if len(all_points) < 20:
            return

        # Step 2: Isolate target cluster
        cluster_points = self.isolate_target_cluster(all_points)
        if len(cluster_points) < 20:
            # self.get_logger().info('No valid cluster found')
            return

        # Step 3: Convert to binary image
        binary_image = self.points_to_binary_image(cluster_points)

        # Step 4: Detect lines using Hough Transform
        raw_lines = self.detect_lines_hough(binary_image)
        if len(raw_lines) == 0:
            # self.get_logger().info('No lines detected')
            return

        # self.get_logger().info(f'Hough detected {len(raw_lines)} raw segments')

        # Step 5: Filter lines - only keep those within cluster region
        cluster_lines = self.filter_lines_in_cluster(raw_lines, cluster_points)
        # self.get_logger().info(f'Lines in cluster: {len(cluster_lines)}')

        if len(cluster_lines) == 0:
            # self.get_logger().info('No lines within cluster region')
            return

        # Step 6: Merge collinear segments
        merged_lines = self.merge_collinear_segments(cluster_lines)
        # self.get_logger().info(f'After merging: {len(merged_lines)} lines')

        # Step 7: Remove overlaps
        clean_lines = self.remove_overlapping_lines(merged_lines)
        # self.get_logger().info(f'Final clean lines: {len(clean_lines)}')

        # Step 8: Order by connectivity
        ordered_lines = self.order_lines(clean_lines)

        # Calculate shape position (centroid of cluster)
        shape_position = np.mean(cluster_points, axis=0)
        shape_distance = np.linalg.norm(shape_position)

        # Step 9: Detect shape type
        shape_type = self.detect_shape(ordered_lines)

        # Step 10: Check if this is a new detection (not within 30cm of previous detections)
        if shape_type and shape_type in ['Pentagon', 'Square', 'Triangle']:
            is_new_detection = self.is_new_detection(shape_position, shape_type, shape_distance)

            if is_new_detection:
                # Mark as detected
                self.detected_shapes.append({
                    'position': shape_position,
                    'type': shape_type,
                    'distance': shape_distance
                })
                self.get_logger().info(f'✓ NEW DETECTION: {shape_type} at position ({shape_position[0]:.2f}, {shape_position[1]:.2f}), distance {shape_distance:.2f}m')
                self.get_logger().info(f'Total unique shapes detected: {len(self.detected_shapes)}')
            else:
                shape_type = f'{shape_type} (Already Detected)'

        # Visualize and log results
        self.visualize_results(cluster_points, binary_image, ordered_lines, shape_type)

    # ====================== FIX LINE ======================
    def detect_shape(self, lines):
        if len(lines) == 2:

            line1 = lines[0]
            line2 = lines[(1) % len(lines)]
            angle = self.calculate_angle_between_lines(line1, line2)
            angle = 180 - angle
            length1 = line1['length']
            length2 = line2['length']
            
            
            if(abs(angle-135)<5 and 3*length2<length1):
                self.get_logger().info('Fucked up Triangle ')
                print(angle)
                return 'Triangle'
        if len(lines) == 3:
            angles = []
            for i in range(len(lines)-1):
                line1 = lines[i]
                line2 = lines[(i + 1) % len(lines)]
                angle = self.calculate_angle_between_lines(line1, line2)
                angle = 180 - angle
                angles.append(angle)
            
            if abs(angles[0]-90)<5 and abs(angles[1]-140)<5 :
                self.get_logger().info('Detected Pentagon ')
                print(angles)
                return 'Pentagon'

            elif all(abs(angle - 90) < 5 for angle in angles): #ok
                self.get_logger().info('Detected Square ')
                print(angles)
                return 'Square'

        if len(lines) == 4:
            angles = []
            for i in range(len(lines)-1):
                line1 = lines[i]
                line2 = lines[(i + 1) % len(lines)]
                angle = self.calculate_angle_between_lines(line1, line2)
                angle = 180 - angle
                angles.append(angle)
            
            if abs(angles[0]-130)<7 and abs(angles[1]-90)<5 and abs(angles[2]-130)<7 :
                self.get_logger().info('Detected Triangle ')
                print(angles)
                return 'Triangle'

    def is_new_detection(self, current_position, shape_type, current_distance):
        """
        Check if this is a new shape detection or a duplicate.
        Returns True if the shape is more than 30cm away from all previously detected shapes.
        """
        for detected in self.detected_shapes:
            # Calculate distance between current position and previously detected position
            distance_between = np.linalg.norm(current_position - detected['position'])

            # If within detection range (30cm), this is the same shape
            if distance_between < self.detection_range:
                return False

        # This is a new shape (not within 30cm of any previous detection)
        return True

    # ==================== SCAN PROCESSING ====================

    def scan_to_points(self, msg):
        """Convert LaserScan to filtered point cloud"""
        # Apply median filter
        ranges = np.array(msg.ranges, dtype=float)
        ranges[np.isnan(ranges) | np.isinf(ranges)] = self.max_range + 1
        filtered_ranges = self.median_filter(ranges)

        # Convert to Cartesian points
        points = []
        for i, r in enumerate(filtered_ranges):
            if self.min_range <= r <= self.max_range:
                angle = msg.angle_min + i * msg.angle_increment
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                points.append([x, y])

        return np.array(points) if points else np.array([]).reshape(0, 2)

    def median_filter(self, data):
        """Optimized median filter"""
        filtered = np.copy(data)
        half_window = self.median_filter_size // 2

        for i in range(len(data)):
            start = max(0, i - half_window)
            end = min(len(data), i + half_window + 1)
            window = data[start:end]
            valid = window[window <= self.max_range]
            if len(valid) > 0:
                filtered[i] = np.median(valid)

        return filtered

    # ==================== CLUSTERING ====================

    def isolate_target_cluster(self, points):
        """
        Isolate the target object (shape on wall).
        Strategy: Find nearest compact cluster.
        """
        # Convert to polar coordinates
        polar_data = []
        for pt in points:
            angle = np.arctan2(pt[1], pt[0])
            distance = np.linalg.norm(pt)
            polar_data.append((pt, angle, distance))

        polar_data.sort(key=lambda x: x[1])  # Sort by angle

        # Cluster by angular and distance gaps
        clusters = self.cluster_points(polar_data)

        # Select best cluster (nearest valid shape)
        best_cluster = self.select_best_cluster(clusters)

        if best_cluster:
            return np.array([p[0] for p in best_cluster])
        return points

    def cluster_points(self, polar_data):
        """Split points into clusters based on gaps"""
        if len(polar_data) == 0:
            return []

        clusters = []
        current_cluster = [polar_data[0]]

        for i in range(1, len(polar_data)):
            prev_angle = polar_data[i-1][1]
            prev_dist = polar_data[i-1][2]
            curr_angle = polar_data[i][1]
            curr_dist = polar_data[i][2]

            # Gap detection
            angle_gap = abs(curr_angle - prev_angle)
            dist_gap = abs(curr_dist - prev_dist)

            if angle_gap > np.radians(10) or dist_gap > 0.25:
                clusters.append(current_cluster)
                current_cluster = [polar_data[i]]
            else:
                current_cluster.append(polar_data[i])

        clusters.append(current_cluster)
        return clusters

    def select_best_cluster(self, clusters):
        """Select cluster most likely to be target shape"""
        best = None
        min_distance = float('inf')

        for cluster in clusters:
            if len(cluster) < 25:  # Minimum points for valid shape
                continue

            points_array = np.array([p[0] for p in cluster])

            # Compute cluster metrics
            centroid = points_array.mean(axis=0)
            max_radius = np.max(np.linalg.norm(points_array - centroid, axis=1))
            avg_distance = np.mean([p[2] for p in cluster])
            angular_span = max(p[1] for p in cluster) - min(p[1] for p in cluster)

            # Valid shape criteria
            if (0.1 < max_radius < 0.8 and
                np.radians(20) < angular_span < np.radians(130) and
                0.3 < avg_distance < 2.5):

                # Choose nearest cluster
                if avg_distance < min_distance:
                    min_distance = avg_distance
                    best = cluster

        return best

    # ==================== IMAGE CONVERSION ====================

    def points_to_binary_image(self, points):
        """Convert point cloud to high-quality binary image"""
        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        center = self.image_size // 2

        for pt in points:
            px = int(pt[0] * self.pixels_per_meter + center)
            py = int(-pt[1] * self.pixels_per_meter + center)

            if 0 <= px < self.image_size and 0 <= py < self.image_size:
                # Draw slightly thicker points for better connectivity
                cv2.circle(image, (px, py), 2, 255, -1)

        return image

    # ==================== LINE FILTERING ====================

    def filter_lines_in_cluster(self, lines, cluster_points):
        """
        Filter lines to only keep those within the cluster region.
        Removes stray lines detected outside the target cluster.
        """
        if len(cluster_points) == 0:
            return lines

        # Calculate cluster bounding region with margin
        cluster_min = cluster_points.min(axis=0)
        cluster_max = cluster_points.max(axis=0)
        margin = 0.05  # 5cm margin

        cluster_min -= margin
        cluster_max += margin

        filtered_lines = []

        for line in lines:
            # Check if line endpoints are within cluster bounds
            start_in = self.point_in_bounds(line['start'], cluster_min, cluster_max)
            end_in = self.point_in_bounds(line['end'], cluster_min, cluster_max)

            # Also check if line midpoint is within cluster bounds
            midpoint = (line['start'] + line['end']) / 2
            mid_in = self.point_in_bounds(midpoint, cluster_min, cluster_max)

            # Keep line if at least 2 of 3 points (start, mid, end) are in cluster
            points_in_cluster = sum([start_in, end_in, mid_in])

            if points_in_cluster >= 2:
                filtered_lines.append(line)

        return filtered_lines

    def point_in_bounds(self, point, min_bounds, max_bounds):
        """Check if point is within bounding box"""
        return (min_bounds[0] <= point[0] <= max_bounds[0] and
                min_bounds[1] <= point[1] <= max_bounds[1])

    # ==================== HOUGH TRANSFORM ====================

    def detect_lines_hough(self, binary_image):
        """Apply Probabilistic Hough Transform for line detection"""
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

        # Convert to real-world coordinates
        lines = []
        center = self.image_size // 2
        ppm = self.pixels_per_meter

        for line in lines_cv:
            x1, y1, x2, y2 = line[0]

            # Convert to meters
            x1_m = (x1 - center) / ppm
            y1_m = -(y1 - center) / ppm
            x2_m = (x2 - center) / ppm
            y2_m = -(y2 - center) / ppm

            length = np.hypot(x2_m - x1_m, y2_m - y1_m)

            if length >= self.min_line_length:
                lines.append({
                    'start': np.array([x1_m, y1_m]),
                    'end': np.array([x2_m, y2_m]),
                    'length': length
                })

        return lines

    # ==================== LINE MERGING ====================

    def merge_collinear_segments(self, lines):
        """
        Aggressively merge collinear segments into single long lines.
        Uses transitive merging for maximum consolidation.
        """
        if len(lines) <= 1:
            return lines

        merged = []
        used = set()

        for i, line in enumerate(lines):
            if i in used:
                continue

            # Start a merge group
            group = [line]
            used.add(i)

            # Iteratively add similar lines (transitive closure)
            changed = True
            while changed:
                changed = False
                for j in range(len(lines)):
                    if j in used:
                        continue

                    # Check if j is similar to ANY line in group
                    for group_line in group:
                        if self.are_collinear(group_line, lines[j]):
                            group.append(lines[j])
                            used.add(j)
                            changed = True
                            break

            # Merge the entire group into one line
            merged.append(self.merge_line_group(group))

        return merged

    def are_collinear(self, line1, line2):
        """Check if two lines should be merged"""
        # Direction vectors
        dir1 = line1['end'] - line1['start']
        dir2 = line2['end'] - line2['start']

        len1 = np.linalg.norm(dir1)
        len2 = np.linalg.norm(dir2)

        if len1 < 1e-6 or len2 < 1e-6:
            return False

        dir1 /= len1
        dir2 /= len2

        # Check angular similarity
        dot = np.abs(np.dot(dir1, dir2))
        angle_diff = np.degrees(np.arccos(np.clip(dot, 0, 1)))

        if angle_diff > self.merge_angle_tolerance:
            return False

        # Check spatial proximity
        endpoints1 = [line1['start'], line1['end']]
        endpoints2 = [line2['start'], line2['end']]

        min_endpoint_dist = min(
            np.linalg.norm(p1 - p2)
            for p1 in endpoints1
            for p2 in endpoints2
        )

        # Also check perpendicular distance (for parallel lines)
        perp_dist = self.perpendicular_distance(line1, line2)

        # Merge if connected OR on same infinite line
        return (min_endpoint_dist < self.merge_distance_tolerance or
                (perp_dist < 0.04 and min_endpoint_dist < 0.20))

    def perpendicular_distance(self, line1, line2):
        """Calculate perpendicular distance between parallel lines"""
        mid2 = (line2['start'] + line2['end']) / 2
        dir1 = line1['end'] - line1['start']
        len1 = np.linalg.norm(dir1)

        if len1 < 1e-6:
            return float('inf')

        dir1 /= len1
        to_mid2 = mid2 - line1['start']
        parallel_comp = np.dot(to_mid2, dir1) * dir1
        perp_comp = to_mid2 - parallel_comp

        return np.linalg.norm(perp_comp)

    def merge_line_group(self, group):
        """Merge multiple lines into one spanning the full extent"""
        all_endpoints = []
        for line in group:
            all_endpoints.extend([line['start'], line['end']])

        all_endpoints = np.array(all_endpoints)

        # Find the two most distant points (true line endpoints)
        max_dist = 0
        best_pair = (all_endpoints[0], all_endpoints[1])

        for i in range(len(all_endpoints)):
            for j in range(i + 1, len(all_endpoints)):
                dist = np.linalg.norm(all_endpoints[i] - all_endpoints[j])
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (all_endpoints[i], all_endpoints[j])

        return {
            'start': best_pair[0],
            'end': best_pair[1],
            'length': max_dist
        }

    # ==================== OVERLAP REMOVAL ====================

    def remove_overlapping_lines(self, lines):
        """
        Remove overlapping lines - keep longer ones.
        Ensures clean, non-redundant line set.
        """
        if len(lines) <= 1:
            return lines

        # Sort by length (longest first)
        sorted_lines = sorted(lines, key=lambda l: l['length'], reverse=True)

        filtered = []
        for line in sorted_lines:
            is_overlapping = False

            for kept_line in filtered:
                if self.lines_overlap(line, kept_line):
                    is_overlapping = True
                    break

            if not is_overlapping:
                filtered.append(line)

        return filtered

    def lines_overlap(self, line1, line2):
        """Check if two lines overlap significantly"""
        # Must be parallel
        dir1 = line1['end'] - line1['start']
        dir2 = line2['end'] - line2['start']

        len1 = np.linalg.norm(dir1)
        len2 = np.linalg.norm(dir2)

        if len1 < 1e-6 or len2 < 1e-6:
            return False

        dir1 /= len1
        dir2 /= len2

        dot = np.abs(np.dot(dir1, dir2))
        if dot < 0.92:  # Not parallel enough
            return False

        # Check spatial overlap
        endpoints1 = [line1['start'], line1['end']]
        endpoints2 = [line2['start'], line2['end']]

        close_count = sum(
            1 for p1 in endpoints1 for p2 in endpoints2
            if np.linalg.norm(p1 - p2) < 0.08
        )

        # Check midpoint distance
        mid1 = (line1['start'] + line1['end']) / 2
        mid2 = (line2['start'] + line2['end']) / 2
        mid_dist = np.linalg.norm(mid1 - mid2)

        return close_count >= 2 or mid_dist < 0.06

    # ==================== ANGLE CALCULATION ====================

    def calculate_angle_between_lines(self, line1, line2):
        """
        Calculate the angle between two line segments.

        Args:
            line1: Dictionary with 'start' and 'end' numpy arrays
            line2: Dictionary with 'start' and 'end' numpy arrays

        Returns:
            float: Angle in degrees (0-180°)
        """
        # Get direction vectors
        dir1 = line1['end'] - line1['start']
        dir2 = line2['end'] - line2['start']

        # Normalize
        len1 = np.linalg.norm(dir1)
        len2 = np.linalg.norm(dir2)

        if len1 < 1e-6 or len2 < 1e-6:
            return 0.0  # Degenerate line

        dir1 = dir1 / len1
        dir2 = dir2 / len2

        # Calculate angle using dot product
        dot_product = np.dot(dir1, dir2)
        dot_product = np.clip(dot_product, -1.0, 1.0)

        # Get angle in radians, then convert to degrees
        angle_rad = np.arccos(np.abs(dot_product))
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    # ==================== LINE ORDERING ====================

    def order_lines(self, lines):
        """Order lines by connectivity to form coherent polygon"""
        if len(lines) <= 2:
            return lines

        ordered = [lines[0]]
        remaining = set(range(1, len(lines)))

        while remaining:
            last_line = ordered[-1]
            best_idx = None
            best_dist = float('inf')

            for idx in remaining:
                # Check all endpoint pairs
                distances = [
                    np.linalg.norm(last_line['end'] - lines[idx]['start']),
                    np.linalg.norm(last_line['end'] - lines[idx]['end']),
                    np.linalg.norm(last_line['start'] - lines[idx]['start']),
                    np.linalg.norm(last_line['start'] - lines[idx]['end'])
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

    # ==================== VISUALIZATION ====================

    def visualize_results(self, points, binary_image, lines, shape_type='Unknown'):
        """Visualize detection results with shape type"""
        self.ax1.clear()
        self.ax2.clear()

        # Left: Binary image
        self.ax1.imshow(binary_image, cmap='gray')
        self.ax1.set_title('Binary Image (Hough Input)', fontsize=12, fontweight='bold')
        self.ax1.axis('off')

        # Right: Detected lines
        self.ax2.scatter(points[:, 0], points[:, 1], c='lightgray', s=2, alpha=0.4, label='Points')

        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
        for i, line in enumerate(lines):
            color = colors[i % len(colors)]
            x = [line['start'][0], line['end'][0]]
            y = [line['start'][1], line['end'][1]]
            self.ax2.plot(x, y, color=color, linewidth=4, alpha=0.9,
                         label=f'L{i+1}: {line["length"]:.2f}m')

        self.ax2.set_xlim(-0.5, 2.5)
        self.ax2.set_ylim(-1.5, 1.5)
        self.ax2.set_aspect('equal')
        self.ax2.set_xlabel('X (m)', fontsize=10)
        self.ax2.set_ylabel('Y (m)', fontsize=10)

        # Show shape type in title
        title_color = 'green' if shape_type in ['Square', 'Pentagon', 'Triangle'] else 'orange'
        self.ax2.set_title(
            f'Shape: {shape_type} | {len(lines)} lines',
            fontsize=14, fontweight='bold', color=title_color
        )

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
