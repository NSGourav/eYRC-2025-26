#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

class HoughLineDetector(Node):
    def __init__(self):
        super().__init__('hough_line_detector')

        # Scan filtering
        self.min_range = 0.1
        self.max_range = 1.0
        self.median_filter_size = 5

        # Hough parameters
        self.rho_resolution = 0.005
        self.theta_resolution = np.pi / 360
        self.hough_threshold = 12
        self.min_line_length = 0.08
        self.max_line_gap = 0.03

        # Image conversion
        self.pixels_per_meter = 300
        self.image_size = 800

        # Line processing
        self.merge_angle_tolerance = 10
        self.merge_distance_tolerance = 0.12

        # Detection tracking
        self.detected_shapes = []
        self.detection_range = 0.3

        self.position_x=0.0
        self.position_y=0.0
        self.yaw=0.0

        self.world_pos=np.array([0.0,0.0])

        self.flag_print=False
        self.shape_print=None

        self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.shape_pub=self.create_publisher(String, '/detection_status', 10)

        self.shape_status_map = {
            "Triangle": "FERTILIZER_REQUIRED",
            "Square": "BAD_HEALTH",
            "Pentagon": "DOCK_STATION"
        }

        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(18, 9))
        self.get_logger().info('Hough Line Detector started')
    
    def odom_callback(self, msg: Odometry):
        self.position_x=msg.pose.pose.position.x
        self.position_y=msg.pose.pose.position.y

        # Extract yaw from quaternion
        quat = msg.pose.pose.orientation
        siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def scan_callback(self, msg):
        all_points = self.scan_to_points(msg)
        if len(all_points) < 20:
            return

        filtered_points = self.filter_by_distance(all_points, 2.0)
        if len(filtered_points) < 10:
            return

        binary_image = self.points_to_image(filtered_points)
        raw_lines = self.detect_lines(binary_image)
        if len(raw_lines) == 0:
            return

        line_groups = self.group_by_proximity(raw_lines, 0.3)

        best_group = None
        best_score = -1

        for group in line_groups:
            processed = self.process_lines(group)
            score = self.score_group(processed)
            if score > best_score:
                best_score = score
                best_group = processed

        if not best_group:
            return

        shape_position = np.mean([[l['start'], l['end']] for l in best_group], axis=(0,1))
        shape_distance = np.linalg.norm(shape_position)
        shape_type = self.detect_shape(best_group)
        if self.flag_print==True:

                # if  math.sqrt((self.world_pos[0]-self.position_x)**2 +(self.world_pos[1]-self.position_y)**2) < 10.6:
                if abs(self.world_pos[1]-self.position_y)<0.02:
                    status_msg = String()
                    status_msg.data = f"{self.shape_print},{self.position_x},{self.position_y}"
                    self.shape_pub.publish(status_msg)
                    self.flag_print=False
        if shape_type and shape_type in ['Pentagon', 'Square', 'Triangle']:
            is_new, self.world_pos = self.is_new_detection(shape_position)
            if is_new:
                self.shape_print=self.shape_status_map.get(shape_type)
                self.flag_print=True
                self.detected_shapes.append({
                    'local_position': shape_position,
                    'world_position': self.world_pos,
                    'type': shape_type,
                    'distance': shape_distance
                })
                self.get_logger().info(f'NEW: {shape_type} at local({shape_position[0]:.2f}, {shape_position[1]:.2f}), world({self.world_pos[0]:.2f}, {self.world_pos[1]:.2f})')
                self.get_logger().info(f'Total: {len(self.detected_shapes)}')

            else:
                shape_type = f'{shape_type} (Already Detected)'

        self.visualize(filtered_points, binary_image, best_group, shape_type)

    def detect_shape(self, lines):

        if len(lines) == 2 and lines[1]['length']<0.25:
            angle = self.calc_angle(lines[0], lines[1])
            angle = 180 - angle
            dist = math.sqrt((lines[0]['end'][0]-lines[1]['start'][0])**2 +
                           (lines[0]['end'][1]-lines[1]['start'][1])**2)
            if abs(angle-135)<5 and dist<0.03:
                self.get_logger().info('Detected Triangle')
                print(angle)
                return 'Triangle'

        if len(lines) == 3 and lines[1]['length']<0.3 and lines[2]['length']<0.3:
            angles = [180 - self.calc_angle(lines[i], lines[i+1]) for i in range(len(lines)-1)]
            if abs(angles[0]-90)<5 and abs(angles[1]-140)<5:
                return 'Pentagon'

            elif all(abs(a-90)<5 for a in angles):
                return 'Square'

        if len(lines) == 4:
            angles = [180 - self.calc_angle(lines[i], lines[i+1]) for i in range(len(lines)-1)]
            for i in range(len(lines)-1):
                dist= math.sqrt((lines[i]['end'][0]-lines[i+1]['start'][0])**2 +
                               (lines[i]['end'][1]-lines[i+1]['start'][1])**2)
                if dist>0.03:
                    return None
            if abs(angles[0]-130)<7 and abs(angles[1]-90)<5 and abs(angles[2]-130)<7:
                self.get_logger().info('Detected Triangle')
                print(angles)
                return 'Triangle'

    def local_to_world(self, local_pos):
        """Convert local robot frame coordinates to world coordinates"""
        # Rotation matrix
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)

        # Transform: world = robot_world_pos + rotation * local
        world_x = self.position_x + (local_pos[0] * cos_yaw - local_pos[1] * sin_yaw)
        world_y = self.position_y + (local_pos[0] * sin_yaw + local_pos[1] * cos_yaw)

        return np.array([world_x, world_y])

    def is_new_detection(self, local_position):
        """Check if shape at local position is new based on world coordinates"""
        # Convert local position to world coordinates
        world_position = self.local_to_world(local_position)

        for detected in self.detected_shapes:
            # Compare world coordinates
            if np.linalg.norm(world_position - detected['world_position']) < self.detection_range:
                return False, world_position
        return True, world_position

    def filter_by_distance(self, points, max_dist):
        return points[np.linalg.norm(points, axis=1) <= max_dist]

    def group_by_proximity(self, lines, threshold):
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

    def score_group(self, lines):
        if not lines:
            return 0

        score = 10.0 if 2 <= len(lines) <= 4 else (2.0 if len(lines)==1 else 1.0)

        all_pts = np.array([[l['start'], l['end']] for l in lines]).reshape(-1, 2)
        bbox_size = np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0))

        if 0.15 < bbox_size < 0.8:
            score += 5.0 / (bbox_size + 0.1)

        if np.mean([l['length'] for l in lines]) < 0.3:
            score += 5.0

        score += max(0, 5.0 - np.linalg.norm(all_pts.mean(axis=0)))

        return score

    def scan_to_points(self, msg):
        ranges = np.array(msg.ranges, dtype=float)
        ranges[np.isnan(ranges) | np.isinf(ranges)] = self.max_range + 1
        filtered = self.median_filter(ranges)

        points = []
        for i, r in enumerate(filtered):
            if self.min_range <= r <= self.max_range:
                angle = msg.angle_min + i * msg.angle_increment
                points.append([r * np.cos(angle), r * np.sin(angle)])

        return np.array(points) if points else np.array([]).reshape(0, 2)

    def median_filter(self, data):
        filtered = np.copy(data)
        half = self.median_filter_size // 2

        for i in range(len(data)):
            window = data[max(0, i-half):min(len(data), i+half+1)]
            valid = window[window <= self.max_range]
            if len(valid) > 0:
                filtered[i] = np.median(valid)

        return filtered

    def points_to_image(self, points):
        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        center = self.image_size // 2

        for pt in points:
            px = int(pt[0] * self.pixels_per_meter + center)
            py = int(-pt[1] * self.pixels_per_meter + center)
            if 0 <= px < self.image_size and 0 <= py < self.image_size:
                cv2.circle(image, (px, py), 2, 255, -1)

        return image

    def detect_lines(self, binary_image):
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

    def process_lines(self, lines):
        """Merge collinear segments and remove overlaps"""
        if len(lines) <= 1:
            return lines

        # Merge collinear
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

        # Order by connectivity
        return self.order_lines(filtered)

    def normalize_dir(self, line):
        """Get normalized direction vector"""
        d = line['end'] - line['start']
        length = np.linalg.norm(d)
        return d/length if length > 1e-6 else None, length

    def are_similar(self, line1, line2):
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

        return min_dist < self.merge_distance_tolerance or (perp_dist < 0.04 and min_dist < 0.20)

    def merge_group(self, group):
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

    def check_overlap(self, line1, line2):
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

    def calc_angle(self, line1, line2):
        """Calculate angle between two lines in degrees"""
        dir1, len1 = self.normalize_dir(line1)
        dir2, len2 = self.normalize_dir(line2)

        if dir1 is None or dir2 is None:
            return 0.0

        dot = np.clip(np.dot(dir1, dir2), -1.0, 1.0)
        return np.degrees(np.arccos(np.abs(dot)))

    def order_lines(self, lines):
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

    def visualize(self, points, binary_image, lines, shape_type='Unknown'):
        self.ax1.clear()
        self.ax2.clear()

        self.ax1.imshow(binary_image, cmap='gray')
        self.ax1.set_title('Binary Image', fontsize=12, fontweight='bold')
        self.ax1.axis('off')

        self.ax2.scatter(points[:, 0], points[:, 1], c='lightgray', s=2, alpha=0.4, label='Points')

        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
        for i, line in enumerate(lines):
            color = colors[i % len(colors)]
            self.ax2.plot([line['start'][0], line['end'][0]],
                         [line['start'][1], line['end'][1]],
                         color=color, linewidth=4, alpha=0.9,
                         label=f'L{i+1}: {line["length"]:.2f}m')

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
