#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from math import atan2, sqrt, pi
from tf_transformations import euler_from_quaternion

class ebotNav(Node):
    def __init__(self):
        super().__init__('ebot_nav_task1A')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Robot initial state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 1.57  # Facing forwards
        self.laser_data = []

        self.forward_inflation = 0.4  # Meters
        self.side_inflation = 0.4     # Meters

        self.waypoints = [
            [-1.53, -1.95, 1.57],       # P1: (x1, y1, yaw1)
            [0.13, 1.24, 0.0],          # P2: (x2, y2, yaw2)
            [0.38, -3.32, -1.57]        # P3: (x3, y3, yaw3)
        ]
        self.w_index=0
        self.timer = self.create_timer(0.1, self.control_loop)  # 1.25 Hz control loop timer

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        # Conversion of quaternion to euler for yaw
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_yaw = yaw

        # siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        # cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        # self.current_yaw = atan2(siny_cosp, cosy_cosp)
    
    def scan_callback(self, msg: LaserScan):
        self.laser_data = msg.ranges

    def control_loop(self):
        if self.w_index >= len(self.waypoints):
            self.stop_robot()
            self.get_logger().info('All waypoints reached.')
            return
        
        target_x, target_y, target_yaw = self.waypoints[self.w_index]
        dx = target_x - self.current_x
        dy = target_y - self.current_y
        distance = sqrt(dx ** 2 + dy ** 2)
            
        angle_to_target = atan2(dy, dx)
        angle_diff = self.normalize_angle(angle_to_target - self.current_yaw)

        yaw_error = self.normalize_angle(target_yaw - self.current_yaw)
        
        cmd_msg = Twist()

        if distance > 0.1:  # If not close enough to the waypoint
            m= self.is_obstacle_ahead()
            if m:
                self.obstacle_avoidance(m)
                return
            else:
                if abs(angle_diff) > 0.2:  # If not facing the waypoint
                    cmd_msg.angular.z = 0.3 * angle_diff
                    cmd_msg.linear.x = 0.1
                else:
                    cmd_msg.linear.x = 0.7
        elif abs(yaw_error) > 0.1:  # Align to final yaw
            self.get_logger().info(f'Aligning to final yaw at waypoint {self.w_index + 1}')
            cmd_msg.angular.z = 1 * yaw_error
        else:
            # Reached the waypoint, move to the next one
            self.get_logger().info(f'Reached waypoint {self.w_index + 1},({self.current_x,self.current_y,self.current_yaw})')
            self.w_index += 1

        self.cmd_pub.publish(cmd_msg)
    
    def is_obstacle_ahead(self) -> int:
        """Check if there’s an obstacle within safety distance in front of the robot."""
        if not self.laser_data:
            return False

        # Take readings in a forward sector (e.g., -30° to +30°)
        ranges_f = [r for r in self.laser_data[0:35] + self.laser_data[-35:] if r > 0.0]
        if ranges_f and min(ranges_f) < self.forward_inflation:
            return 1 if self.laser_data.index(min(ranges_f)) < 35 else -1  # Right or left        
        return 0
    
    def obstacle_avoidance(self, direction: int):
        self.get_logger().warn("Obstacle detected! Rotating...")
        cmd = Twist()
        cmd.angular.z = 0.5*direction
        cmd.linear.x = 0.7
        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        stop_msg = Twist()
        self.cmd_pub.publish(stop_msg)

    @staticmethod
    def normalize_angle(angle):
        while angle > pi:
            angle -= 2.0 * pi
        while angle < -pi:
            angle += 2.0 * pi
        return angle



def main(args=None):
    rclpy.init(args=args)
    ebot_nav = ebotNav()
    rclpy.spin(ebot_nav)
    ebot_nav.destroy_node()
    rclpy.shutdown()    

if __name__ == '__main__':
    main()