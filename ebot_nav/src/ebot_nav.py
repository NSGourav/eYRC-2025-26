#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String,Bool
from nav_msgs.msg import Odometry
import time
import math

class EbotNav(Node):
    def __init__(self):
        super().__init__('ebot_nav')
        self.pose_pub=self.create_publisher(String,"/set_immediate_goal",10)
        self.create_subscription(Odometry,"/odom",self.odom_callback,10)
        self.create_subscription(String,"/shape_pose",self.shape_pose_recieved_callback,10)
        self.create_subscription(Bool,"/immediate_goal_status",self.goal_reached_callback,10)
        self.shape_pub=self.create_publisher(String, '/detection_status', 10)

        self.current_x=None
        self.current_y=None
        self.current_yaw=None

        # self.waypoints = [
        #     (0.3,-5.0, 1.57),
        #     (0.3,-3.0, 1.57),
        #     (0.3,-2.0, 1.57),
        #     (0.3,-1.0, 1.57)]
        # only for sim use
        # self.waypoints = [
        #     (0.3,-5.5, 1.57),
        #     (0.53, -1.95, 1.57),            # P1: (x1, y1, yaw1)
        #     (0.26, 1.1, 1.57),
        #     (-1.53, 1.1, -1.57),
        #     (-1.53, -5.52, -1.57),
        #     (-3.56, -5.52, +1.57),
        #     (-3.56, 1.1, +1.57),         # P2: (x2, y2, yaw2)
        #     (-1.53, -6.61, -1.57)           # P3: (x3, y3, yaw3)
        # ]
        self.waypoints=  [
            (1.03, -1.857, 1.57),           # 1st lane start
            (2.072, -1.704, 1.57),           # Dock station
            (5.035, -1.857, 1.57),           # 1st lane end
            (4.495, 0.006, -1.57),           # 2nd lane start
            (1.448, 0.072, -1.57),           # 2nd lane end
            (1.108, 1.656, 1.57),            # 3rd lane start
            (5.379, 1.955, 1.57),            # 3rd lane end
            (0.0, 0.0, -1.57)]               # Home
        self.i = 0

        # Priority navigation state
        self.priority_goal = None  # Stores (shape_type, x, y) for priority navigation
        self.interrupted_waypoint_index = None  # Index of waypoint we were going to

        # Publish first waypoint directly (no timer needed)
        self.get_logger().info("ebot_nav initialized. Publishing first waypoint...")
        self.publish_next_waypoint()

    def odom_callback(self,msg:Odometry):
        self.current_x=msg.pose.pose.position.x
        self.current_y=msg.pose.pose.position.y

        # Extract yaw from quaternion
        orientation_q = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y * orientation_q.y + orientation_q.z * orientation_q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
        
    def publish_next_waypoint(self):
        if self.waypoints and self.i < len(self.waypoints):
            next_wp = self.waypoints[self.i]
            msg = String()
            msg.data = f"{next_wp[0]},{next_wp[1]},{next_wp[2]}"
            self.pose_pub.publish(msg)
            self.get_logger().info(f"Published waypoint: {msg.data}")
            self.i += 1
            time.sleep(1)  # Small delay to ensure message is sent
        else:
            self.get_logger().info("All waypoints reached.")

    def shape_pose_recieved_callback(self, msg):
        """Handle priority navigation when a shape is detected
        Format: "Shape,x,y" e.g., "Square,1.5,2.3"
        """
        try:
            parts = msg.data.split(',')
            if len(parts) != 3:
                self.get_logger().error(f"Invalid shape pose format: {msg.data}")
                return

            shape_type = parts[0]
            x = float(parts[1])
            y = float(parts[2])

            self.get_logger().info(f"PRIORITY: {shape_type} detected at ({x:.3f}, {y:.3f})")

            # Store current waypoint index if not already interrupted
            if self.priority_goal is None:
                self.interrupted_waypoint_index = self.i - 1  # Current waypoint we're navigating to
                self.get_logger().info(f"Interrupting navigation to waypoint {self.interrupted_waypoint_index}")

            # Store priority goal
            self.priority_goal = (shape_type, x, y)

            # Choose yaw (1.57 or -1.57) based on which is closer to current yaw
            target_yaw = 1.57  # Default
            if self.current_yaw is not None:
                # Calculate angular differences
                diff_1_57 = abs(self._normalize_angle(self.current_yaw - 1.57))
                diff_neg_1_57 = abs(self._normalize_angle(self.current_yaw - (-1.57)))
                target_yaw = 1.57 if diff_1_57 < diff_neg_1_57 else -1.57
                self.get_logger().info(f"Current yaw: {self.current_yaw:.3f}, choosing target yaw: {target_yaw:.3f}")

            # Immediately publish priority goal
            priority_msg = String()
            priority_msg.data = f"{x:.3f},{y:.3f},{target_yaw:.2f}"
            self.pose_pub.publish(priority_msg)
            self.get_logger().info(f"Published PRIORITY goal: {priority_msg.data}")

        except Exception as e:
            self.get_logger().error(f"Error in shape_pose_recieved_callback: {str(e)}")

    def goal_reached_callback(self, msg):
        if msg.data == True:
            # Check if we just completed a priority goal
            if self.priority_goal is not None:
                shape_type, x, y = self.priority_goal
                self.get_logger().info(f"Reached PRIORITY goal: {shape_type} at ({x:.3f}, {y:.3f})")
                if shape_type == "Square":
                    self.publish_dock_status("BAD_HEALTH",0)
                if shape_type == "Triangle":
                    self.publish_dock_status("FERTILIZER_REQUIRED",0)

                # Clear priority goal
                self.priority_goal = None

                # Resume navigation to interrupted waypoint
                if self.interrupted_waypoint_index is not None:
                    self.get_logger().info(f"Resuming navigation to waypoint {self.interrupted_waypoint_index}")
                    self.i = self.interrupted_waypoint_index
                    self.interrupted_waypoint_index = None
                    self.publish_next_waypoint()
                else:
                    # Continue with next waypoint
                    self.publish_next_waypoint()
            else:
                # Normal waypoint reached
                if self.i==2:
                    self.publish_dock_status("DOCK_STATION",0)
                    time.sleep(2)
                self.publish_next_waypoint()

    def publish_dock_status(self,shape_status,plant_id):
            status_msg = String()
            status_msg.data = f"{shape_status},{self.current_x},{self.current_y},{plant_id}"
            self.shape_pub.publish(status_msg)
            time.sleep(0.1)

    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi] range"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
            
def main(args=None):
    rclpy.init(args=args)
    node = EbotNav()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()
