#!/usr/bin/python3
import rclpy
from rclpy.node import Node  
from std_msgs.msg import String,Bool
from nav_msgs.msg import Odometry
import time

class EbotNav(Node):
    def __init__(self):
        super().__init__('ebot_nav')
        self.pose_pub=self.create_publisher(String,"/set_immediate_goal",10)
        self.create_subscription(Odometry,"/odom",self.odom_callback,10)

        self.create_subscription(Bool,"/immediate_goal_status",self.goal_reached_callback,10)
        self.shape_pub=self.create_publisher(String, '/detection_status', 10)

        self.current_x=None
        self.current_y=None

        self.waypoints = [
            (0.3,-5.0, 1.57),
            (0.3,-3.0, 1.57),
            (0.3,-2.0, 1.57),
            (0.3,-1.0, 1.57)]
        
        # self.waypoints=  [5.035, -1.857, 1.57],           # 1st lane end
            # [4.495, 0.006, -1.57],           # 2nd lane start
            # [1.448, 0.072, -1.57],           # 2nd lane end
            # [1.108, 1.656, 1.57],            # 3rd lane start
            # [5.379, 1.955, 1.57],            # 3rd lane end
            # [0.0, 0.0, -1.57]               # Home
        self.i = 0

        # Publish first waypoint directly (no timer needed)
        self.get_logger().info("ebot_nav initialized. Publishing first waypoint...")
        self.publish_next_waypoint()

    def odom_callback(self,msg:Odometry):
        self.current_x=msg.pose.pose.position.x
        self.current_y=msg.pose.pose.position.y
        
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

    def goal_reached_callback(self, msg):
        if msg.data == True:
            if self.i==2:
                self.publish_detection_status()
                time.sleep(2)
            self.publish_next_waypoint()

    def publish_detection_status(self):
            status_msg = String()
            status_msg.data = f"DOCK_STATION,{self.current_x},{self.current_y},0"
            self.shape_pub.publish(status_msg)
            time.sleep(0.1)
            
def main(args=None):
    rclpy.init(args=args)
    node = EbotNav()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
