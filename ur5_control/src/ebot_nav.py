#!/usr/bin/python3
import rclpy
from rclpy.node import Node  
from std_msgs.msg import String,Bool
import time
class EbotNav(Node):
    def __init__(self):
        super().__init__('ebot_nav')
        self.pose_pub=self.create_publisher(String,"/set_immediate_goal",10)
        self.create_subscription(Bool,"/immediate_goal_status",self.goal_reached_callback,10)
        self.waypoints = [
            (0.3,-5.0, 1.57),
            (0.3,-3.0, 1.57),
            (0.3,-2.0, 1.57),
            (0.3,-1.0, 1.57)]
        self.i = 0

        # Publish first waypoint directly (no timer needed)
        self.get_logger().info("ebot_nav initialized. Publishing first waypoint...")
        self.publish_next_waypoint()

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
        # self.get_logger().info(f"Goal reached message received: {msg.data}")
        if msg.data == True:
            self.publish_next_waypoint()
            
def main(args=None):
    rclpy.init(args=args)
    node = EbotNav()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
