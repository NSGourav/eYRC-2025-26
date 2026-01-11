#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String,Bool
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool
import time
import math

class EbotNav(Node):
    def __init__(self):
        super().__init__('ebot_nav')

        self.create_subscription(Odometry,"/odom",self.odom_callback,10)
        self.create_subscription(String,"/shape_pose",self.shape_pose_recieved_callback,10)
        self.create_subscription(Bool,"/immediate_goal_status",self.goal_reached_callback,10)
        self.shape_pub = self.create_publisher(String, '/detection_status', 10)
        self.pose_pub = self.create_publisher(String,"/set_immediate_goal",10)

        self.pick_place_client = self.create_client(SetBool, '/pick_and_place')
        self.waiting_for_pick_place = False
        
        self.current_x = None
        self.current_y = None
        self.current_yaw = None

        self.square_counter = 0
        self.triangle_counter = 0

        self.waypoints = [
            (+0.40, -5.30, +1.57),      # 1st lane start
            (+0.53, -1.95, +1.57),      # Dock Station
            (+0.40, +1.10, +1.57),      # 1st lane end
            (-1.53, +1.10, -1.57),      # 2st lane start
            (-1.53, -5.50, -1.57),      # 2st lane end
            (-3.56, -5.50, +1.57),      # 3st lane start
            (-3.56, +1.10, +1.57),      # 3st lane end
            (+0.40, +1.10, +0.00),      # 1st lane end
            (+0.40, +1.10, -1.57),      # 1st lane end
            (+0.53, -1.95, -1.57),      # Dock Station
            (-1.53, -6.61, -1.57)       # Home position  
        ]

        self.waypoint_counter = 0

        # Priority navigation state
        self.priority_goal = None               # Stores (shape_type, x, y) for priority navigation
        self.interrupted_waypoint_index = None  # Index of waypoint we were going to

        # Publish first waypoint directly (no timer needed)
        self.get_logger().info("ebot_nav initialized. Publishing first waypoint...")
        self.publish_next_waypoint()

    def odom_callback(self,msg:Odometry):

        self.current_x=msg.pose.pose.position.x
        self.current_y=msg.pose.pose.position.y

        orientation_quat = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation_quat.w * orientation_quat.z + orientation_quat.x * orientation_quat.y)
        cosy_cosp = 1 - 2 * (orientation_quat.y * orientation_quat.y + orientation_quat.z * orientation_quat.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def publish_next_waypoint(self):

        if self.waypoint_counter < len(self.waypoints):
            next_waypoint = self.waypoints[self.waypoint_counter]
            msg = String()
            msg.data = f"{next_waypoint[0]},{next_waypoint[1]},{next_waypoint[2]}"
            self.pose_pub.publish(msg)
            self.get_logger().info(f"Published waypoint: {msg.data}")
            self.waypoint_counter += 1
            time.sleep(1)  # Small delay to ensure message is sent
        else:
            self.get_logger().info("All waypoints reached.")

    def call_pick_and_place_service(self, data_value: bool):

        req = SetBool.Request()
        req.data = data_value

        self.waiting_for_pick_place = True
        future = self.pick_place_client.call_async(req)
        future.add_done_callback(self.pick_place_response_callback)

    def pick_place_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Pick and place service completed successfully")
            else:
                self.get_logger().warn("Pick and place service failed")

        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

        self.waiting_for_pick_place = False
        self.publish_next_waypoint()

    def shape_pose_recieved_callback(self, msg):

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
                self.interrupted_waypoint_index = self.waypoint_counter - 1  # Current waypoint we're navigating to
                self.get_logger().info(f"Interrupting navigation to waypoint {self.interrupted_waypoint_index}")

            # Store priority goal
            self.priority_goal = (shape_type, x, y)

            # Choose yaw (1.57 or -1.57) based on which is closer to current yaw
            target_yaw = 0.0  # Default
            
            while self.current_yaw > math.pi:
                current_normalized_yaw -= 2 * math.pi
            while self.current_yaw < -math.pi:
                current_normalized_yaw += 2 * math.pi
            
            target_yaw = 0.0 if abs(current_normalized_yaw) < math.pi/2 else math.pi
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
                    self.square_counter += 1
                    if self.square_counter == 1:
                        self.publish_dock_status("BAD_HEALTH",3)
                    elif self.square_counter == 2:
                        self.publish_dock_status("BAD_HEALTH",7)
                    time.sleep(3)

                if shape_type == "Triangle":
                    self.triangle_counter+=1
                    if self.triangle_counter == 1:
                        self.publish_dock_status("FERTILIZER_REQUIRED",4)
                    elif self.triangle_counter == 2:
                        self.publish_dock_status("FERTILIZER_REQUIRED",5)
                    elif self.triangle_counter == 3:
                        self.publish_dock_status("FERTILIZER_REQUIRED",7)
                    time.sleep(3)

                # Resume navigation to interrupted waypoint
                self.get_logger().info(f"Resuming navigation to waypoint {self.interrupted_waypoint_index}")
                self.waypoint_counter = self.interrupted_waypoint_index

                self.priority_goal = None
                self.interrupted_waypoint_index = None

                self.publish_next_waypoint()

            else:
                if (self.waypoint_counter == 2 or self.waypoint_counter == 10) and not self.waiting_for_pick_place:
                    self.get_logger().info("Reached Dock Station. Triggering pick and place service.")
                    self.publish_dock_status("DOCK_STATION",0)
                    time.sleep(2)
                    if self.waypoint_counter == 2:
                        service_data = False
                    elif self.waypoint_counter == 10:
                        service_data = True
                    self.call_pick_and_place_service(service_data)
                    return
                else:
                    if not self.waiting_for_pick_place:
                        self.publish_next_waypoint()

    def publish_dock_status(self,shape_status,plant_id):
            status_msg = String()
            status_msg.data = f"{shape_status},{self.current_x},{self.current_y},{plant_id}"
            self.shape_pub.publish(status_msg)
            self.get_logger().info(f"<<<<Published SHAPE>>>>")
            time.sleep(0.1)

def main(args=None):
    rclpy.init(args=args)
    node = EbotNav()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
