#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String,Bool
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool
import time
import math
import numpy as np

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

        self.system="SIM"

        # self.square_counter = 0
        # self.triangle_counter = 0

        self.waypoints = [
            (+0.40, -4.30, +1.57),      # 1st lane start
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
            if self.system =="SIM":
                target_yaw = 1.57
            else:
                target_yaw = 0.0  # Default

            current_normalized_yaw = self.current_yaw
            while current_normalized_yaw > math.pi:
                current_normalized_yaw -= 2 * math.pi
            while current_normalized_yaw < -math.pi:
                current_normalized_yaw += 2 * math.pi

            if self.system =="SIM":
                target_yaw = 1.57 if abs(current_normalized_yaw - 1.57) < abs(current_normalized_yaw + 1.57)  else -1.57
            else:
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
                    
                    goal_x, goal_y = self.priority_goal[1], self.priority_goal[2]
                    shape_pos = [goal_x, goal_y]
                    plant_id = self.assign_plant_id(shape_pos, self.current_x, self.current_y)
                    self.get_logger().info(f"Square: Goal=({goal_x:.3f}, {goal_y:.3f}), Actual=({goal_x:.3f}, {goal_y:.3f}), Plant ID={plant_id}")
                    self.publish_dock_status("BAD_HEALTH", plant_id)

                    time.sleep(3)

                if shape_type == "Triangle":

                    goal_x, goal_y = self.priority_goal[1], self.priority_goal[2]

                    shape_pos = [goal_x, goal_y]
                    plant_id = self.assign_plant_id(shape_pos, self.current_x, self.current_y)
                    self.get_logger().info(f"Triangle: Goal=({goal_x:.3f}, {goal_y:.3f}), Actual=({goal_x:.3f}, {goal_y:.3f}), Plant ID={plant_id}")
                    self.publish_dock_status("FERTILIZER_REQUIRED", plant_id)

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

    def assign_plant_id(self, shape_world_pos: np.ndarray, robot_x: float, robot_y: float):
        """
        Assign plant_id (0-8) based on bounding boxes.
        Returns plant_id if shape is within any bounding box, else returns 0.
        """
        # Define bounding boxes for each plant ID
        bounding_boxes = {
            1: [(0.3335, -5.1725), (0.3718, -3.4826), (-1.4512, -5.1725), (-1.4499, -3.4548)],
            2: [(0.3718, -3.4826), (0.3729, -2.1404), (-1.4499, -3.4548), (-1.4488, -2.1265)],
            3: [(0.3729, -2.1404), (0.3740, -0.7632), (-1.4488, -2.1265), (-1.4477, -0.7380)],
            4: [(0.3740, -0.7632), (0.3752, 0.8777), (-1.4477, -0.7380), (-1.4464, 0.8791)],
            5: [(-1.4512, -5.1725), (-1.4499, -3.4548), (-3.3366, -5.1725), (-3.3352, -3.4357)],
            6: [(-1.4499, -3.4548), (-1.4488, -2.1265), (-3.3352, -3.4357), (-3.3342, -2.1359)],
            7: [(-1.4488, -2.1265), (-1.4477, -0.7380), (-3.3342, -2.1359), (-3.3331, -0.7601)],
            8: [(-1.4477, -0.7380), (-1.4464, 0.8791), (-3.3331, -0.7601), (-3.3317, 0.9461)]
        }
        
        def point_in_polygon(point, polygon):
            """Check if a point is inside a polygon using ray casting algorithm."""
            x, y = point
            n = len(polygon)
            inside = False
            
            p1x, p1y = polygon[0]
            for i in range(1, n + 1):
                p2x, p2y = polygon[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
                p1x, p1y = p2x, p2y
            
            return inside
        
        # Check which plant ID the shape belongs to
        shape_x, shape_y = shape_world_pos[0], shape_world_pos[1]
        
        for plant_id, bbox in bounding_boxes.items():
            if point_in_polygon((shape_x, shape_y), bbox):
                return plant_id
        
        # If not in any bounding box, return 0
        return 0
    
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