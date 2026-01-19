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

        self.system = "HW"

        self.waypoints = [
            (0.0, 0.0, -1.57),
            (0.7, -1.857, 0.0),             # 1st lane start
            (2.4, -1.6, 0.0),           # Dock station
            (4.5, -1.6, 0.0),           # 1st lane end
            (4.5, 0.1, 3.14),           # 2nd lane start
            (1.0, 0.072, 3.14),           # 2nd lane end
            (1.1, 1.656, 0.0),            # 3rd lane start
            (4.5, 1.955, 0.0),              # 3rd lane end
            (1.1, 1.656, 0.0),            # 3rd lane start
            (0.0, 0.0, 3.14)                # Home
        ]

        self.waypoint_counter = 0

        # Priority navigation state
        self.priority_goal = None               # Stores (shape_type, x, y, yaw, plant_id) for priority navigation
        self.interrupted_waypoint_index = None  # Index of waypoint we were going to

        # Goal management
        self.goal_reached = False
        self.is_moving = False
        self.current_goal_x = None
        self.current_goal_y = None
        self.goal_queue = []  # Queue to store pending goals: [('waypoint', index) or ('priority', shape_type, x, y, yaw, plant_id)]

        # Create timer for control loop
        self.control_timer = self.create_timer(0.1, self.control)

        self.get_logger().info("ebot_nav initialized.")

    def odom_callback(self,msg:Odometry):

        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        orientation_quat = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation_quat.w * orientation_quat.z + orientation_quat.x * orientation_quat.y)
        cosy_cosp = 1 - 2 * (orientation_quat.y * orientation_quat.y + orientation_quat.z * orientation_quat.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def call_pick_and_place_service(self, data_value: bool):

        req = SetBool.Request()
        req.data = data_value

        self.waiting_for_pick_place = True
        future = self.pick_place_client.call_async(req)

        def pick_place_response_callback(future):
            try:
                response = future.result()
                if response.success:
                    self.get_logger().info("Pick and place service completed successfully")
                else:
                    self.get_logger().warn("Pick and place service failed")

            except Exception as e:
                self.get_logger().error(f"Service call failed: {e}")

            self.waiting_for_pick_place = False
            self.goal_reached = False
            self.waypoint_counter += 1

        future.add_done_callback(pick_place_response_callback)

    def publish_shape_status(self,shape_status,plant_id):
            status_msg = String()
            status_msg.data = f"{shape_status},{self.current_x},{self.current_y},{plant_id}"
            self.shape_pub.publish(status_msg)
            self.get_logger().info(f"<<<<Published SHAPE>>>>")
            time.sleep(0.1)

    def publish_waypoint(self, x, y, yaw):

        msg = String()
        msg.data = f"{x},{y},{yaw}"
        self.pose_pub.publish(msg)
        self.get_logger().info(f"Published waypoint: {msg.data}")
        time.sleep(0.1)

    def shape_pose_recieved_callback(self, msg):

        try:
            parts = msg.data.split(',')
            if len(parts) != 4:
                self.get_logger().error(f"Invalid shape pose format: {msg.data}")
                return

            shape_type = parts[0]
            x = float(parts[1])
            y = float(parts[2])
            plant_id = float(parts[3])

            self.get_logger().info(f"{shape_type} detected at ({x:.3f}, {y:.3f})")

            # Choose yaw (1.57 or -1.57) based on which is closer to current yaw
            current_normalized_yaw = self.current_yaw
            while current_normalized_yaw > math.pi:
                current_normalized_yaw -= 2 * math.pi
            while current_normalized_yaw < -math.pi:
                current_normalized_yaw += 2 * math.pi

            if self.system == "SIM":
                target_yaw = 1.57 if abs(current_normalized_yaw - 1.57) < abs(current_normalized_yaw + 1.57)  else -1.57
            else:
                target_yaw = 0.0 if abs(current_normalized_yaw) < math.pi/2 else math.pi

            # Create new priority goal
            new_priority_goal = ('priority', shape_type, x, y, target_yaw, plant_id)

            # If robot is currently moving to a goal, compare distances
            if self.is_moving and self.current_goal_x is not None and self.current_x is not None:
                # Calculate distances
                dist_to_current_goal = math.sqrt((self.current_goal_x - self.current_x)**2 + (self.current_goal_y - self.current_y)**2)
                dist_to_new_priority = math.sqrt((x - self.current_x)**2 + (y - self.current_y)**2)

                self.get_logger().info(f"Distance to current goal: {dist_to_current_goal:.3f}m")
                self.get_logger().info(f"Distance to new priority: {dist_to_new_priority:.3f}m")

                # If new priority is closer, swap them
                if dist_to_new_priority < dist_to_current_goal:
                    self.get_logger().info("New priority is closer! Swapping goals.")
                    # Add current goal to queue
                    if self.priority_goal is not None:
                        # Current goal is a priority goal
                        self.goal_queue.append(self.priority_goal)
                    else:
                        # Current goal is a waypoint
                        self.goal_queue.append(('waypoint', self.interrupted_waypoint_index if self.interrupted_waypoint_index is not None else self.waypoint_counter))

                    # Make new priority the current goal
                    self.priority_goal = new_priority_goal
                    if self.interrupted_waypoint_index is None:
                        self.interrupted_waypoint_index = self.waypoint_counter
                    # Reset goal reached to trigger new navigation
                    self.goal_reached = True
                else:
                    self.get_logger().info("Current goal is closer. Adding new priority to queue.")
                    # Add new priority to queue
                    self.goal_queue.append(new_priority_goal)
            else:
                # Not moving or can't calculate distance, set as priority goal
                if self.priority_goal is not None:
                    # Already have a priority goal, add to queue
                    self.goal_queue.append(new_priority_goal)
                else:
                    self.priority_goal = new_priority_goal
                    if self.interrupted_waypoint_index is None:
                        self.interrupted_waypoint_index = self.waypoint_counter

        except Exception as e:
            self.get_logger().error(f"Error in shape_pose_recieved_callback: {str(e)}")

    def goal_reached_callback(self, msg):
        if msg.data == True:
            self.goal_reached = True
            self.is_moving = False
            self.get_logger().info("Goal reached!")

    def control(self):
        # Check if we're waiting for pick and place service
        if self.waiting_for_pick_place:
            return

        # Check if we've completed all waypoints and no pending goals
        if self.waypoint_counter >= len(self.waypoints) and len(self.goal_queue) == 0 and self.priority_goal is None:
            self.get_logger().info("All waypoints completed!")
            return

        # If currently moving and goal not reached, just spin to process callbacks
        if self.is_moving and not self.goal_reached:
            rclpy.spin_once(self, timeout_sec=0.01)
            return

        # Goal was reached or we're ready for a new goal
        if self.goal_reached or not self.is_moving:

            # Check if we have a priority goal to handle
            if self.priority_goal is not None:
                shape_type, x, y, yaw, plant_id = self.priority_goal[1], self.priority_goal[2], self.priority_goal[3], self.priority_goal[4], self.priority_goal[5]

                # If we just reached the priority goal, handle it
                if self.goal_reached and not self.is_moving and abs(self.current_goal_x - x) < 0.01 and abs(self.current_goal_y - y) < 0.01:
                    self.get_logger().info(f"Reached PRIORITY goal: {shape_type} at ({x:.3f}, {y:.3f})")

                    if shape_type == "Square":
                        self.get_logger().info(f"Square: Goal=({x:.3f}, {y:.3f}), Actual=({self.current_x:.3f}, {self.current_y:.3f}), Plant ID={plant_id}")
                        self.publish_shape_status("BAD_HEALTH", plant_id)
                        time.sleep(2)

                    elif shape_type == "Triangle":
                        self.get_logger().info(f"Triangle: Goal=({x:.3f}, {y:.3f}), Actual=({self.current_x:.3f}, {self.current_y:.3f}), Plant ID={plant_id}")
                        self.publish_shape_status("FERTILIZER_REQUIRED", plant_id)
                        time.sleep(2)

                    # Clear priority goal
                    self.priority_goal = None
                    self.goal_reached = False

                    # Check if there's another goal in queue
                    if len(self.goal_queue) > 0:
                        # Find the closest goal in the queue based on current position
                        min_dist = float('inf')
                        closest_index = 0

                        for i, goal in enumerate(self.goal_queue):
                            if goal[0] == 'priority':
                                goal_x, goal_y = goal[2], goal[3]
                            elif goal[0] == 'waypoint':
                                waypoint = self.waypoints[goal[1]]
                                goal_x, goal_y = waypoint[0], waypoint[1]

                            dist = math.sqrt((goal_x - self.current_x)**2 + (goal_y - self.current_y)**2)
                            if dist < min_dist:
                                min_dist = dist
                                closest_index = i

                        # Pop the closest goal from queue
                        next_goal = self.goal_queue.pop(closest_index)
                        self.get_logger().info(f"Closest goal in queue at distance {min_dist:.3f}m")

                        if next_goal[0] == 'priority':
                            self.priority_goal = next_goal
                            self.get_logger().info(f"Queue has priority goal. Processing it next.")
                        elif next_goal[0] == 'waypoint':
                            # It's a waypoint - continue to that waypoint
                            self.waypoint_counter = next_goal[1]
                            self.get_logger().info(f"Queue has waypoint {self.waypoint_counter}. Continuing to it.")

                    return
                else:
                    # Navigate to priority goal
                    self.get_logger().info(f"Navigating to PRIORITY goal: {shape_type} at ({x:.3f}, {y:.3f})")
                    self.publish_waypoint(x, y, yaw)
                    self.current_goal_x = x
                    self.current_goal_y = y
                    self.is_moving = True
                    self.goal_reached = False
                    return

            # No priority goal, handle waypoint navigation
            if self.waypoint_counter < len(self.waypoints):
                waypoint = self.waypoints[self.waypoint_counter]
                x, y, yaw = waypoint

                # Check if this is a dock station
                if (self.waypoint_counter == 2 or self.waypoint_counter == 10):
                    if self.goal_reached:
                        self.get_logger().info("Reached Dock Station. Publishing status.")
                        self.publish_shape_status("DOCK_STATION", 0)
                        time.sleep(2)

                        # Uncomment when ready to use pick and place
                        service_data = False if self.waypoint_counter == 2 else True
                        self.call_pick_and_place_service(service_data)
                        return

                        # # Increment waypoint counter
                        # self.waypoint_counter += 1
                        # self.goal_reached = False

                        # # Check if there's a goal in queue before continuing
                        # if len(self.goal_queue) > 0:
                        #     next_goal = self.goal_queue.pop(0)
                        #     if next_goal[0] == 'priority':
                        #         self.priority_goal = next_goal
                        #         if self.interrupted_waypoint_index is None:
                        #             self.interrupted_waypoint_index = self.waypoint_counter
                        #         self.get_logger().info(f"Queue has priority goal. Processing it before waypoint {self.waypoint_counter}")
                        # else:
                        #     self.interrupted_waypoint_index = None
                        # return

                # Check if we just reached this waypoint
                if self.goal_reached:
                    self.waypoint_counter += 1
                    self.goal_reached = False

                    # Check if there's a goal in queue
                    if len(self.goal_queue) > 0:
                        next_goal = self.goal_queue.pop(0)
                        if next_goal[0] == 'priority':
                            self.priority_goal = next_goal
                            if self.interrupted_waypoint_index is None:
                                self.interrupted_waypoint_index = self.waypoint_counter
                            self.get_logger().info(f"Queue has priority goal. Processing it before waypoint {self.waypoint_counter}")
                    else:
                        self.interrupted_waypoint_index = None
                    return

                # Publish the waypoint
                self.publish_waypoint(x, y, yaw)
                self.get_logger().info(f"Moving to waypoint {self.waypoint_counter}: ({x:.3f}, {y:.3f}, {yaw:.3f})")
                self.current_goal_x = x
                self.current_goal_y = y
                self.is_moving = True
                self.goal_reached = False

def main(args=None):
    rclpy.init(args=args)
    node = EbotNav()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()