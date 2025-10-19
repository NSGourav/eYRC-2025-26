#!/usr/bin/env python3
import os
import random
import subprocess
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory
import cryptocode
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

# -------------------------------
# Shapes configuration
# -------------------------------

triangle_poses_dict = {
    # "11": {"x": -2.5942,   "y": -0.0461,    "z": 0.2505, "roll": 1.57, "pitch": 0.0, "yaw": 0.0},
    # "12": {"x": -2.5942,   "y": -1.4044,    "z": 0.2505, "roll": 1.57, "pitch": 0.0, "yaw": 0.0},
    # "13": {"x": -2.5942,   "y": -2.7702,    "z": 0.2505, "roll": 1.57, "pitch": 0.0, "yaw": 0.0},
    # "14": {"x": -2.5942,   "y": -4.0941,    "z": 0.2505, "roll": 1.57, "pitch": 0.0, "yaw": 0.0},
    "31": {"x": -0.7241,   "y": -0.0461,    "z": 0.2505, "roll": 1.57, "pitch": 0.0, "yaw": 0.0},
    "32": {"x": -0.7241,   "y": -1.4044,    "z": 0.2505, "roll": 1.57, "pitch": 0.0, "yaw": 0.0},
    "33": {"x": -0.7241,   "y": -2.7702,    "z": 0.2505, "roll": 1.57, "pitch": 0.0, "yaw": 0.0},
    "34": {"x": -0.7241,   "y": -4.0941,    "z": 0.2505, "roll": 1.57, "pitch": 0.0, "yaw": 0.0},
}

square_poses_dict = {
    "21": {"x": -2.1946,   "y": -0.0461,    "z": 0.2505, "roll": 1.57, "pitch": 0.0, "yaw": 3.14},
    "22": {"x": -2.1946,   "y": -1.4044,    "z": 0.2505, "roll": 1.57, "pitch": 0.0, "yaw": 3.14},
    "23": {"x": -2.1946,   "y": -2.7702,    "z": 0.2505, "roll": 1.57, "pitch": 0.0, "yaw": 3.14},
    "24": {"x": -2.1946,   "y": -4.0941,    "z": 0.2505, "roll": 1.57, "pitch": 0.0, "yaw": 3.14},
    "41": {"x": -0.3245,   "y": -0.0461,    "z": 0.2505, "roll": 1.57, "pitch": 0.0, "yaw": 3.14},
    # "42": {"x": -0.3245,   "y": -1.4044,    "z": 0.2505, "roll": 1.57, "pitch": 0.0, "yaw": 3.14},
    # "43": {"x": -0.3245,   "y": -2.7702,    "z": 0.2505, "roll": 1.57, "pitch": 0.0, "yaw": 3.14},
    # "44": {"x": -0.3245,   "y": -4.0941,    "z": 0.2505, "roll": 1.57, "pitch": 0.0, "yaw": 3.14},
}

triangle_shape_model = os.path.join(
    get_package_share_directory('eyantra_warehouse'),
    'models', 'triangle_shape', 'model.sdf'
)
square_shape_model = os.path.join(
    get_package_share_directory('eyantra_warehouse'),
    'models', 'square_shape', 'model.sdf'
)

# Randomize spawning shapes
triangle_slots = list(triangle_poses_dict.keys())
square_slots = list(square_poses_dict.keys())
random.shuffle(triangle_slots)
random.shuffle(square_slots)
good_shuffle_achieved = False
while not good_shuffle_achieved:
    random.shuffle(triangle_slots)
    random.shuffle(square_slots)
    
    t0 = triangle_slots[0]
    s0 = square_slots[0]
    s1 = square_slots[1]

    # Check if the distance between any two shapes is less than 2
    if abs(int(t0[1]) - int(s0[1])) < 1:
        continue
    if abs(int(t0[1]) - int(s1[1])) < 2:
        continue
    if abs(int(s0[1]) - int(s1[1])) < 1:
        continue
    
    # Check if squares are in the same column
    if s0[0] == s1[0]:
        continue
        
    good_shuffle_achieved = True
triangle_slots = triangle_slots[:1]
square_slots = square_slots[:2]

# -------------------------------
# ROS 2 Node
# -------------------------------
class ShapePublisher(Node):
    def __init__(self):
        super().__init__('_armed0')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        self.pub = self.create_publisher(String, '/_shape_info', qos_profile)

        # Password for cryptocode
        self.password = "ultimate1103"
        self.success = True

    def publish_shape_info(self, triangle, square):
        data = json.dumps({"status": self.success, "triangle": triangle, "square": square})
        encrypted = cryptocode.encrypt(data, self.password)
        msg = String()
        msg.data = encrypted
        self.pub.publish(msg)

# Spawn function
def spawn_shape(name, model_path, pose):
    cmd = [
        "ros2", "run", "ros_gz_sim", "create",
        "-name", name,
        "-file", model_path,
        "-x", str(pose["x"]),
        "-y", str(pose["y"]),
        "-z", str(pose["z"]),
        "-R", str(pose["roll"]),
        "-P", str(pose["pitch"]),
        "-Y", str(pose["yaw"]),
    ]
    result = subprocess.run(cmd)
    return result.returncode

# Main
def main(args=None):
    rclpy.init(args=args)
    node = ShapePublisher()
    print(f"sssss {node.success}")

    try:
        for slot_name in triangle_slots:
            result = spawn_shape(f"triangle_{slot_name}", triangle_shape_model, triangle_poses_dict[slot_name])
            node.success = (result==0) and node.success
            print(node.success)
        for slot_name in square_slots:
            result = spawn_shape(f"square_{slot_name}", square_shape_model, square_poses_dict[slot_name])
            node.success = (result==0) and node.success
            print(node.success)
        node.get_logger().info(f"status: {1 if node.success else 0}")
    except:
        node.success = False
        node.get_logger().info("status: -1")

    timer_period = 2.0  # seconds
    def timer_callback():
        node.publish_shape_info(triangle_slots, square_slots)

    timer = node.create_timer(timer_period, timer_callback)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
