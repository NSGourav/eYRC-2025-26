#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from builtin_interfaces.msg import Time


class DeltaTwistPublisher(Node):

    def __init__(self):
        super().__init__('delta_twist_publisher')

        self.publisher_ = self.create_publisher(
            TwistStamped,
            '/delta_twist_cmd',
            10
        )

        self.start_time = self.get_clock().now()
        self.timer = self.create_timer(0.05, self.timer_callback)  # 20 Hz

    def timer_callback(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9

        if elapsed_time < 2.0:
            # Publish velocity for 2 seconds
            msg.twist.linear.x = 0.2
            msg.twist.linear.y = 0.0
            msg.twist.linear.z = 0.0
            msg.twist.angular.x = 0.0
            msg.twist.angular.y = 0.0
            msg.twist.angular.z = 0.3
        else:
            # Stop after 2 seconds
            msg.twist.linear.x = 0.0
            msg.twist.angular.z = 0.0
            self.publisher_.publish(msg)
            self.get_logger().info('Stopped publishing velocity')
            self.timer.cancel()
            return

        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DeltaTwistPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
