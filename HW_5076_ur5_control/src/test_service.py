#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool


class PickAndPlaceTestClient(Node):

    def __init__(self):
        super().__init__('pick_and_place_test_client')

        self.client = self.create_client(SetBool, 'pick_and_place')

        # Wait until service is available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for pick_and_place service...')

        self.get_logger().info('pick_and_place service available')

    def send_request(self, enable: bool):
        request = SetBool.Request()
        request.data = enable

        self.get_logger().info(f'Sending request: data={enable}')

        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            response = future.result()
            self.get_logger().info(
                f'Response received -> success: {response.success}, '
                f'message: "{response.message}"'
            )
        else:
            self.get_logger().error('Service call failed')


def main(args=None):
    rclpy.init(args=args)

    node = PickAndPlaceTestClient()

    # Trigger pick-and-place (True = start)
    node.send_request(True)

    # Optional: trigger stop/reset
    # node.send_request(False)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
