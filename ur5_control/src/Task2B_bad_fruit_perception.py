#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
*****************************************************************************************
*
*        		===============================================
*           		    Krishi coBot (KC) Theme (eYRC 2025-26)
*        		===============================================
*
*  This script should be used to implement Task 1B of Krishi coBot (KC) Theme (eYRC 2025-26).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''
  
# Team ID:          [ eYRC#5076 ]
# Author List:		[ Nagulapalli Shavaneeth Gourav, Pradeep J, Anand Vardhan, Raj Mohammad ]
# Filename:		    Task3A_bad_fruit_perception.py
# Functions:
#			        [ depthimagecb, colorimagecb, bad_fruit_detection, process_image ]
# Nodes:		    Add your publishing and subscribing node
#			        Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/image_raw, /camera/depth/image_raw ]

import rclpy
import cv2
import numpy as np 
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError 
from geometry_msgs.msg import TransformStamped, PointStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener 

# runtime parameters
SHOW_IMAGE = True

class FruitsTF(Node):

    def __init__(self):
        
        """Initialize the FruitsTF node with subscribers, publishers, and TF components."""
        super().__init__('fruits_tf')
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None
        self.cb_group = ReentrantCallbackGroup()

        # Subscriptions
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.colorimagecb, 10, callback_group=self.cb_group)             
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depthimagecb, 10, callback_group=self.cb_group) 
        # Subscriptions
        # self.create_subscription(Image, '/camera/image_raw', self.colorimagecb, 10, callback_group=self.cb_group)             # Cnanged Subscription from /camera/color/image_raw to /camera/image_raw
        # self.create_subscription(Image, '/camera/depth/image_raw', self.depthimagecb, 10, callback_group=self.cb_group)       # Changed Subscription from /camera/depth/image_rect_raw to /camera/depth/image_raw
      

        # Timer for periodic processing
        self.create_timer(0.2, self.process_image, callback_group=self.cb_group)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()                                               
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.team_id = "5076"

        self.get_logger().info("FruitsTF boilerplate node started.")

    # ---------------- Callbacks ----------------
    def depthimagecb(self, data):

        try:
            # Convert ROS Image message to OpenCV format
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except CvBridgeError as e:
            self.get_logger().error(f"Depth image conversion failed: {e}")
            self.depth_image = None

    def colorimagecb(self, data):

        try:
            # Convert ROS Image to OpenCV RGB format
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding ='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Color image conversion failed: {e}")
            self.cv_image = None

    def bad_fruit_detection(self, bgr_image):
        bad_fruits = []
        
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        
        # STEP 1: Green mask for all fruits
        lower_green = np.array([45, 70, 170])
        upper_green = np.array([60, 100, 195])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # STEP 2: Grey/brown mask (LOW saturation = grey appearance)
        lower_grey = np.array([0, 0, 60])
        upper_grey = np.array([180, 45, 160])
        grey_mask = cv2.inRange(hsv, lower_grey, upper_grey)
        
        # Find contours in GREEN mask
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fruit_id = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10 or area > 5000:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            # STEP 3: Create ring around the fruit
            inner_mask = np.zeros(grey_mask.shape, dtype=np.uint8)
            cv2.drawContours(inner_mask, [cnt], -1, 255, -1)
                   
            # Create ring (doughnut shape) around fruit
            kernel = np.ones((20, 20), np.uint8)  # Adjust for ring thickness
            outer_mask = cv2.dilate(inner_mask, kernel, iterations=1)
            ring_mask = cv2.subtract(outer_mask, inner_mask)
            
            # STEP 4: Check for grey in the ring
            grey_in_ring = cv2.bitwise_and(grey_mask, ring_mask)
            grey_pixel_count = np.count_nonzero(grey_in_ring)
            
            ring_area = np.count_nonzero(ring_mask)
            grey_percentage = (grey_pixel_count / ring_area) * 100 if ring_area > 0 else 0
                        
            # STEP 5: Lower threshold for semi-ring detection
            # Even a small amount of grey on the edge = bad fruit
            GREY_THRESHOLD = 5.0  # Lower threshold (5-10%) to catch partial grey edges
            
            if grey_percentage < GREY_THRESHOLD:
                continue
                        
            # Calculate center
            cX = x + w // 2
            cY = y + h // 2
            
            # Get depth and angle
            distance = 0.0
            angle = 0.0
            if self.depth_image is not None:
                distance = float(self.depth_image[cY, cX])
                angle = np.degrees(np.arctan((cX - self.centerCamX) / self.focalX))
            
            fruit_info = {
                'center': (cX, cY),
                'distance': distance,
                'angle': angle,
                'width': w,
                'id': fruit_id,
                'grey_percentage': grey_percentage
            }
            bad_fruits.append(fruit_info)
            fruit_id += 1
        
        return bad_fruits
    def process_image(self):

        self.sizeCamX = 1280
        self.sizeCamY = 720
        self.centerCamX = 642.724365234375
        self.centerCamY = 361.9780578613281
        self.focalX = 915.3003540039062
        self.focalY = 914.0320434570312
            
        # Wait until both RGB and depth images are available
        if self.cv_image is None or self.depth_image is None:
            return
        bgr_image = self.cv_image.copy()

        # Step 1: Detect bad fruits
        center_fruit_list = self.bad_fruit_detection(bgr_image)

        # rgb_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)

        # Process each detected fruit
        for fruit in center_fruit_list:
            cX, cY = fruit['center']
            fruit_id = fruit['id']

            # Draw bounding box and label on the image for visualization
            w = fruit['width']
            x1, y1 = cX - w//2, cY - w//2
            x2, y2 = cX + w//2, cY + w//2
            cv2.rectangle(bgr_image, (x1, y1), (x2, y2), (0, 255, 0), 2)    # Green box
            cv2.putText(bgr_image, "bad_fruit", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)      # Red label


            # Step 2: Get depth at fruit center
            distance_from_rgb = 0.0
            if self.depth_image is not None:
                distance_from_rgb = float(self.depth_image[cY, cX])/1000.0  # in meters

            # Step 3: Convert 2D pixel coordinates to 3D depth coordinates
            x = distance_from_rgb * (self.sizeCamX - cX - self.centerCamX) / self.focalX
            y = distance_from_rgb * (self.sizeCamY - cY - self.centerCamY) / self.focalY
            z = distance_from_rgb
            # print(f"Fruit ID: {fruit_id}, X: {x:.3f} m, Y: {y:.3f} m, Z: {z:.3f} m")

            # Step 4: Draw center on image
            cv2.circle(bgr_image, (cX, cY), 5, (0, 255, 0), -1)

            # Step 5: Publish TF from camera_link to fruit frame
            t_cam = TransformStamped()
            t_cam.header.stamp = self.get_clock().now().to_msg()
            t_cam.header.frame_id = 'camera_link'
            t_cam.child_frame_id = f'cam_fruit_{fruit_id}'
            t_cam.transform.translation.x = z
            t_cam.transform.translation.y = x
            t_cam.transform.translation.z = y

            t_cam.transform.rotation.x = -0.660
            t_cam.transform.rotation.y =  0.660
            t_cam.transform.rotation.z = -0.253
            t_cam.transform.rotation.w =  0.253


            self.tf_broadcaster.sendTransform(t_cam)

            # # Step 6: Lookup transform from base_link to fruit frame
            try:
                now = rclpy.time.Time()
                trans = self.tf_buffer.lookup_transform(
                    'base_link',
                    t_cam.child_frame_id,
                    now
                )   
            except Exception as e:
                self.get_logger().warn(f"TF lookup failed for fruit {fruit_id}: {e}")
                continue

            # Step 7: Publish final TF from base_link to fruit object frame
            t_base = TransformStamped()
            t_base.header.stamp = self.get_clock().now().to_msg()
            t_base.header.frame_id = 'base_link'
            t_base.child_frame_id = f'{self.team_id}_bad_fruit_{fruit_id}'
            t_base.transform = trans.transform 
            self.tf_broadcaster.sendTransform(t_base)

        # Step 8: Show annotated image
        cv2.imshow('Detected Bad Fruits', bgr_image)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = FruitsTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down FruitsTF")
        node.destroy_node()
        rclpy.shutdown()
        if SHOW_IMAGE:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()