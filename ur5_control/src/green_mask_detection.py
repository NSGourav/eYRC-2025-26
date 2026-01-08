#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Green Mask Visualization Tool
Shows what your green mask is detecting in real-time
Adjust HSV trackbars to tune detection
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class GreenMaskVisualizer(Node):
    def __init__(self):
        super().__init__('green_mask_visualizer')
        self.bridge = CvBridge()
        self.cv_image = None
        
        # Subscribe to camera
        self.create_subscription(
            Image, 
            '/camera/camera/color/image_raw', 
            self.image_callback, 
            10
        )
        
        # Your current green parameters
        self.lower_h = 45
        self.lower_s = 70
        self.lower_v = 170
        self.upper_h = 60
        self.upper_s = 100
        self.upper_v = 195
        
        # Create window and trackbars
        cv2.namedWindow('Green Mask Controls')
        cv2.createTrackbar('Lower H', 'Green Mask Controls', self.lower_h, 180, self.nothing)
        cv2.createTrackbar('Lower S', 'Green Mask Controls', self.lower_s, 255, self.nothing)
        cv2.createTrackbar('Lower V', 'Green Mask Controls', self.lower_v, 255, self.nothing)
        cv2.createTrackbar('Upper H', 'Green Mask Controls', self.upper_h, 180, self.nothing)
        cv2.createTrackbar('Upper S', 'Green Mask Controls', self.upper_s, 255, self.nothing)
        cv2.createTrackbar('Upper V', 'Green Mask Controls', self.upper_v, 255, self.nothing)
        
        # Create timer for processing
        self.create_timer(0.1, self.process_image)
        
        self.get_logger().info("Green Mask Visualizer started")
        print("\n" + "="*60)
        print("Green Mask Visualizer")
        print("="*60)
        print("Adjust trackbars to tune green detection")
        print("Windows will show:")
        print("  1. Original Image")
        print("  2. Green Mask (white = detected green)")
        print("  3. Green Overlay (green areas highlighted)")
        print("  4. Contours (detected green regions with bounding boxes)")
        print("\nPress 'q' to quit")
        print("Press 'p' to print current HSV values")
        print("="*60 + "\n")
    
    def nothing(self, x):
        pass
    
    def image_callback(self, msg):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Image conversion failed: {e}")
    
    def process_image(self):
        if self.cv_image is None:
            return
        
        # Get current trackbar values
        self.lower_h = cv2.getTrackbarPos('Lower H', 'Green Mask Controls')
        self.lower_s = cv2.getTrackbarPos('Lower S', 'Green Mask Controls')
        self.lower_v = cv2.getTrackbarPos('Lower V', 'Green Mask Controls')
        self.upper_h = cv2.getTrackbarPos('Upper H', 'Green Mask Controls')
        self.upper_s = cv2.getTrackbarPos('Upper S', 'Green Mask Controls')
        self.upper_v = cv2.getTrackbarPos('Upper V', 'Green Mask Controls')
        
        # Convert to HSV
        hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
        
        # Create green mask
        lower_green = np.array([self.lower_h, self.lower_s, self.lower_v])
        upper_green = np.array([self.upper_h, self.upper_s, self.upper_v])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Create overlay image
        green_overlay = self.cv_image.copy()
        green_overlay[green_mask > 0] = [0, 255, 0]  # Highlight green areas
        
        # Blend overlay with original
        blended = cv2.addWeighted(self.cv_image, 0.6, green_overlay, 0.4, 0)
        
        # Find contours
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = self.cv_image.copy()
        
        valid_contours = 0
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 10 or area > 5000:  # Your area filters
                continue
            
            valid_contours += 1
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Draw contour and bounding box
            cv2.drawContours(contour_image, [cnt], -1, (0, 255, 0), 2)
            cv2.rectangle(contour_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Calculate center
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(contour_image, (cX, cY), 5, (0, 0, 255), -1)
                cv2.putText(contour_image, f"#{i+1} A:{int(area)}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Add info text
        info_text = f"Detected: {valid_contours} green regions | HSV: [{self.lower_h},{self.lower_s},{self.lower_v}] to [{self.upper_h},{self.upper_s},{self.upper_v}]"
        cv2.putText(contour_image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show all windows
        cv2.imshow('1. Original Image', self.cv_image)
        cv2.imshow('2. Green Mask', green_mask)
        cv2.imshow('3. Green Overlay', blended)
        cv2.imshow('4. Detected Contours', contour_image)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            print("\n" + "="*60)
            print("Current HSV Parameters:")
            print(f"lower_green = np.array([{self.lower_h}, {self.lower_s}, {self.lower_v}])")
            print(f"upper_green = np.array([{self.upper_h}, {self.upper_s}, {self.upper_v}])")
            print(f"Detected {valid_contours} valid green regions")
            print("="*60 + "\n")

def main(args=None):
    rclpy.init(args=args)
    node = GreenMaskVisualizer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down Green Mask Visualizer")
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()