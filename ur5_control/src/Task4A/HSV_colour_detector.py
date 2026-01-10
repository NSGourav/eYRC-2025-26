#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
HSV Color Picker Tool
Click on any pixel in the image to see its HSV and BGR values
Press 'q' to quit
"""

import cv2
import numpy as np

# Global variables
hsv_image = None
bgr_image = None

def mouse_callback(event, x, y, flags, param):
    """Mouse callback function to display HSV values when clicked"""
    global hsv_image, bgr_image
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if hsv_image is not None and bgr_image is not None:
            # Get HSV and BGR values at clicked position
            hsv_value = hsv_image[y, x]
            bgr_value = bgr_image[y, x]
            
            # print("\n" + "="*60)
            # print(f"Pixel Position: ({x}, {y})")
            print(f"HSV Value: H={hsv_value[0]:3d}, S={hsv_value[1]:3d}, V={hsv_value[2]:3d}")
            # print(f"BGR Value: B={bgr_value[0]:3d}, G={bgr_value[1]:3d}, R={bgr_value[2]:3d}")
            # print("="*60)
            
            # Draw a circle at clicked position for visual feedback
            temp_img = bgr_image.copy()
            cv2.circle(temp_img, (x, y), 10, (0, 255, 255), 2)
            cv2.putText(temp_img, f"H:{hsv_value[0]} S:{hsv_value[1]} V:{hsv_value[2]}", 
                       (x + 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.imshow('HSV Color Picker', temp_img)

def main():
    global hsv_image, bgr_image
    
    # OPTION 1: Read from image file
    image_path = input("Enter image file path (or press Enter to use webcam): ").strip()
    
    if image_path:
        # Read image from file
        bgr_image = cv2.imread(image_path)
        if bgr_image is None:
            print(f"Error: Could not read image from {image_path}")
            return
        
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        
        # Create window and set mouse callback
        cv2.namedWindow('HSV Color Picker')
        cv2.setMouseCallback('HSV Color Picker', mouse_callback)
        
        print("\nHSV Color Picker Tool Started")
        print("Click on any pixel to see its HSV and BGR values")
        print("Press 'q' to quit")
        
        cv2.imshow('HSV Color Picker', bgr_image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    else:
        # OPTION 2: Use webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Create window and set mouse callback
        cv2.namedWindow('HSV Color Picker')
        cv2.setMouseCallback('HSV Color Picker', mouse_callback)
        
        print("\nHSV Color Picker Tool Started (Webcam Mode)")
        print("Click on any pixel to see its HSV and BGR values")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            bgr_image = frame
            hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
            
            cv2.imshow('HSV Color Picker', bgr_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()