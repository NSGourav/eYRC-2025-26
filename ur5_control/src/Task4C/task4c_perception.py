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
# Filename:		    Task4C_perception.py
# Functions:
#			        [ calculate_rectangle_area, detect_aruco, depthimagecb, colorimagecb, bad_fruit_detection, process_image ]
# Nodes:		    Add your publishing and subscribing node
#			        Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/image_raw, /camera/depth/image_raw ]

import rclpy
import cv2
import math
import numpy as np
import tf_transformations
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError 
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener 
from scipy.spatial.transform import Rotation as R

# runtime parameters
SHOW_IMAGE = True

def calculate_rectangle_area(coordinates):

    area = None
    width = None
    (TL,TR,BR,BL)=coordinates
    TR=(int(TR[0]),int(TR[1]))
    BR=(int(BR[0]),int(BR[1]))
    TL=(int(TL[0]),int(TL[1]))
    BL=(int(BL[0]),int(BL[1]))

    length=math.sqrt((TR[0]-TL[0])**2 + (TR[1]-TL[1])**2)
    width=math.sqrt((TR[0]-BR[0])**2 + (TR[1]-BR[1])**2)

    if(length<width):
        length,width=width,length
    area=length*width

    return area, width

def detect_aruco(image):

    aruco_area_threshold = 1500
    cam_mat = np.array([[931.1829833984375, 0.0, 640.0], 
                        [0.0, 931.1829833984375, 360.0], 
                        [0.0, 0.0, 1.0]])
    dist_mat = np.array([0.0,0.0,0.0,0.0,0.0])  # Camera distortion coefficients (assuming no distortion)

    size_of_aruco_m = 0.13
    center_aruco_list = []
    angle_aruco_list = []
    ids = []
    ids_filtered = []  # ← New filtered list

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load ArUco dictionary (4x4 markers with 50 unique IDs)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()

    # Create detector and find markers in the image
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return [], [], []

    # Draw detected markers on the image for visualization
    cv2.aruco.drawDetectedMarkers(image, corners, ids)

    for i in range(len(ids)):
        c = corners[i][0]
        area, width = calculate_rectangle_area(c)
        if area < aruco_area_threshold:
            continue

        # Estimate pose (position and orientation) of the marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, size_of_aruco_m, cam_mat, dist_mat
        )

        # Calculate center point of the marker by averaging corner coordinates
        center = np.mean(c, axis=0)
        ids_filtered.append(ids[i])  # ← Add this
        center_aruco_list.append(center)

        # Convert rotation vector to rotation matrix, applying axis reordering
        rotation_matrix, _ = cv2.Rodrigues(np.array([rvecs[i][0][1], rvecs[i][0][0], rvecs[i][0][2]]))

        # Convert rotation matrix to Euler angles
        rotation = R.from_matrix(rotation_matrix)
        euler_angles = -rotation.as_euler('xyz', degrees=False)

        # Convert Euler angles to quaternion representation (for ROS transforms)
        quat_angles = tf_transformations.quaternion_from_euler(-euler_angles[2], euler_angles[1], euler_angles[0])
        angle_aruco_list.append(quat_angles)

        # Draw coordinate axes on the marker for visualization
        cv2.drawFrameAxes(image, cam_mat, dist_mat, rvecs[i], tvecs[i], 0.05)

    return center_aruco_list, angle_aruco_list, ids_filtered


class Fert_Bad_Fruit_Detector(Node):

    def __init__(self):
        
        super().__init__('combined_perception')
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None
        self.cb_group = ReentrantCallbackGroup()

        # Subscriptions     
        self.create_subscription(Image, '/camera/image_raw', self.colorimagecb, 10, callback_group=self.cb_group)             
        self.create_subscription(Image, '/camera/depth/image_raw', self.depthimagecb, 10, callback_group=self.cb_group)    

        # Timer for periodic processing
        image_processing_rate = 0.2                         # rate of time to process image (seconds)
        self.timer = self.create_timer(image_processing_rate, self.process_image, callback_group=self.cb_group)

        self.tf_buffer = Buffer()                                               
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.team_id = "5076"

        # Parameters for stable TF's
        self.tracked_fruits = {}         # Dictionary storing all currently tracked bad fruits with their smoothed positions and tracking state
        self.next_stable_id = 0          # Counter for assigning unique IDs to newly detected bad fruits (increments with each new detection)
        self.position_threshold = 40     # Maximum pixel distance to consider a detection as the same fruit across frames
        self.max_missing_frames = 100    # Number of consecutive frames a fruit can be missing before it's removed from tracking
        self.smoothing_alpha = 0.3       # Exponential smoothing factor for position filtering (0.2-0.4 recommended, lower = smoother but slower response)

        # Fruit colour thresholding
        self.lower_green = np.array([40, 240, 160])          # HSV lower bound for green color detection 
        self.upper_green = np.array([48, 255, 180])          # HSV upper bound for green color detection 

        self.lower_cyan = np.array([163, 235, 50])           # HSV lower bound for cyan/turquoise color detection 
        self.upper_cyan = np.array([167, 255, 140])          # HSV upper bound for cyan/turquoise color detection 

        self.lower_grey = np.array([10, 20, 70])             # HSV lower bound for grey color detection 
        self.upper_grey = np.array([20, 35, 175])  

        self.CYAN_THRESHOLD = 15.0                          # Amount of cyan pixels present around the fruit
        self.GREY_THRESHOLD = 20.0                          # Amount of grey pixels present around the fruit

        self.min_pixel_area_fruit = 200                     # Minimum pixel area of the fruit in the image
        self.max_pixel_area_fruit = 2000                    # Maximum pixel area of the fruit in the image

        self.get_logger().info("Combined Perception node started.")

    # ---------------- Callbacks ----------------
    def depthimagecb(self, data):
        try:
            # Convert ROS Image message to OpenCV format
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="32FC1")
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
        green_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        cyan_mask = cv2.inRange(hsv, self.lower_cyan, self.upper_cyan)
        grey_mask = cv2.inRange(hsv, self.lower_grey, self.upper_grey)

        # Find contours in GREEN mask (all fruits have green center)
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fruit_id = 1
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_pixel_area_fruit or area > self.max_pixel_area_fruit :
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            
            # STEP 4: Create mask for green circle
            green_circle_mask = np.zeros(grey_mask.shape, dtype=np.uint8)
            cv2.drawContours(green_circle_mask, [cnt], -1, 255, -1)
            
            # STEP 5: Create ring around the green circle
            kernel = np.ones((25, 25), np.uint8)  # Ring thickness
            outer_mask = cv2.dilate(green_circle_mask, kernel, iterations=1)
            ring_mask = cv2.subtract(outer_mask, green_circle_mask)
            
            # STEP 6: Percentage of grey or cyan inside the ring
            grey_in_ring = cv2.bitwise_and(grey_mask, ring_mask)
            cyan_in_ring = cv2.bitwise_and(cyan_mask, ring_mask)

            grey_pixels = np.count_nonzero(grey_in_ring)
            cyan_pixels = np.count_nonzero(cyan_in_ring)
            ring_area = np.count_nonzero(ring_mask)
            
            grey_percentage = (grey_pixels / ring_area) * 100
            cyan_percentage = (cyan_pixels / ring_area) * 100
            
            # print(f" Fruits :- Grey: {grey_percentage:.1f}%, Cyan: {cyan_percentage:.1f}%")
            
            if cyan_percentage > self.CYAN_THRESHOLD:
                continue  # Good fruit, skip
            
            if grey_percentage < self.GREY_THRESHOLD:
                continue  # Not enough grey
            
            if cyan_percentage > grey_percentage:
                continue  # Cyan is more dominant, probably good fruit
                        
            # Calculate center using moments (more accurate)
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                # Fallback to bounding box center
                cX = x + w // 2
                cY = y + h // 2
            else:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            
            # Get depth and angle
            distance = 0.0
            angle = 0.0
            if self.depth_image is not None:
                distance = float(self.depth_image[cY, cX])
                angle = np.degrees(np.arctan((cX - self.centerCamX) / self.focalX))

            if distance <= 1.0 or distance > 2.0:  # Skip if distance between camera and bad fruit is not between 1000mm to 2000mm
                continue
            
            # print(f"Detected Bad Fruit at position ({cX}, {cY}) - Distance: {distance:.3f} m, Grey: {grey_percentage:.1f}%, Cyan: {cyan_percentage:.1f}%")

            fruit_info = {
                'center': (cX, cY),
                'distance': distance,
                'angle': angle,
                'width': w,
                'id': fruit_id,
                'grey_percentage': grey_percentage,
                'cyan_percentage': cyan_percentage
            }
            bad_fruits.append(fruit_info)
            fruit_id += 1
        
        return bad_fruits

    def process_image(self):
        
        if self.cv_image is None or self.depth_image is None:
            return
        
        def convert_2d_to_3d(cX, cY, depth_image):
            depth_mm = depth_image[int(cY), int(cX)]
            distance_m = float(depth_mm)
            x_3d = distance_m * (self.sizeCamX - cX - self.centerCamX) / self.focalX
            y_3d = distance_m * (self.sizeCamY - cY - self.centerCamY) / self.focalY
            z_3d = distance_m
            return x_3d, y_3d, z_3d
        
        def lookup_transform_safe(target_frame, source_frame, entity_type, entity_id):
            try:
                return self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            except Exception as e:
                self.get_logger().warn(f"TF lookup failed for {entity_type} {entity_id}: {e}")
                return None
        
        def publish_transform_chain(x, y, z, entity_id, entity_type, quaternion=None):

            # Set default quaternion for fruits
            if quaternion is None:
                quaternion = [-0.660, 0.660, -0.253, 0.253]
            
            # Determine frame names based on entity type
            if entity_type == 'marker':
                cam_frame = f'cam_{entity_id}'
                final_frame = None
                if entity_id == 3:
                    final_frame = f'{self.team_id}_fertiliser_can'
                elif entity_id == 6:
                    final_frame = f'{self.team_id}_ebot_{entity_id}'
                else:
                    return False
            elif entity_type == 'fruit':
                cam_frame = f'cam_fruit_{entity_id}'
                final_frame = f'{self.team_id}_bad_fruit_{entity_id}'
            else:
                return False
            
            # Publish camera_link -> cam frame
            t_cam = TransformStamped()
            t_cam.header.stamp = self.get_clock().now().to_msg()
            t_cam.header.frame_id = 'camera_link'
            t_cam.child_frame_id = cam_frame
            t_cam.transform.translation.x = z
            t_cam.transform.translation.y = x
            t_cam.transform.translation.z = y
            t_cam.transform.rotation.x = quaternion[0]
            t_cam.transform.rotation.y = quaternion[1]
            t_cam.transform.rotation.z = quaternion[2]
            t_cam.transform.rotation.w = quaternion[3]
            self.tf_broadcaster.sendTransform(t_cam)
            
            # Lookup and publish base_link -> final frame
            trans = lookup_transform_safe('base_link', cam_frame, entity_type, entity_id)
            if trans is None:
                return False
            
            t_base = TransformStamped()
            t_base.header.stamp = self.get_clock().now().to_msg()
            t_base.header.frame_id = 'base_link'
            t_base.child_frame_id = final_frame
            t_base.transform = trans.transform
            self.tf_broadcaster.sendTransform(t_base)
            return True
        
        self.sizeCamX = 1280
        self.sizeCamY = 720
        self.centerCamX = 642.724365234375
        self.centerCamY = 361.9780578613281
        self.focalX = 915.3003540039062
        self.focalY = 914.0320434570312
        
        bgr_image = self.cv_image.copy()

        # =============== ARUCO MARKER DETECTION ===============
        center_aruco_list, angle_aruco_list, ids = detect_aruco(bgr_image)

        if len(ids) > 0:
            for i in range(len(ids)):
                cX, cY = center_aruco_list[i]
                marker_id = int(ids[i][0])
                qx, qy, qz, qw = angle_aruco_list[i]

                # Apply camera-to-robot rotation
                camera_to_robot_quat = [0.0, -0.7068252, 0.0, 0.7068252]
                marker_orientation_quat = R.from_quat([qx, qy, qz, qw])
                camera_rotation = R.from_quat(camera_to_robot_quat)
                marker_in_robot_frame = marker_orientation_quat * camera_rotation
                final_quaternion = marker_in_robot_frame.as_quat()
                
                x, y, z = convert_2d_to_3d(cX, cY, self.depth_image)
                
                cv2.circle(bgr_image, (int(cX), int(cY)), 5, (0, 0, 255), -1)

                publish_transform_chain(x, y, z, marker_id, 'marker', quaternion=final_quaternion)

        # =============== BAD FRUIT DETECTION ===============
        center_fruit_list = self.bad_fruit_detection(bgr_image)
        current_frame_fruits = set()

        for fruit in center_fruit_list:
            cX, cY = fruit['center']
            
            x_3d, y_3d, z_3d = convert_2d_to_3d(cX, cY, self.depth_image)
            
            # Match with existing tracked fruit
            matched_id = None
            min_distance = float('inf')
            
            for tracked_id, data in self.tracked_fruits.items():
                tx, ty = data['pos']
                dist = np.sqrt((cX - tx)**2 + (cY - ty)**2)
                if dist < self.position_threshold and dist < min_distance:
                    min_distance = dist
                    matched_id = tracked_id
            
            if matched_id is None:
                # New fruit
                matched_id = self.next_stable_id
                self.next_stable_id += 1
                self.tracked_fruits[matched_id] = {
                    'pos': (cX, cY),
                    'smooth_pos': (x_3d, y_3d, z_3d),
                    'frame_count': 0
                }
                self.get_logger().info(f"New bad fruit detected with ID: {matched_id}")
            else:
                # Apply smoothing
                old_x, old_y, old_z = self.tracked_fruits[matched_id]['smooth_pos']
                smooth_x = self.smoothing_alpha * x_3d + (1 - self.smoothing_alpha) * old_x
                smooth_y = self.smoothing_alpha * y_3d + (1 - self.smoothing_alpha) * old_y
                smooth_z = self.smoothing_alpha * z_3d + (1 - self.smoothing_alpha) * old_z
                
                self.tracked_fruits[matched_id] = {
                    'pos': (cX, cY),
                    'smooth_pos': (smooth_x, smooth_y, smooth_z),
                    'frame_count': 0
                }
            
            current_frame_fruits.add(matched_id)
            fruit['stable_id'] = matched_id

        # Increment frame_count for unseen fruits
        for tracked_id in list(self.tracked_fruits.keys()):
            if tracked_id not in current_frame_fruits:
                self.tracked_fruits[tracked_id]['frame_count'] += 1

        # Remove fruits that exceeded max missing frames
        fruits_to_remove = [
            tracked_id for tracked_id, data in self.tracked_fruits.items()
            if data['frame_count'] > self.max_missing_frames
        ]

        for tracked_id in fruits_to_remove:
            del self.tracked_fruits[tracked_id]

        # Publish fruit transforms and draw visualizations
        for fruit in center_fruit_list:
            cX, cY = fruit['center']
            fruit_id = fruit['stable_id']
            
            # Get smoothed 3D position
            x, y, z = self.tracked_fruits[fruit_id]['smooth_pos']

            # Draw visualization
            w = fruit['width']
            cv2.rectangle(bgr_image, (cX - w//2, cY - w//2), (cX + w//2, cY + w//2), (0, 255, 0), 2)
            cv2.putText(bgr_image, f"bad_fruit_{fruit_id}", (cX - w//2, cY - w//2 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(bgr_image, (cX, cY), 5, (0, 255, 0), -1)

            # Publish transforms
            publish_transform_chain(x, y, z, fruit_id, 'fruit')
        
        # Display result
        cv2.imshow('Combined Detection', bgr_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = Fert_Bad_Fruit_Detector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down Combined Perception Node")
        node.destroy_node()
        rclpy.shutdown()
        if SHOW_IMAGE:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()