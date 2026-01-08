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
# Filename:		    Task4A_perception.py
# Functions:
#			        [ calculate_rectangle_area, detect_aruco, depthimagecb, colorimagecb, bad_fruit_detection, process_image ]
# Nodes:		    Add your publishing and subscribing node
#			        Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/image_raw, /camera/depth/image_raw ]

import rclpy
import sys
import cv2
import math
import tf2_ros
import numpy as np
import tf_transformations
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError 
from geometry_msgs.msg import TransformStamped, PointStamped
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


    cam_mat = np.array([[915.3003540039062, 0.0,642.724365234375], [0.0, 914.0320434570312, 361.9780578613281], [0.0, 0.0, 1.0]]) # Camera intrinsic matrix
    dist_mat = np.array([0.0,0.0,0.0,0.0,0.0])  # Camera distortion coefficients (assuming no distortion)

    size_of_aruco_m = 0.13
    center_aruco_list = []
    distance_from_rgb_list = []
    angle_aruco_list = []
    width_aruco_list = []
    ids = []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load ArUco dictionary (4x4 markers with 50 unique IDs)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()

    # Create detector and find markers in the image
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return [], [], [], [], []

    # Draw detected markers on the image for visualization
    cv2.aruco.drawDetectedMarkers(image, corners, ids)

    for i in range(len(ids)):
        c = corners[i][0]
        area, width = calculate_rectangle_area(c)
        width_aruco_list.append(width)
        if area < aruco_area_threshold:
            continue

        # Estimate pose (position and orientation) of the marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, size_of_aruco_m, cam_mat, dist_mat
        )

        # Calculate center point of the marker by averaging corner coordinates
        center = np.mean(c, axis=0)
        center_aruco_list.append(center)

        distance = np.linalg.norm(tvecs[i][0])
        distance_from_rgb_list.append(distance)

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

    return center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, ids

class Fert_Bad_Fruit_Detector(Node):

    def __init__(self):
        
        super().__init__('combined_perception')
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None
        self.cb_group = ReentrantCallbackGroup()

        # Subscriptions
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.colorimagecb, 10, callback_group=self.cb_group)             
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depthimagecb, 10, callback_group=self.cb_group)       

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
        self.max_missing_frames = 5      # Number of consecutive frames a fruit can be missing before it's removed from tracking
        self.smoothing_alpha = 0.3       # Exponential smoothing factor for position filtering (0.2-0.4 recommended, lower = smoother but slower response)

        # Fruit colour thresholding
        self.lower_green = np.array([45, 55, 155])          # HSV lower bound for green color detection 
        self.upper_green = np.array([65, 95, 230])          # HSV upper bound for green color detection 

        self.lower_cyan = np.array([155, 75, 75])           # HSV lower bound for cyan/turquoise color detection 
        self.upper_cyan = np.array([170, 160, 225])         # HSV upper bound for cyan/turquoise color detection 

        self.lower_grey = np.array([0, 0, 45])              # HSV lower bound for grey color detection 
        self.upper_grey = np.array([180, 75, 160])          # HSV upper bound for grey color detection 

        self.CYAN_THRESHOLD = 15.0                          # Amount of cyan pixels present around the fruit
        self.GREY_THRESHOLD = 20.0                          # Amount of grey pixels present around the fruit

        self.min_pixel_area_fruit = 200                     # Minimum pixel area of the fruit in the image
        self.max_pixel_area_fruit = 2000                    # Maximum pixel area of the fruit in the image

        self.get_logger().info("Combined Perception node started.")

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
        green_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        cyan_mask = cv2.inRange(hsv, self.lower_cyan, self.upper_cyan)
        grey_mask = cv2.inRange(hsv, self.lower_grey, self.upper_grey)

        # Find contours in GREEN mask (all fruits have green center)
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fruit_id = 0
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

            if distance <= 1000.0 or distance > 2000.0:  # Skip if distance between camera and bad fruit is not between 1000mm to 2000mm
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

        # Camera parameters for depth-to-3D conversion
        self.sizeCamX = 1280
        self.sizeCamY = 720
        self.centerCamX = 642.724365234375
        self.centerCamY = 361.9780578613281
        self.focalX = 915.3003540039062
        self.focalY = 914.0320434570312

        # self.get_logger().info("Inside process image timer")
            
        # Wait until both RGB and depth images are available
        if self.cv_image is None or self.depth_image is None:
            return
        
        bgr_image = self.cv_image.copy()

        # ============= ARUCO MARKER DETECTION =============
        center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, ids = detect_aruco(bgr_image)

        if len(ids) > 0:
            for i in range(len(ids)):
                try:
                    marker_id = int(ids[i][0])      # get ID value
                    cX, cY = center_aruco_list[i]   # marker center in pixels

                    quat_angles = angle_aruco_list[i]  # full quaternion from detection
                    qx, qy, qz, qw = quat_angles

                    # Apply camera frame rotation
                    q_rot = [0.0, -0.7068252, 0.0, 0.7068252]
                    q_rot_2=[0.0,0.0,0.0,1.0]
                    r_quat = R.from_quat([qx, qy, qz, qw])
                    r_rot = R.from_quat(q_rot)
                    r_rot_2=R.from_quat(q_rot_2)
                    r_combined = r_quat * r_rot * r_rot_2
                    q_combined = r_combined.as_quat()

                    # Get depth value at marker center (in millimeters)
                    depth_mm = self.depth_image[int(cY), int(cX)]
                    c = float(depth_mm)/1000.0  # Convert mm to meters

                    # Convert 2D pixel to 3D depth coordinates in camera frame
                    x = c * (self.sizeCamX - cX - self.centerCamX) / self.focalX
                    y = c * (self.sizeCamY - cY - self.centerCamY) / self.focalY
                    z = c

                    cv2.circle(bgr_image, (int(cX), int(cY)), radius=5, color=(0, 0, 255), thickness=-1)

                    # Create transform from camera_link to marker
                    t = TransformStamped()
                    t.header.stamp = self.get_clock().now().to_msg()
                    t.header.frame_id = 'camera_link'
                    t.child_frame_id = f'cam_{marker_id}'

                    # translation
                    t.transform.translation.x = z
                    t.transform.translation.y = x
                    t.transform.translation.z = y

                    # rotation quaternion
                    t.transform.rotation.x = q_combined[0]
                    t.transform.rotation.y = q_combined[1]
                    t.transform.rotation.z = q_combined[2]
                    t.transform.rotation.w = q_combined[3]

                    # Publish transform to camera_link
                    self.tf_broadcaster.sendTransform(t)

                    # This converts from camera coordinates to robot base coordinates
                    try:
                        obj_frame = f'cam_{marker_id}'                          # child frame in camera_link
                        base_to_obj = self.tf_buffer.lookup_transform(
                            'base_link',                                        # target frame
                            obj_frame,                                          # source frame
                            rclpy.time.Time()
                        )
                    except Exception as e:
                        self.get_logger().warn(f"TF lookup failed for {obj_frame}: {e}")
                        continue
                except:
                    print("Error in processing marker ID:", ids[i])

                try:
                    t_obj = TransformStamped()
                    t_obj.header.stamp = self.get_clock().now().to_msg()
                    t_obj.header.frame_id = 'base_link'
                    if marker_id == 3:
                        t_obj.child_frame_id = f'{self.team_id}_fertilizer_1'
                    else :
                        #pass
                        t_obj.child_frame_id = f'{self.team_id}_ebot_{marker_id}'

                    # copy translation and rotation from lookup
                    t_obj.transform.translation = base_to_obj.transform.translation
                    t_obj.transform.rotation = base_to_obj.transform.rotation
                    self.tf_broadcaster.sendTransform(t_obj)

                    # pos = base_to_obj.transform.translation
                    # quat = base_to_obj.transform.rotation
                    # print(f"Marker: {marker_id}")
                    # print(f'{{"pos": [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}], '
                    #     f'"quat": [{quat.x:.3f}, {quat.y:.3f}, {quat.z:.3f}, {quat.w:.3f}]}}')
                except:
                    print("Id not found")

        # ============= BAD FRUIT DETECTION =============
        # Step 1: Detect bad fruits
        center_fruit_list = self.bad_fruit_detection(bgr_image)

        # Step 2: Track fruits and assign stable IDs with smoothing
        current_frame_fruits = set()

        for fruit in center_fruit_list:
            cX, cY = fruit['center']
            
            # Get depth at fruit center
            distance_from_rgb = 0.0
            if self.depth_image is not None:
                distance_from_rgb = float(self.depth_image[cY, cX])/1000.0

            # Convert 2D to 3D coordinates
            x_3d = distance_from_rgb * (self.sizeCamX - cX - self.centerCamX) / self.focalX
            y_3d = distance_from_rgb * (self.sizeCamY - cY - self.centerCamY) / self.focalY
            z_3d = distance_from_rgb
            
            # Try to match with existing tracked fruit
            matched_id = None
            min_distance = float('inf')
            
            for tracked_id, data in self.tracked_fruits.items():
                tx, ty = data['pos']
                dist = np.sqrt((cX - tx)**2 + (cY - ty)**2)
                if dist < self.position_threshold and dist < min_distance:
                    min_distance = dist
                    matched_id = tracked_id
            
            if matched_id is None:
                # New fruit - initialize
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

        # Remove fruits not seen
        fruits_to_remove = []
        for tracked_id, data in self.tracked_fruits.items():
            if tracked_id not in current_frame_fruits:
                new_frame_count = data['frame_count'] + 1
                if new_frame_count > self.max_missing_frames:
                    fruits_to_remove.append(tracked_id)
                else:
                    data['frame_count'] = new_frame_count

        for tracked_id in fruits_to_remove:
            del self.tracked_fruits[tracked_id]

        # Step 3: Publish TFs using SMOOTHED positions
        for fruit in center_fruit_list:
            cX, cY = fruit['center']
            fruit_id = fruit['stable_id']  # CHANGED: use stable_id
            
            # Get SMOOTHED position
            x, y, z = self.tracked_fruits[fruit_id]['smooth_pos']  # CHANGED: use smoothed

            # Draw bounding box and label
            w = fruit['width']
            x1, y1 = cX - w//2, cY - w//2
            x2, y2 = cX + w//2, cY + w//2
            cv2.rectangle(bgr_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(bgr_image, f"bad_fruit_{fruit_id}", (x1, y1 - 10),  # CHANGED: show ID
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(bgr_image, (cX, cY), 5, (0, 255, 0), -1)

            # Step 5: Publish TF (using smoothed x, y, z)
            t_cam = TransformStamped()
            t_cam.header.stamp = self.get_clock().now().to_msg()
            t_cam.header.frame_id = 'camera_link'
            t_cam.child_frame_id = f'cam_{fruit_id}'
            t_cam.transform.translation.x = z  # Already smoothed
            t_cam.transform.translation.y = x  # Already smoothed
            t_cam.transform.translation.z = y  # Already smoothed

            t_cam.transform.rotation.x = -0.660
            t_cam.transform.rotation.y =  0.660
            t_cam.transform.rotation.z = -0.253
            t_cam.transform.rotation.w =  0.253

            self.tf_broadcaster.sendTransform(t_cam)

            # Step 6: Lookup transform
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

            # Step 7: Publish final TF
            t_base = TransformStamped()
            t_base.header.stamp = self.get_clock().now().to_msg()
            t_base.header.frame_id = 'base_link'
            t_base.child_frame_id = f'{self.team_id}_bad_fruit_{fruit_id}'
            t_base.transform = trans.transform 
            self.tf_broadcaster.sendTransform(t_base)
        
        # Step 8: Show annotated image
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