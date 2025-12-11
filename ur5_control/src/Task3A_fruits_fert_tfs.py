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
# Filename:		    Task3A_fruits_fert_tfs.py
# Functions:
#			        [ depthimagecb, colorimagecb, bad_fruit_detection, process_image ]
# Nodes:		    Add your publishing and subscribing node
#			        Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/image_raw, /camera/depth/image_raw ]

import rclpy
import cv2
import numpy as np 
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError 
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener 
from scipy.spatial.transform import Rotation as R
import tf_transformations
import tf2_ros
import math

# runtime parameters
SHOW_IMAGE = True


def calculate_rectangle_area(coordinates):

    #Calculate the area and width of a rectangle defined by four corner coordinates.

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

    # Camera intrinsic matrix
    cam_mat = np.array([[915.3003540039062, 0.0,642.724365234375], [0.0, 914.0320434570312, 361.9780578613281], [0.0, 0.0, 1.0]])

    # Camera distortion coefficients (assuming no distortion)
    dist_mat = np.array([0.0,0.0,0.0,0.0,0.0])

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



class FruitsFertTF(Node):

    def __init__(self):
        
        """Initialize the FruitsTF node with subscribers, publishers, and TF components."""
        super().__init__('fruits_fert_tf')

        self.cb_group = ReentrantCallbackGroup()
        image_processing_rate = 0.2                                                     # rate of time to process image (seconds)
        self.bridge = CvBridge()                                                        # initialise CvBridge object for image conversion
        self.cv_image = None                                                            # colour raw image variable (from colorimagecb())
        self.depth_image = None    

        self.br = tf2_ros.TransformBroadcaster(self)                                    # object as transform broadcaster to send transform wrt some frame_id
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()                                               
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.team_id = "5076"

        self.sizeCamX = 1280
        self.sizeCamY = 720
        self.centerCamX = 642.724365234375
        self.centerCamY = 361.9780578613281
        self.focalX = 915.3003540039062
        self.focalY = 914.0320434570312

    
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.colorimagecb, 10, callback_group=self.cb_group)             
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depthimagecb, 10, callback_group=self.cb_group)   
        self.create_timer(image_processing_rate, self.bad_fruit_processing, callback_group=self.cb_group)
        self.create_timer(image_processing_rate, self.aruco_processing, callback_group=self.cb_group)

        self.get_logger().info("Task 3A node started.")

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
    

    def bad_fruit_processing(self):

        if self.cv_image is None or self.depth_image is None:
            return
        bgr_image = self.cv_image.copy()

        # Step 1: Detect bad fruits
        center_fruit_list = self.bad_fruit_detection(bgr_image)

        for fruit in center_fruit_list:
            cX, cY = fruit['center']
            fruit_id = fruit['id']

            w = fruit['width']
            x1, y1 = cX - w//2, cY - w//2
            x2, y2 = cX + w//2, cY + w//2
            cv2.rectangle(bgr_image, (x1, y1), (x2, y2), (0, 255, 0), 2)    # Green box
            cv2.putText(bgr_image, "bad_fruit", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)      # Red label


            distance_from_rgb = 0.0
            if self.depth_image is not None:
                distance_from_rgb = float(self.depth_image[cY, cX])/1000.0  # in meters

            x = distance_from_rgb * (self.sizeCamX - cX - self.centerCamX) / self.focalX
            y = distance_from_rgb * (self.sizeCamY - cY - self.centerCamY) / self.focalY
            z = distance_from_rgb

            cv2.circle(bgr_image, (cX, cY), 5, (0, 255, 0), -1)

            t_cam = TransformStamped()
            t_cam.header.stamp = self.get_clock().now().to_msg()
            t_cam.header.frame_id = 'camera_link'
            t_cam.child_frame_id = f'{self.team_id}_bad_fruit_cam_{fruit_id}'
            t_cam.transform.translation.x = z
            t_cam.transform.translation.y = x
            t_cam.transform.translation.z = y

            t_cam.transform.rotation.x = -0.660
            t_cam.transform.rotation.y =  0.660
            t_cam.transform.rotation.z = -0.253
            t_cam.transform.rotation.w =  0.253


            self.tf_broadcaster.sendTransform(t_cam)

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

            t_base = TransformStamped()
            t_base.header.stamp = self.get_clock().now().to_msg()
            t_base.header.frame_id = 'base_link'
            t_base.child_frame_id = f'{self.team_id}_bad_fruit_{fruit_id}'
            t_base.transform = trans.transform 
            self.tf_broadcaster.sendTransform(t_base)

        cv2.imshow('Detected Bad Fruits', bgr_image)
        cv2.waitKey(1)

    def aruco_processing(self):

        if self.cv_image is None or self.depth_image is None:
            return

        center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, ids = detect_aruco(self.cv_image)

        if len(ids) == 0:
            return

        for i in range(len(ids)):
            marker_id = int(ids[i][0])      # get ID value
            cX, cY = center_aruco_list[i]   # marker center in pixels

            quat_angles = angle_aruco_list[i]  # full quaternion from detection
            qx, qy, qz, qw = quat_angles

            # Apply camera frame rotation
            q_rot = [0.0, -0.7068252, 0.0, 0.7068252]
            r_quat = R.from_quat([qx, qy, qz, qw])
            r_rot = R.from_quat(q_rot)
            r_combined = r_quat * r_rot
            q_combined = r_combined.as_quat()

            # Get depth value at marker center (in millimeters)
            depth_mm = self.depth_image[int(cY), int(cX)]
            c = float(depth_mm)/1000.0  # Convert mm to meters

            # Convert 2D pixel to 3D depth coordinates in camera frame
            x = c * (self.sizeCamX - cX - self.centerCamX) / self.focalX
            y = c * (self.sizeCamY - cY - self.centerCamY) / self.focalY
            z = c

            cv2.circle(self.cv_image, (int(cX), int(cY)), radius=5, color=(0, 0, 255), thickness=-1)

            # Create transform from camera_link to marker
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'camera_link'
            t.child_frame_id = f'cam_{marker_id}'

            t.transform.translation.x = z
            t.transform.translation.y = x
            t.transform.translation.z = y

            t.transform.rotation.x = q_combined[0]
            t.transform.rotation.y = q_combined[1]
            t.transform.rotation.z = q_combined[2]
            t.transform.rotation.w = q_combined[3]
            self.br.sendTransform(t)

            try:
                obj_frame = f'{self.team_id}_cam_{marker_id}'                          # child frame in camera_link
                base_to_obj = self.tf_buffer.lookup_transform(
                    'base_link',                                        # target frame
                    obj_frame,                                          # source frame
                    rclpy.time.Time()
                )
            except Exception as e:
                self.get_logger().warn(f"TF lookup failed for {obj_frame}: {e}")

            try:
                t_obj = TransformStamped()
                t_obj.header.stamp = self.get_clock().now().to_msg()
                t_obj.header.frame_id = 'base_link'
                if marker_id == 3:
                    t_obj.child_frame_id = f'{self.team_id}_fertilizer_1'
                else : 
                    t_obj.child_frame_id = f'{self.team_id}_ebot_{marker_id}'

                t_obj.transform.translation = base_to_obj.transform.translation
                t_obj.transform.rotation = base_to_obj.transform.rotation
                self.br.sendTransform(t_obj)

                # pos = base_to_obj.transform.translation
                # quat = base_to_obj.transform.rotation

                # print(f"Marker: {marker_id}")
                # print(f'{{"pos": [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}], '
                #     f'"quat": [{quat.x:.3f}, {quat.y:.3f}, {quat.z:.3f}, {quat.w:.3f}]}}')

            except:
                print("Id not found")

def main(args=None):
    rclpy.init(args=args)
    node = FruitsFertTF()
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