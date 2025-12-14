#!/usr/bin/python3
# -*- coding: utf-8 -*-
 
# Team ID:          [ eYRC#5076 ]
# Author List:		[ Nagulapalli Shavaneeth Gourav, Pradeep J, Anand Vardhan, Raj Mohammad ]
# Filename:		    Task2B_aruco_perception.py
# Functions:
#			        [ calculate_rectangle_area, detect_aruco, depthimagecb, colorimagecb, process_image ]
# Nodes:		    Add your publishing and subscribing node
#			        Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/image_raw, /camera/depth/image_raw ]

################### IMPORT MODULES #######################
 
import rclpy
import sys
import cv2
import math
import tf2_ros
import numpy as np
import tf_transformations
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image

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

class aruco_tf(Node):

    def __init__(self):

        super().__init__('aruco_tf_publisher')                                          # registering node

        ############ Topic SUBSCRIPTIONS ############

        self.color_cam_sub = self.create_subscription(Image, '/camera/image_raw', self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, '/camera/depth/image_raw', self.depthimagecb, 10)

        ############ Constructor VARIABLES/OBJECTS ############

        image_processing_rate = 0.2                                                     # rate of time to process image (seconds)
        self.bridge = CvBridge()                                                        # initialise CvBridge object for image conversion
        self.tf_buffer = tf2_ros.buffer.Buffer()                                        # buffer time used for listening transforms
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)                                    # object as transform broadcaster to send transform wrt some frame_id
        self.timer = self.create_timer(image_processing_rate, self.process_image)       # creating a timer based function which gets called on every 0.2 seconds (as defined by 'image_processing_rate' variable)
        
        self.cv_image = None                                                            # colour raw image variable (from colorimagecb())
        self.depth_image = None                                                         # depth image variable (from depthimagecb())
        self.team_id = "5076"                                                           # Team ID

    def depthimagecb(self, data):
        try:
            # Convert ROS Image to OpenCV format (32-bit float, single channel)
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="32FC1")
        except CvBridgeError as e:
            self.get_logger().error(f"Depth image conversion failed: {str(e)}")

    def colorimagecb(self, data):
        try:
            # Convert ROS Image to OpenCV BGR format
            self.cv_image=self.bridge.imgmsg_to_cv2(data,"bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"Color image conversion failed: {str(e)}")

    def process_image(self):

        # Camera parameters for depth-to-3D conversion
        sizeCamX = 1280
        sizeCamY = 720
        centerCamX = 640 
        centerCamY = 360
        focalX = 931.1829833984375
        focalY = 931.1829833984375
 
        if self.cv_image is None or self.depth_image is None:
            return

        center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, ids = detect_aruco(self.cv_image)

        if len(ids) == 0:
            return

        for i in range(len(ids)):
            try:
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
                c = float(depth_mm)

                # Convert 2D pixel to 3D depth coordinates in camera frame
                x = c * (sizeCamX - cX - centerCamX) / focalX
                y = c * (sizeCamY - cY - centerCamY) / focalY
                z = c

                cv2.circle(self.cv_image, (int(cX), int(cY)), radius=5, color=(0, 0, 255), thickness=-1)

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
                self.br.sendTransform(t)
            except:
                print("Error in processing marker ID:", ids[i])
           
            try:
                obj_frame = f'cam_{marker_id}'                          # child frame in camera_link
                base_to_obj = self.tf_buffer.lookup_transform(
                    'base_link',                                        # target frame
                    obj_frame,                                          # source frame
                    rclpy.time.Time()
                )

                t_obj = TransformStamped()
                t_obj.header.stamp = self.get_clock().now().to_msg()
                t_obj.header.frame_id = 'base_link'
                if marker_id == 3:
                    t_obj.child_frame_id = f'{self.team_id}_fertiliser_can'
                else :
                    t_obj.child_frame_id = f'{self.team_id}_ebot_{marker_id}'

                # copy translation and rotation from lookup
                t_obj.transform.translation = base_to_obj.transform.translation
                t_obj.transform.rotation = base_to_obj.transform.rotation
                self.br.sendTransform(t_obj)

                pos = base_to_obj.transform.translation
                quat = base_to_obj.transform.rotation

                # print(f"Marker: {marker_id}")
                # print(f'{{"pos": [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}], '
                #     f'"quat": [{quat.x:.3f}, {quat.y:.3f}, {quat.z:.3f}, {quat.w:.3f}]}}')
            except Exception as e:
                    self.get_logger().warn(f"TF lookup failed for {obj_frame}: {e}")
                    continue
        # cv2.imshow("Detected ArUco Markers", self.cv_image)
        # cv2.waitKey(1) 

def main():

    rclpy.init(args=sys.argv)                                       # initialisation

    node = rclpy.create_node('aruco_tf_process')                    # creating ROS node

    node.get_logger().info('Node created: Aruco tf process')        # logging information

    aruco_tf_class = aruco_tf()                                     # creating a new object for class 'aruco_tf'

    rclpy.spin(aruco_tf_class)                                      # spining on the object to make it alive in ROS 2 DDS

    aruco_tf_class.destroy_node()                                   # destroy node after spin ends

    rclpy.shutdown()                                                # shutdown process


if __name__ == '__main__':
    main()