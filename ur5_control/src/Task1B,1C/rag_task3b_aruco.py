#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
*****************************************************************************************
*
*        		===============================================
*           		    Logistic coBot (LB) Theme (eYRC 2024-25)
*        		===============================================
*
*  This script should be used to implement Task 1B of Logistic coBot (LB) Theme (eYRC 2024-25).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:          [ 2788 ]
# Author List:		[ Raj Mohammad, Ragini Vinay Mehta, Priyanshu Kishore, Nitin Mane]
# Filename:		    task1b_Raj_Mohammad.py
# Functions:
#			        [ Comma separated list of functions in this file ]
# Nodes:		    Add your publishing and subscribing node
#			        Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/aligned_depth_to_color/image_raw, /etc... ]


################### IMPORT MODULES #######################

import rclpy
import sys
import cv2
import math
import tf2_ros
import tf_transformations
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import CompressedImage, Image
from example_interfaces.srv import SetBool


##################### FUNCTION DEFINITIONS #######################


################################################################

def calculate_rectangle_area(coordinates):

    area = None
    width = None

    (TL,TR,BR,BL)=coordinates
    TR=(int(TR[0]),int(TR[1]))
    BR=(int(BR[0]),int(BR[1]))
    TL=(int(TL[0]),int(TL[1]))
    BL=(int(BL[0]),int(BL[1]))
    
    len=math.sqrt((TR[0]-TL[0])**2 + (TR[1]-TL[1])**2)
    width=math.sqrt((TR[0]-BR[0])**2 + (TR[1]-BR[1])**2)
    if(len<width):
        width=len

    area=len*width
    
    return area, width




##################### CLASS DEFINITION #######################

class aruco_tf(Node):


    def __init__(self):


        super().__init__('aruco_tf_publisher')                                          # registering node

        ############ Topic SUBSCRIPTIONS ############

        self.color_cam_sub = self.create_subscription(Image, '/camera/image_raw', self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, '/camera/depth/image_raw', self.depthimagecb, 10)
        
        ############ Constructor VARIABLES/OBJECTS ############

        image_processing_rate = 0.5 #0.2                                                # rate of time to process image (seconds)
        self.bridge = CvBridge()                                                        # initialise CvBridge object for image conversion
        self.tf_buffer = tf2_ros.buffer.Buffer()                                        # buffer time used for listening transforms
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)                                    # object as transform broadcaster to send transform wrt some frame_id
        self.timer = self.create_timer(image_processing_rate, self.process_image)       # creating a timer based function which gets called on every 0.2 seconds (as defined by 'image_processing_rate' variable)
        
        self.cv_image = None                                                            # colour raw image variable (from colorimagecb())
        self.depth_image = None                                                         # depth image variable (from depthimagecb())
        
        self.cam_mat = np.array([[931.1829833984375, 0.0, 640.0], [0.0, 931.1829833984375, 360.0], [0.0, 0.0, 1.0]])
        self.dist_mat = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

        self.service = self.create_service(SetBool, 'detect_aruco', self.control_processing_callback)
        self.processing_enabled =True #False   #to control the processing of Aruco Markers

    
    def control_processing_callback(self,request,response):

        if request.data:
            self.processing_enabled = True
            response.message = "Aruco enabled."
        else:
            self.processing_enabled = False
            response.message = "Aruco disabled."
        response.success = True
        self.get_logger().info(response.message)
        return response

    def depthimagecb(self, data):
        try:
            # Convert ROS Image message to OpenCV depth image (float32)
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="32FC1")
        except CvBridgeError as e:
            self.get_logger().error(f"Depth image conversion failed: {str(e)}")

    def colorimagecb(self, data):

        try:
            bgr_color_img=self.bridge.imgmsg_to_cv2(data,"bgr8")
            rgb_color_img=cv2.cvtColor(bgr_color_img,cv2.COLOR_BGR2RGB)
            self.cv_image=rgb_color_img
        except CvBridgeError as e:
            print(e)

    #rotation matrix appied to marker axis (viewed from camera) to orient in robot axis.
    
    def rot_cam_frame(self,id):

        try:
            tf_msg = TransformStamped()
            tf_msg.header.stamp = self.get_clock().now().to_msg()
            tf_msg.header.frame_id ='2788_cam_'+str(id)
            tf_msg.child_frame_id ='2788_rot_tf_'+str(id)
            
            tf_msg.transform.translation.x =0.0
            tf_msg.transform.translation.y =0.0
            tf_msg.transform.translation.z =0.0
            tf_msg.transform.rotation.w = 0.7068252
            tf_msg.transform.rotation.x = 0.0
            tf_msg.transform.rotation.y = -0.7068252
            tf_msg.transform.rotation.z = 0.0
            self.br.sendTransform(tf_msg)
        except Exception as e:
                self.get_logger().info(f"Error in rot_cam transform: {str(e)}")

    #complete location and orientation of the Aruco marker with respect to the robot's primary frame of reference.
    def look_up_transform(self,id):
        try:
            transform = self.tf_buffer.lookup_transform('base_link', '2788_rot_tf_'+str(id[0]), rclpy.time.Time())
        
            tf_msg1= TransformStamped()
            tf_msg1.header.stamp = self.get_clock().now().to_msg()
            tf_msg1.header.frame_id = 'base_link'
            tf_msg1.child_frame_id = '2788_base_'+str(id[0])

            tf_msg1.transform.translation = transform.transform.translation
            tf_msg1.transform.rotation = transform.transform.rotation

            
            
            self.br.sendTransform(tf_msg1)

        except Exception as e:
            self.get_logger().info(f"Error looking up or publishing obj transform: {str(e)}")


    def publish_transform(self,x,y,z,qx,qy,qz,qw,name):
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self. get_clock().now().to_msg()
        tf_msg.header.frame_id = 'camera_link'
        tf_msg.child_frame_id = '2788_cam_'+str(name)
        tf_msg.transform.translation.x = z#z
        tf_msg.transform.translation.y = x#x
        tf_msg.transform.translation.z = y#y    
        tf_msg.transform.rotation.w = qw
        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz

        self.br.sendTransform(tf_msg)


    def process_image(self):

        ############ Function VARIABLES ############        
        sizeCamX = 1280
        sizeCamY = 720
        centerCamX = 640 
        centerCamY = 360
        focalX = 931.1829833984375
        focalY = 931.1829833984375

        if not self.processing_enabled:
            return

        try:
            center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, ids= self.detect_aruco(self.cv_image)

            for i in range (len(ids)):
                cX,cY=center_aruco_list[i][0],center_aruco_list[i][1]
                qx,qy,qz,qw = angle_aruco_list[i]
                id =ids[i]
                
                depth_value=self.depth_image[(int)(cY),(int)(cX)]
                c = float(depth_value) 
                # c=depth_value[0]
                # c=float(c/1000)

                x = c* (sizeCamX - cX - centerCamX) / focalX*1.0
                y = c * (sizeCamY - cY - centerCamY) / focalY*1.0
                z = c
             
                frame=self.cv_image

                cv2.waitKey(1)

                self.publish_transform(x,y,z,qx,qy,qz,qw,id[0])
                self.rot_cam_frame(id[0])
                self.look_up_transform(id)
        except Exception as e:
            print(e, "in process fxn")

    def detect_aruco(self,image):

        aruco_area_threshold = 1500
        size_of_aruco_m = 0.15

        center_aruco_list = []
        distance_from_rgb_list = []
        angle_aruco_list = []
        width_aruco_list = []
        ids = []

        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        gray_img=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) 

        aruco_params=cv2.aruco.DetectorParameters()
        
        try:
            if image.shape[0] > 0 and image.shape[1] > 0:
                (corners, id, rejected)=cv2.aruco.detectMarkers(gray_img, self.dictionary,parameters=aruco_params)
            
                for (markerCorner, markerID) in zip(corners,id):
                    corners_reshaped=markerCorner.reshape((4,2))
                    area,width=calculate_rectangle_area(corners_reshaped)

    
                    if(area>=aruco_area_threshold):
                        ids.append(markerID)
                        width_aruco_list.append(width)

                        (TL,TR,BR,BL)=corners_reshaped
                        TR=(int(TR[0]),int(TR[1]))
                        BR=(int(BR[0]),int(BR[1]))
                        TL=(int(TL[0]),int(TL[1]))
                        BL=(int(BL[0]),int(BL[1]))

                        cX=(TL[0]+BR[0])/2.0
                        cY=(TL[1]+BR[1])/2.0
                        center=(cX,cY)

                        center_aruco_list.append(center)
    
                for i in range (len(ids)):

                    corner=[corners[i]]

                    ret= cv2.aruco.estimatePoseSingleMarkers(corner ,size_of_aruco_m, cameraMatrix=self.cam_mat,distCoeffs= self.dist_mat)
                    rvec,tvec=ret[0][0,0,:],ret[1][0,0,:]

                    ######################################
                    rotation_matrix, _ = cv2.Rodrigues(np.array([rvec[1],rvec[0],rvec[2]]))
                    rotation = R.from_matrix(rotation_matrix)
                    euler_angles = -rotation.as_euler('xyz', degrees=False)
                    quat_angles= tf_transformations.quaternion_from_euler(-euler_angles[2],euler_angles[1],(euler_angles[0]))
                    ######################################
            
                    angle_aruco_list.append(quat_angles)

                    distance_from_rgb_list.append(np.linalg.norm(tvec))

                    if area>aruco_area_threshold:
                        cv2.aruco.drawDetectedMarkers(image, corners, id)
                        cv2.drawFrameAxes(image, self.cam_mat, self.dist_mat, rvec, tvec, length =1)
                        cv2.circle(image,(int(cX),int(cY)),4,(0,255,0),1)
                        cv2.putText(image,f"center",(int(cX),int(cY)),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2,cv2.LINE_AA)        

                # cv2.imshow("image",image)   #to show the image
            
            else:
                print("Invalid image dimensions")

            return center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, ids
        except Exception as e:
            print(e)

def main():
    '''
    Description:    Main function which creates a ROS node and spin around for the aruco_tf class to perform it's task
    '''

    rclpy.init(args=sys.argv)                                       # initialisation

    node = rclpy.create_node('aruco_tf_process')                    # creating ROS node

    node.get_logger().info('Node created: Aruco tf process')        # logging information

    aruco_tf_class = aruco_tf()                                     # creating a new object for class 'aruco_tf'

    rclpy.spin(aruco_tf_class)                                      # spining on the object to make it alive in ROS 2 DDS

    aruco_tf_class.destroy_node()                                   # destroy node after spin ends

    rclpy.shutdown()                                                # shutdown process


if __name__ == '__main__':
    main()
