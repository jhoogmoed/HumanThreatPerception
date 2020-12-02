#!/usr/bin/env python

# Import main
import sys
import os

# import select
import roslib
import rospy
import cv2
import time
import numpy as np
import tracklet_parser

# Import specific other
from cv_bridge import CvBridge, CvBridgeError


# Import specific ros
from std_msgs.msg import String
from htpm_parameters.msg import msg_par, msg_road

# from std_msgs.msg import Header
# from std_msgs.msg import Float32
from std_msgs.msg import Bool
# from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import Image

from htpm_kitti_parser.msg import msg_kitti_object, msg_kitti_object_list, msg_kitti_oxts


class image_parser:
    def __init__(self):
        # Set base paths
        self.dataPath               = '/home/jim/HDDocuments/university/master/thesis/ROS/data/2011_09_26'
        # self.drive                  = '/city/2011_09_26_drive_0093_sync'
        self.drive                  = '/test_images'
        self.results_folder         = '/home/jim/HDDocuments/university/master/thesis/results'

        # Check if exists
        if(not os.path.exists(self.dataPath+self.drive)):
            print("Drive does not exist")
            raise SystemExit 
        
        # Image paths
        try:
            self.left_color_image_list  = sorted(os.listdir(self.dataPath + self.drive + '/image_02/data'),key=self.sorter)
        except:
            print("No image data")
            raise SystemExit 

        # Imu paths
        try:
            self.imu_list               = sorted(os.listdir(self.dataPath + self.drive + '/oxts/data/'),key=self.sorter)
        except:
            print("No oxts data")
            raise SystemExit 
            
        # Object paths
        try:
            self.objects_list           = sorted(os.listdir(self.dataPath + self.drive + '/label_2'),key=self.sorter)
        except:
            print("No object data, create from xml...")
            try:
                tracklet_parser.main(self.dataPath, self.drive)
                self.objects_list           = sorted(os.listdir(self.dataPath + self.drive + '/label_2'),key=self.sorter)
            except:
                print("No object xml")
                raise SystemExit 
        
        # Check variables
        self.frame = 0
        self.done = 0
        
        # Setup data acquisition
        try:
            os.remove(os.path.join(self.results_folder,'model_responses/model_results.csv'))
        except:         
            pass   
        self.csvFile = open(os.path.join(self.results_folder,'model_responses/model_results.csv'),'a')
        self.csvFile.write('Frame number,Combination parameter,Type parameter,Imminence parameter,Probability parameter\n')
         
        #Initiation of functions
        self.cv_bridge = CvBridge()
        
        #Initiation of pubs
        self.left_color_image       = None
        self.left_color_image_ros   = None
        
        # Initiation of ROS
        self.init_ros()
        self.init_subscribers()
        self.init_publishers()
        
        # Time for setup
        print ('htpm_kitti_parser: Initiation complete')
        print ('\r')
        time.sleep(1)
            
            
    def init_publishers(self):
        # Camera publishers
        self.left_color_image_pub = rospy.Publisher('htpm/left_color', Image, queue_size = 1) 		
        	
        # Object publisher
        self.objects_pub = rospy.Publisher('htpm/objects', msg_kitti_object_list , queue_size = 1) 	      
        
        # IMU publisher
        self.imu_pub = rospy.Publisher('htpm/imu', msg_kitti_oxts, queue_size = 1)	
        
        # Road type publisher
        self.road_pub = rospy.Publisher('htpm/road', msg_road, queue_size=1)
        
        # Master switch
        self.master_pub = rospy.Publisher('htpm/master',Bool, queue_size = 1)
        
        
    def init_subscribers(self):
        self.combi_frame    = -1
        self.combi_sub      = rospy.Subscriber('htpm/parameters/combi', msg_par, self.callback_par_combi)	
        self.type_sub       = rospy.Subscriber('htpm/parameters/type', msg_par, self.callback_par_type)
        self.imminence_sub  = rospy.Subscriber('htpm/parameters/imminence', msg_par, self.callback_par_imminence)
        self.probability_sub  = rospy.Subscriber('htpm/parameters/probability', msg_par, self.callback_par_probability)

        
    def init_ros(self):
        #ROS node start
        rospy.init_node('htpm_kitti_parser', anonymous = False) #Anonymous true or false  
        
    
    def sorter(self,name):
        frame = int(name.split('.')[0])
        return frame
    
        
    def callback_par_combi(self,combi):
        self.combi_frame    = combi.frame
        self.combi_par      = combi.par
        
        # Save to csv file
        self.csvFile.write(str(self.combi_frame) + ',' + str(self.combi_par) + ',' + str(self.type_par) + ',' + str(self.imminence_par) + ',' + str(self.probability_par) +'\n')
        self.parse()
        
        
    def callback_par_type(self,type):
        self.type_par      = type.par
        
        
    def callback_par_imminence(self,imm):
        self.imminence_par = imm.par
        
        
    def callback_par_probability(self,prob):
        self.probability_par = prob.par
        
            
    def parse(self):     
        # Set endpoint
        if (self.frame == len(self.left_color_image_list)): # if (self.frame == 5): # If less than max
            self.done = 1
            self.csvFile.close()
            self.master_pub.publish(True)
            print('\rhtpm_kitti_parser: Finished parsing %s images' %self.frame)
            rospy.signal_shutdown('\rhtpm_kitti_parser: Finished parsing %s images' %self.frame)
            raise SystemExit
        
        # Terminal info    
        print('Working on frame %s' % self.frame)
        
        # Load images
        self.left_color_image = cv2.imread(self.dataPath + self.drive + '/image_02/data/' + self.left_color_image_list[self.frame])
        
        # Load Imu
        imu_file        = open(self.dataPath + self.drive +'/oxts/data/' + self.imu_list[self.frame], "r")
        imu_msg         = msg_kitti_oxts()
        imu_msg.frame   = self.frame
        line            = imu_file.readline() 
        imuArgs         = line.split(' ')
        imu_msg.location = Quaternion(
            float(imuArgs[0]), 
            float(imuArgs[1]), 
            float(imuArgs[2]), 
            float(imuArgs[5]))
        imu_msg.linear_velocity = Vector3(
            float(imuArgs[8]), 
            float(imuArgs[9]), 
            float(imuArgs[10]))
        imu_msg.linear_acceleration = Vector3(
            float(imuArgs[11]), 
            float(imuArgs[12]), 
            float(imuArgs[13]))
        imu_file.close    
        
        # Load objects
        object_file  = open(self.dataPath + self.drive +'/label_2/' + self.objects_list[self.frame], "r")

        # Empty object list 
        objects_msg = []
        
        # Read next line
        lines = object_file.readlines()      
        for object in lines:
            # Create new object from data
            newObj      = msg_kitti_object()
            oArgs       = object.split(' ')   
            newObj.type       = oArgs[0]
            newObj.truncated   = float(oArgs[1])
            newObj.occluded    = int(oArgs[2])
            newObj.alpha       = float(oArgs[3])
            newObj.bbox        = Quaternion(float(oArgs[4]), 
                                    float(oArgs[5]), 
                                    float(oArgs[6]), 
                                    float(oArgs[7]))
            newObj.dimensions  = Vector3(float(oArgs[8]), 
                                float(oArgs[9]), 
                                float(oArgs[10]))
            newObj.location    = Vector3(float(oArgs[11]), 
                                float(oArgs[12]), 
                                float(oArgs[13]))
            newObj.location_y  = float(oArgs[14])
            
            # Append object list of frame 
            objects_msg.append(newObj)
                        
        # Close file
        object_file.close
        
        
        # # Draw rectangles on objects
        # for i in range(0, len(objects_msg)):
        #     if objects_msg[i].type != "DontCare":
        #             self.left_color_image = cv2.rectangle(self.left_color_image, 
        #                                                 (int(objects_msg[i].bbox.x), int(objects_msg[i].bbox.y)), 
        #                                                 (int(objects_msg[i].bbox.z), int(objects_msg[i].bbox.w)), 
        #                                                 self.colors[self.objectTypes.index(objects_msg[i].type)], 3)
        
        # # Change image to ROS message type
        # self.left_color_image_ros = self.cv_bridge.cv2_to_imgmsg(self.left_color_image, "bgr8")
        
        road_file  = open(self.dataPath + self.drive +'/uniform_image_list.txt', "r")
        line = road_file.readlines()[self.frame]
        road_type = line.split('/')[0]
        
        # Publish to ROS
        try:
            # self.left_color_image_pub.publish(self.left_color_image_ros)
            self.objects_pub.publish(self.frame, objects_msg)
            self.imu_pub.publish(imu_msg)
            self.road_pub.publish(self.frame, road_type)
        except CvBridgeError as e:
            print(e)
        
        # Loop through frames
        self.frame = self.frame+1   
        
             
    
# Main spin
if __name__ == "__main__":
    image_parser = image_parser()
    image_parser.parse()
    while(image_parser.done == 0 and not(rospy.is_shutdown())):        
        try:
            rospy.spin() 
        except KeyboardInterrupt:
            pass
            print('\rhtpm_kitti_parser: Shutting down by interrupt')


    

