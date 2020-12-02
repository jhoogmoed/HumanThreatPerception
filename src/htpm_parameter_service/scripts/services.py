# !/usr/bin/env python

import os
import rospy
import numpy as np

# Import specific ros
from std_msgs.msg import String

from htpm_parameter_service.srv import Stype
from htpm_parameter_service.msg import msg_kitti_object, msg_kitti_object_list, msg_kitti_oxts

from std_msgs.msg import Float32
from std_msgs.msg import Bool
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import Image



class kitti_parser():
    def __init__(self):
        rospy.init_node('htpm_kitti_parser', anonymous = False)
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
            self.left_color_image_list  = sorted(os.listdir(self.dataPath + self.drive + '/image_02/data'), key = self.sorter)
        except:
            print("No image data")
            raise SystemExit 

        # Imu paths
        try:
            self.imu_list               = sorted(os.listdir(self.dataPath + self.drive + '/oxts/data/'), key = self.sorter)
        except:
            print("No oxts data")
            raise SystemExit 
            
        # Object paths
        try:
            self.objects_list           = sorted(os.listdir(self.dataPath + self.drive + '/label_2'), key = self.sorter)
        except:
            print("No object data, create from xml...")
            try:
                tracklet_parser.main(self.dataPath, self.drive)
                self.objects_list           = sorted(os.listdir(self.dataPath + self.drive + '/label_2'), key = self.sorter)
            except:
                print("No object xml")
                raise SystemExit 
        
        # Check variables
        self.frame = 0
        self.done = 0
        
        # Setup data acquisition
        try:
            os.remove(os.path.join(self.results_folder, 'model_responses/model_results.csv'))
        except:         
            pass   
        
    def sorter(self, name):
        frame = int(name.split('.')[0])
        return frame
    
    def get_type(self):
        rospy.wait_for_service('parameter/type')
        get_type = rospy.ServiceProxy('parameter/type', Stype)
        try:
            resp = get_type(self.objects_msg)
            self.par_type = resp.par
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
        
    def parameter_server(self):
        # Open results file
        self.csvFile    = open(os.path.join(self.results_folder, 'model_responses/model_results.csv'), 'a')
        self.csvFile.write('Frame number, Combination parameter, Type parameter, Imminence parameter, Probability parameter\n')
        
        while (self.frame != len(self.left_color_image_list)): # if (self.frame == 5): # If less than max
            print('Working on frame:%s' %self.frame)  
            # Get information 
            self.get_objects()
            self.get_imu()
            
            # Get parameter values from info
            self.get_type()
            # self.get_imminence()
            # self.get_probability()
            
            # Combine parameters
            par_combi = self.par_type + self.par_imminence + self.par_probability
            
            # Loop through frames
            self.frame = self.frame+1  
            
        self.csvFile.close
        
    def get_objects(self):
        # Open file
        self.object_file  = open(self.dataPath + self.drive +'/label_2/' + self.objects_list[self.frame], "r")

        # Empty object list 
        self.objects_msg = []
        
        # Read next line
        lines = self.object_file.readlines()      
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
            self.objects_msg.append(newObj)
            
        # Close file
        self.object_file.close

    def get_imu(self):
        # Open file 
        self.imu_file   = open(self.dataPath + self.drive +'/oxts/data/' + self.imu_list[self.frame], "r")
        
        # Create new imu msg
        self.imu_msg         = msg_kitti_oxts()
        self.imu_msg.frame   = self.frame
        line            = self.imu_file.readline() 
        imuArgs         = line.split(' ')
        self.imu_msg.location = Quaternion(
            float(imuArgs[0]), 
            float(imuArgs[1]), 
            float(imuArgs[2]), 
            float(imuArgs[5]))
        self.imu_msg.linear_velocity = Vector3(
            float(imuArgs[8]), 
            float(imuArgs[9]), 
            float(imuArgs[10]))
        self.imu_msg.linear_acceleration = Vector3(
            float(imuArgs[11]), 
            float(imuArgs[12]), 
            float(imuArgs[13]))
        
        # Close file
        self.imu_file.close    


if __name__ == "__main__":
    kp = kitti_parser()
    kp.parameter_server()
