#!/usr/bin/env python

import os
import rospy
import numpy as np
# from tqdm import tqdm 
import math


# Import specific ros
from std_msgs.msg import String

from htpm_parameter_service.srv import Stype, Simm, Sprob
from htpm_parameter_service.msg import msg_kitti_object,  msg_kitti_oxts

from std_msgs.msg import Float32
from std_msgs.msg import Bool
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import Image



class kitti_parser():
    def __init__(self):
        # rospy.init_node('htpm_kitti_parser', anonymous = False)
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
        
        # Get information   
        self.get_road()
        self.get_objects()
        self.get_imu()

    def sorter(self, name):
        frame = int(name.split('.')[0])
        return frame
    
    def get_type(self,x):
        rospy.wait_for_service('parameter/type')
        get_type = rospy.ServiceProxy('parameter/type', Stype)
        try:
            resp = get_type(self.objects[self.frame],x)
            par_type = resp.par
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
        return par_type       
            
    def get_imminence(self,x):
        rospy.wait_for_service('parameter/imminence')
        get_imminence = rospy.ServiceProxy('parameter/imminence', Simm)
        try:
            resp = get_imminence(self.objects[self.frame],
                                 self.imus[self.frame].location,
                                 self.imus[self.frame].linear_velocity,
                                 self.imus[self.frame].linear_acceleration,x)
            par_imminence = resp.par
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
        return par_imminence
    
    def get_probability(self,x):
        rospy.wait_for_service('parameter/probability')
        get_prob = rospy.ServiceProxy('parameter/probability', Sprob)
        try:
            resp = get_prob(self.road_types[self.frame],x)
            par_probability = resp.par
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
        return par_probability
    
    def typeSwitch(self,objType,parameters):
        # Switch to type to assign weight based on...
        typeSwitch = {
        'Car'           : parameters[0],
        'Van'           : parameters[1],
        'Truck'         : parameters[2],
        'Pedestrian'    : parameters[3],
        'Person_sitting': parameters[4],
        'Cyclist'       : parameters[5],
        'Tram'          : parameters[6],
        'Misc'          : parameters[7],
        'DontCare'      : parameters[8],
        }
        return typeSwitch.get(objType, "Invalid object type")
    
    def roadSwitch(self,roadType,parameters):
        # Switch to type to assign weight based on...
        roadSwitch = {
        'city'          : parameters[9],
        'residential'   : parameters[10],
        'road'          : parameters[11],
        }
        return roadSwitch.get(roadType, "Invalid object type")
        
    def fast_type(self,x):
        par_type = []
        for frame_objects in self.objects:
            types= []
            for object in frame_objects:
                types.append(self.typeSwitch(object.type,x))
            par_type.append(sum(types))
        return par_type
    
    def fast_imm(self,x):
        a = x[12]
        b = x[13]
        par_imm = []
        for frame_objects in self.objects:
            imms = []
            for object in frame_objects:
                distance = math.sqrt(object.location.x*object.location.x + 
                                object.location.y*object.location.y + 
                                object.location.z*object.location.z)
                imm = a ** distance + b
                imms.append(imm)
            par_imm.append(sum(imms))
        return par_imm
    
    def fast_prob(self,x):
        probability_par = []
        for road in self.road_types:
            probability_par.append(self.roadSwitch(road,x))
            
        
                        
        return probability_par

    def parameter_server(self,x):
        # Open results file
        # self.csvFile    = open(os.path.join(self.results_folder, 'model_responses/model_results.csv'), 'a')
        # self.csvFile.write('Frame number, Combination parameter, Type parameter, Imminence parameter, Probability parameter\n')
        results = {'Frame number':[],
                   'Combination parameter':[],
                   'Type parameter':[],
                   'Imminence parameter':[],
                   'Probability parameter':[]}
        par_imminence = self.fast_imm(x)
        par_type      = self.fast_type(x)
        par_probability = self.fast_prob(x)
        
        
        # for self.frame in tqdm (range (len(self.left_color_image_list)), desc="Working on frames..."): 
        # for self.frame in range(len(self.left_color_image_list)): 
        #     # print('Working on frame:%s' %self.frame)  
            
        #     # Get parameter values from info
        #     par_type = self.get_type(x)
        #     par_imminence = self.get_imminence(x)
        #     par_probability = self.get_probability(x)
            
        #     # Combine parameters
        
        
        
        par_combi = []
        for i in range(len(par_imminence)):
            par_combi.append(par_imminence[i]+ par_type[i] + par_probability[i])
                    
        # Update dict
        results['Frame number'] = range(len(self.left_color_image_list))
        results['Combination parameter'] = par_combi
        results['Type parameter'] =par_type
        results['Imminence parameter'] = par_imminence
        results['Probability parameter'] = par_probability

  
            # Save to csv file
            # self.csvFile.write(str(self.frame) + ',' + str(par_combi) + ',' + str(par_type) + ',' + str(par_imminence) + ',' + str(par_probability) +'\n')
            
        # self.csvFile.close
        return results
        
    def get_objects(self):
        self.objects = []
        for i in range(len(self.objects_list)):
            # Open file
            self.object_file  = open(self.dataPath + self.drive +'/label_2/' + self.objects_list[i], "r")

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
            self.objects.append(self.objects_msg)

    def get_imu(self):
        self.imus = []
        for file in self.imu_list:
            # Open file 
            self.imu_file   = open(self.dataPath + self.drive +'/oxts/data/' + file, "r")
            
            # Create new imu msg
            imu_msg         = msg_kitti_oxts()
            imu_msg.frame   = self.frame
            line            = self.imu_file.readline() 
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
            
            # Close file
            self.imu_file.close    

            self.imus.append(imu_msg)
            
    def get_road(self):
        road_file  = open(self.dataPath + self.drive +'/uniform_image_list.txt', "r")
        lines = road_file.readlines()
        self.road_types = []
        for i in range(len(lines)):
            self.road_types.append(lines[i].split('/')[0])

if __name__ == "__main__":
    kp = kitti_parser()
    x = [ 0.44280227,  3.34705824,  5.76650168,  1.71872993, -1.05546119,
        4.27986996, -3.84714484, -0.67921923, 18.11550273,  4.97899823,
        4.1454334 ,  0.58620334, 19.83541916, 12.93964481, 13.93964481]
    kp.parameter_server(x)
