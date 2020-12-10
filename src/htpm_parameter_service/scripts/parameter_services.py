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

    def sorter(self,name):
        frame = int(name.split('.')[0])
        return frame
    
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
        par_total_distance      = []
        par_closest_distance    = []
        par_velocity            = []
        par_imm                 = []
              
        # Get object and ego vehicle data per frame   
        for frame in range(len(self.objects)):
            # Get ego velocity
            velocity = math.sqrt(self.imus[frame].linear_velocity.x **2 +
                                 self.imus[frame].linear_velocity.y **2 +
                                 self.imus[frame].linear_velocity.z **2) 
            
            # Save velocity parameter
            par_velocity.append(velocity)  
            
            # Get object data per object in frame
            imms = []
            smallest_distance = 10000
            smallest_velocity = 10000
            for object in self.objects[frame]:
                
                distance = math.sqrt(object.location.x ** 2+ 
                                object.location.y ** 2 + 
                                object.location.z ** 2)
                if distance<smallest_distance:
                    smallest_distance = distance
                    smallest_velocity = velocity
                
            # Calculate time headway per vehicle
            thw = smallest_distance / smallest_velocity
        
            #Linear imminence parameter
            # imm =  a * thw +b
            
            # Quadratic imminence parameter
            # if thw>100:
            #     thw = 100
            if thw<0:
                thw = 0
            imm = a*thw**(1/b)
            # imm = math.exp(-(thw*a)) + b
            imms.append(imm)
            par_imm.append(sum(imms))
        return par_imm
    
    def fast_prob(self,x):
        probability_par = []
        for road in self.road_types:
            probability_par.append(self.roadSwitch(road,x))
          
        return probability_par
    
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
            # imu_msg.frame   = self.frame
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

    def get_model(self,x):        
        results = {'Frame number':[],
                   'Combination parameter':[],
                   'Type parameter':[],
                   'Imminence parameter':[],
                   'Probability parameter':[]}
        
        # Get individual model results
        par_imminence   = self.fast_imm(x)
        par_type        = self.fast_type(x)
        par_probability = self.fast_prob(x)     
        
        # Get combined model results
        par_combi = []
        for i in range(len(par_imminence)):
            par_combi.append(par_imminence[i]+ par_type[i] + par_probability[i])
                    
        # Update dict
        results['Frame number']             = range(len(self.left_color_image_list))
        results['Combination parameter']    = par_combi
        results['Type parameter']           = par_type
        results['Imminence parameter']      = par_imminence
        results['Probability parameter']    = par_probability
  
        return results
    
    def save_model(self,x):  
        # Open results file
        csvFile    = open(os.path.join(self.results_folder, 'model_responses/model_results.csv'), 'a')
        csvFile.write('Frame,Combi,Type,Imminence,Probabiltiy\n')
        
        # Get individual model results
        par_imminence   = self.fast_imm(x)
        par_type        = self.fast_type(x)
        par_probability = self.fast_prob(x)     
        
        
        # Get combined model results
        par_combi = []
        for frame in range(len(par_imminence)):
            par_combi.append(par_imminence[frame]+ par_type[frame] + par_probability[frame])
            
            # Save to results file
            csvFile.write(str(frame) + ',' + str(par_combi[frame]) + ',' + str(par_type[frame]) + ',' + str(par_imminence[frame]) + ',' + str(par_probability[frame]) +'\n')
        
        # Close results file
        csvFile.close


if __name__ == "__main__":
    kp = kitti_parser()
    # x = [ 0.78603735,  0.89275405,  0.97970909,  0.68928311,  0.64975938,
    #      0.83689267,  0.62785914,  0.74671845,  1.        ,  1.06649842,
    #      1.03987374,  0.89361434,  0.3830448 , -0.79077126]
    
    x = [0.97249244, -0.73035007,  3.78233866,  1.71937009, 2.80132008,
         0.13719511, -2.1247334, 0.11580047,  1.4249498,
         0.12294739,  2.06632313, -0.75077797,
         -0.19353087,  0.15254572]
    kp.save_model(x)
