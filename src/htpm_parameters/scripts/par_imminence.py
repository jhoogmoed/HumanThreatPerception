#!/usr/bin/env python


# Import main
import sys
import os

# Import select
import rospy
import numpy as np

# Import specific ros
from htpm_kitti_parser.msg import msg_kitti_object_list, msg_kitti_oxts
from htpm_parameters.msg import msg_par
from std_msgs.msg import Bool



class par_imminence:
    def __init__(self):
        # Initiation of ROS
        self.init_ros()
        self.init_subscribers()
        self.init_publishers()
        print ('htpm_parameters_imminence: Initiation complete')
        
    def init_ros(self):
        #ROS node start
        rospy.init_node('htpm_parameters_imminence', anonymous = False) #Anonymous true or false  
        
    def init_subscribers(self):
        self.pub_obj            = False
        self.pub_imu            = False
        self.param_sub_objects  = rospy.Subscriber('htpm/objects', msg_kitti_object_list, self.callback_objects)
        self.param_sub_imu      = rospy.Subscriber('htpm/imu', msg_kitti_oxts, self.callback_imu)
        self.master_pub         = rospy.Subscriber('htpm/master',Bool, self.callback_shutdown)

            
    def init_publishers(self):        	
        self.param_pub = rospy.Publisher('htpm/parameters/imminence', msg_par , queue_size = 2) 	
        
        
    def callback_imu(self,data):
        self.frame_imu  = data.frame
        self.ego_vel    = data.linear_velocity
        self.ego_acc    = data.linear_acceleration
        self.location   = data.location
        self.pub_imu    = True
        
        
    def callback_objects(self,data):
        self.frame_objects  = data.frame
        self.objects        = data.objects
        self.pub_obj        = True
        
    def callback_shutdown(self,message):
        rospy.signal_shutdown('exit')
     
    def spin(self):
        if self.pub_obj == True and self.pub_imu ==True\
            and self.frame_imu == self.frame_objects:
            # Set arbitrarily large initial smallest distance
            smallest_distance = 10000000000
            for i in range(0,len(self.objects)):     
                # Get smallest distance to object from list
                if self.objects[i].type != "DontCare":
                    location = self.objects[i].location
                    npLocation = np.array([location.x,location.y,location.z])
                    distance = np.linalg.norm(npLocation)
                    if distance < smallest_distance:
                        smallest_distance = distance
            # If no objects or really far, set to nan
            if smallest_distance == 10000000000:
                smallest_distance = float('nan')
            time_headway = 1 / (smallest_distance / self.ego_vel.x)
            # self.param_pub.publish(self.frame_imu,time_headway)     
            if time_headway > 5:
                imminence_par = 0
            elif time_headway > 4:
                imminence_par = 1
            elif time_headway > 3:
                imminence_par = 2
            elif time_headway > 2:
                imminence_par = 3
            elif time_headway > 1:
                imminence_par = 4
            elif time_headway > 0:
                imminence_par = 5
            else:
                imminence_par = 0
                
            self.param_pub.publish(self.frame_imu,imminence_par)     

            # print('Frame number: %s' %self.frame_imu)
            # print('Imminence Parameter %s' %time_headway)
            # print('speed was: %s' %self.ego_vel.x)
            self.pub_imu = False
            self.pub_obj = False


par_imminence =par_imminence()
while not rospy.is_shutdown():
    try:
        par_imminence.spin()
    except KeyboardInterrupt:
        pass
print('\rhtpm_parameters_imminence: Shutting down')
