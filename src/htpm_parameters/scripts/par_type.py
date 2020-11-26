#!/usr/bin/env python


# Import main
import sys
import os

# Import select
import rospy
import numpy as np

# Import specific ros
from htpm_kitti_parser.msg import msg_kitti_object_list
from htpm_parameters.msg import msg_par
from std_msgs.msg import Bool


def typeSwitch(objType):
    # Switch to type to assign weight based on...
    typeSwitch = {
    'Car'   : .4,
    'Van'   : .4,
    'Truck' :.6,
    'Pedestrian': 1,
    'Person_sitting': .2,
    'Cyclist': 1,
    'Tram': .8,
    'Misc': .2,
    'DontCare': 0,
    }
    return typeSwitch.get(objType, "Invalid object type")

class par_type: 
    def __init__(self):
        # Initiation of ROS
        self.init_ros()
        self.init_subscribers()
        self.init_publishers()
        print ('htpm_parameters_type: Initiation complete')

    
    def init_ros(self):
        #ROS node start
        rospy.init_node('htpm_parameters_type', anonymous = False) #Anonymous true or false  
        
        
    def init_subscribers(self):
        self.param_sub = rospy.Subscriber('htpm/objects', msg_kitti_object_list, self.callback)
        self.master_pub = rospy.Subscriber('htpm/master',Bool, self.callback_shutdown)

            
    def init_publishers(self):        	
        self.param_pub = rospy.Publisher('htpm/parameters/type', msg_par , queue_size = 2) 	
        
        
    def callback(self,data):
        # print('Frame number: %s' %data.frame)
        
        objects = data.objects
        weight = []
        n = len(objects)
        for i in range(0,n):
            # print objects[i].type
            # print typeSwitch(objects[i].type)
            
            weight.append(typeSwitch(objects[i].type))
        
        if weight == []:
            weightN = 0
        else:
            weightN = np.sum(weight)
            
        # print('Type Parameter: %s' %weightN)  
            
        self.param_pub.publish(data.frame,weightN)  
    
    
    def callback_shutdown(self,message):
        rospy.signal_shutdown('exit')
        

    def spin(self):
        rospy.spin()

par_type = par_type()
while not rospy.is_shutdown():
    try:
        par_type.spin()
    except KeyboardInterrupt:
        pass
    print('\rhtpm_parameters_type: Shutting down')
    