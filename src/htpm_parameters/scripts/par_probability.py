#!/usr/bin/env python


# Import main
import sys
import os

# Import select
import rospy
import numpy as np

# Import specific ros
from htpm_kitti_parser.msg import msg_road
from htpm_parameters.msg import msg_par
from std_msgs.msg import Bool



class par_probability:
    def __init__(self):
        # Initiation of ROS
        self.init_ros()
        self.init_subscribers()
        self.init_publishers()
        print ('htpm_parameters_probability: Initiation complete')
     
        
    def init_ros(self):
        #ROS node start
        rospy.init_node('htpm_parameters_probability', anonymous = False) #Anonymous true or false  
        
        
    def init_subscribers(self):
        self.param_sub = rospy.Subscriber('htpm/road', msg_road, self.callback)
        self.master_pub = rospy.Subscriber('htpm/master',Bool, self.callback_shutdown)
          
            
    def init_publishers(self):        	
        self.param_pub = rospy.Publisher('htpm/parameters/probability', msg_par , queue_size = 1) 	
        
        
    def callback(self,data):
        road_type = data.road
        if road_type == 'city':
            probability_par = 3
        elif road_type == 'residential':     
            probability_par = 1.5
        elif road_type == 'road':
            probability_par = 0
        else:
            probability_par = 0
            print('Unknown roadtype')
        self.param_pub.publish(data.frame,probability_par)
    
    
    def spin(self):
        rospy.spin()
     
        
    def callback_shutdown(self,message):
        rospy.signal_shutdown('exit')

par_probability =par_probability()
while not rospy.is_shutdown():
    try:
        par_probability.spin()
    except KeyboardInterrupt:
        pass
    print('\rhtpm_parameters_probability: Shutting down')


    