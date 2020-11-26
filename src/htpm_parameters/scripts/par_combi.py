#!/usr/bin/env python


# Import main
import sys
import os

# Import select
import rospy
import numpy as np

# Import specific ROS
from htpm_parameters.msg import msg_par
from std_msgs.msg import Bool


class par_combi: 
    def __init__(self):
        # Initiation of ROS
        self.init_ros()
        self.init_subscribers()
        self.init_publishers()
        print ('htpm_parameters_combi: Initiation complete')
    
        
    def init_ros(self):
        #ROS node start
        rospy.init_node('htpm_parameters_combi', anonymous = False) #Anonymous true or false  
    
        
    def init_subscribers(self):
        self.type_sub = rospy.Subscriber('htpm/parameters/type', msg_par, self.callback_par_type)
        self.imm_sub = rospy.Subscriber('htpm/parameters/imminence', msg_par, self.callback_par_imminence)
        self.prob_sub = rospy.Subscriber('htpm/parameters/probability',msg_par, self.callback_par_probability)
        self.master_pub = rospy.Subscriber('htpm/master',Bool, self.callback_shutdown)

            
    def init_publishers(self):        	
        self.combi_pub = rospy.Publisher('htpm/parameters/combi', msg_par , queue_size = 1) 	
        
    
    def callback_par_probability(self,par_type):
        self.probability = par_type
        if np.isnan(self.probability.par):
            self.probability.par = 0
        self.push()
              

    def callback_par_type(self,par_type):
        self.type = par_type
        if np.isnan(self.type.par):
            self.type.par = 0
        self.push()

        
    def callback_par_imminence(self,par_imminence):
        self.imminence = par_imminence
        if np.isnan(self.imminence.par):
            self.imminence.par = 0
        self.push()


    def callback_shutdown(self,message):
        rospy.signal_shutdown('exit')


    def spin(self):
        rospy.spin()
        
    
    def push(self):
        try:
            if((self.type.frame == self.imminence.frame) and (self.type.frame == self.probability.frame)):
                combi = self.type.par+self.imminence.par
                # if combi>100:
                #     combi = 100
                # elif combi<0:
                #     combi = 0
                # else:
                #     pass
                self.combi_pub.publish(self.type.frame,combi) 
        except:
            pass
            # print('Frame number: %s' %self.type.frame)
            # print('Combi parameter: %s' %combi)

        
    

        
par_combi = par_combi()
while not rospy.is_shutdown():
    try:
        par_combi.spin()
    except KeyboardInterrupt:
        pass
    print('\rhtpm_parameters_combi: Shutting down')

