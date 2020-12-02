#!/usr/bin/env python


import rospy
import numpy as np

# Import specific ros
from std_msgs.msg import String

from htpm_parameter_service.srv import Simm
 

def get_imminence(data):
    objects = data.objects
    ego_vel    = data.linear_velocity
    ego_acc    = data.linear_acceleration
    location   = data.location
    
    # Set arbitrarily large initial smallest distance
    smallest_distance = 10000000000
    for i in range(0,len(objects)):     
        # Get smallest distance to object from list
        if objects[i].type != "DontCare":
            location = objects[i].location
            npLocation = np.array([location.x,location.y,location.z])
            distance = np.linalg.norm(npLocation)
            if distance < smallest_distance:
                smallest_distance = distance
    # If no objects or really far, set to nan
    if smallest_distance == 10000000000:
        smallest_distance = float('nan')
    time_headway = 1 / (smallest_distance / ego_vel.x)
    # param_pub.publish(frame_imu,time_headway)     
    if time_headway > 5:
        imminence_par = 0
    elif time_headway > 4:
        imminence_par = 0.1
    elif time_headway > 3:
        imminence_par = 0.5
    elif time_headway > 2:
        imminence_par = 1
    elif time_headway > 1:
        imminence_par = 2
    elif time_headway > 0:
        imminence_par = 3
    else:
        imminence_par = 0
    
    return imminence_par
            

def type_server():
    rospy.init_node('htpm_imminence_server')
    server_type = rospy.Service('parameter/imminence', Simm, get_imminence)
    rospy.spin()

type_server()  

