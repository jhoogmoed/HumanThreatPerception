#!/usr/bin/env python
import rospy
import numpy as np

# Import specific ros
from std_msgs.msg import String

from htpm_parameter_service.srv import Sprob
 
def get_probability(data):
    road_type = data.road
    parameters = data.x
    if road_type == 'city':
        probability_par = parameters[9]
    elif road_type == 'residential':     
        probability_par = parameters[10]
    elif road_type == 'road':
        probability_par = parameters[11]
    else:
        probability_par = 0
        print('Unknown roadtype')
    return probability_par

def type_server():
    rospy.init_node('htpm_probability_server')
    server_type = rospy.Service('parameter/probability', Sprob, get_probability)
    rospy.spin()

type_server()  

