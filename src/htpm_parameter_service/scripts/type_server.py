#!/usr/bin/env python


import rospy
import numpy as np

# Import specific ros
from std_msgs.msg import String

from htpm_parameter_service.srv import Stype
 
def typeSwitch(objType):
    # Switch to type to assign weight based on...
    typeSwitch = {
    'Car'   : .2,
    'Van'   : .4,
    'Truck' :.6,
    'Pedestrian': 1,
    'Person_sitting': .2,
    'Cyclist': 1,
    'Tram': .6,
    'Misc': .2,
    'DontCare': 0,
    }
    return typeSwitch.get(objType, "Invalid object type")

def get_type(data):
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
    return weightN

def type_server():
    rospy.init_node('htpm_type_server')
    server_type = rospy.Service('parameter/type', Stype, get_type)
    rospy.spin()

type_server()  

