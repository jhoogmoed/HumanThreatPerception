#!/usr/bin/env python


import rospy
import numpy as np

# Import specific ros
from std_msgs.msg import String

from htpm_parameter_service.srv import Stype
 
def typeSwitch(objType,parameters):
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

def get_type(data):
    print(data)
    objects = data.objects
    parameters = data.x
    weight = []
    n = len(objects)
    for i in range(0,n):
        # print objects[i].type
        # print typeSwitch(objects[i].type)
        
        weight.append(typeSwitch(objects[i].type,parameters))
    
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

