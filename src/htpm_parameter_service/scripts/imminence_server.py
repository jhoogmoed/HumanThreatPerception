#!/usr/bin/env python


import rospy
import numpy as np

# Import specific ros
# from std_msgs.msg import String

from htpm_parameter_service.srv import Simm
 

def get_imminence(data):
    objects = data.objects
    ego_vel    = data.linear_velocity
    ego_acc    = data.linear_acceleration
    location   = data.location
    parameters = data.x
    gain = parameters[12]
    bias = parameters[13]
    imm_max = parameters[14]
    
    # Set arbitrarily large initial smallest distance
    # smallest_distance = 10000000000
    distances = []
    for i in range(0,len(objects)):     
        # Get smallest distance to object from list
        
        if objects[i].type != "DontCare":
            location = objects[i].location
            npLocation = np.array([location.x,location.y,location.z])
            distance = np.linalg.norm(npLocation)
            distances.append(distance)
            # if distance < smallest_distance:
            #     smallest_distance = distance
    dist_sum = sum(distances)
    if dist_sum == 0:
        imminence_par= 0 
    else:
        imminence_par = gain / sum(distances)
    # If no objects or really far, set to nan
    # if smallest_distance == 10000000000:
    #     smallest_distance = float('nan')
    # time_headway = (smallest_distance / ego_vel.x)
    # param_pub.publish(frame_imu,time_headway)     
    # if time_headway > 5:
    #     imminence_par = 0
    # elif time_headway > 4:
    #     imminence_par = 0.2
    # elif time_headway > 3:
    #     imminence_par = .4
    # elif time_headway > 2:
    #     imminence_par = .6
    # elif time_headway > 1:
    #     imminence_par = 1
    # elif time_headway > 0:
    #     imminence_par = 3
    # else:
    #     imminence_par = 0
    # print('Gain:',gain,' Bias:',bias,' THW:',time_headway)
    # imminence_par = (gain / time_headway) 
    # if imminence_par > imm_max:
    #     imminence_par = imm_max
    # if imminence_par < 0:
    #     imminence_par = 0 
    return imminence_par

def type_server():
    rospy.init_node('htpm_imminence_server')
    server_type = rospy.Service('parameter/imminence', Simm, get_imminence)
    rospy.spin()

type_server()  

