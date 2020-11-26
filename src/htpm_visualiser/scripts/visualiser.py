#!/usr/bin/env python

# Import main
import sys
import os

# Import select
import rospy
import numpy as np

# Import specific ros
from kitti_parser.msg import kitti_object_list
from kitti_parameters.msg import param_type
from std_msgs.msg import Float32
from geometry_msgs.msg import Vector3