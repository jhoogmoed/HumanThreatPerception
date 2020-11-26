cmake_minimum_required(VERSION 2.8.3)
project(htpm_visualiser)
find_package(catkin REQUIRED COMPONENTS
  roscpp 
  rospy 
  std_msgs 
  geometry_msgs 
  sensor_msgs 
  )



catkin_package(CATKIN_DEPENDS
  message_runtime 
  roscpp rospy 
  std_msgs 
  geometry_msgs 
  sensor_msgs
  htpm_kitti_parser
  htpm_parameters)

