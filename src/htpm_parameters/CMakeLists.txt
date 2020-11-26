cmake_minimum_required(VERSION 2.8.3)
project(htpm_parameters)
find_package(catkin REQUIRED COMPONENTS
  roscpp 
  rospy 
  std_msgs 
  geometry_msgs 
  sensor_msgs 
  message_generation)

add_message_files( FILES 
  msg_par.msg
  msg_road.msg)

generate_messages(DEPENDENCIES 
  std_msgs 
  geometry_msgs 
  sensor_msgs
  htpm_kitti_parser
  )

catkin_package(CATKIN_DEPENDS
  message_runtime 
  roscpp rospy 
  std_msgs 
  geometry_msgs 
  sensor_msgs
  htpm_kitti_parser)


