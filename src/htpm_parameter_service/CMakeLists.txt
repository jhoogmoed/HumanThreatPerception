cmake_minimum_required(VERSION 2.8.3)
project(htpm_parameter_service)
find_package(catkin REQUIRED COMPONENTS
  roslaunch
  roscpp 
  rospy 
  std_msgs 
  geometry_msgs 
  sensor_msgs 
  message_generation)

roslaunch_add_file_check(launch)

add_message_files( FILES 
  msg_kitti_object.msg
  msg_kitti_object_list.msg
  msg_kitti_oxts.msg
)

add_service_files(
  FILES
  Stype.srv
)

generate_messages(DEPENDENCIES 
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
  )



