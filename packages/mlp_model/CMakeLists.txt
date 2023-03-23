cmake_minimum_required(VERSION 2.8.3)
project(mlp_model)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  nav_msgs
  std_msgs
  message_generation
  duckietown_msgs
)

catkin_python_setup()

add_service_files(
  FILES
  MLPPredict.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  duckietown_msgs
)

add_dependencies(${catkin_EXPORTED_TARGETS})

catkin_package(
  CATKIN_DEPENDS
  message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)