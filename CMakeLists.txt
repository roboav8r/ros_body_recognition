cmake_minimum_required(VERSION 3.0.2)
project(ros_body_recognition)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs genmsg geometry_msgs ros_openpose cv_bridge image_transport)

include_directories(
    ${catkin_INCLUDE_DIRS} 
    include
)

## Declare ROS messages and services
# add_message_files(DIRECTORY msg FILES Num.msg)
# add_service_files(DIRECTORY srv FILES AddTwoInts.srv)

## Generate added messages and services
# generate_messages(DEPENDENCIES std_msgs ros_openpose)

## Declare a catkin package
catkin_package()


add_executable(body_rec_node src/main.cpp src/lbp.cpp)
target_link_libraries(body_rec_node ${catkin_LIBRARIES})
# add_dependencies(body_rec_node ros_body_recognition_generate_messages_cpp)