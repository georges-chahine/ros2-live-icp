cmake_minimum_required(VERSION 3.5)
project(ros2_live_icp)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(libpointmatcher REQUIRED)
find_package(rosidl_default_generators REQUIRED)

add_executable(ros2_live_icp src/main.cpp)
ament_target_dependencies(ros2_live_icp rclcpp pcl_ros std_msgs tf2 tf2_ros sensor_msgs geometry_msgs libpointmatcher)
# target_link_libraries (traxx_mapper ${PCL_LIBRARIES}

install(TARGETS
  ros2_live_icp
  DESTINATION lib/${PROJECT_NAME})

add_definitions("-DENABLE_SSE")
set(CMAKE_BUILD_TYPE Debug)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line ctpplint skips (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
