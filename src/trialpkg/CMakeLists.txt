cmake_minimum_required(VERSION 3.8)
project(trialpkg)

# Find dependencies
find_package(ament_cmake REQUIRED)
#find_package(ament_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate custom messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/WorkpieceInfo.msg"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

# Install your Python package
ament_python_install_package(${PROJECT_NAME})

# Install shared resources (like launch, urdf, etc.)
install(DIRECTORY
  launch
  config
  urdf
  meshes
  worlds
  DESTINATION share/${PROJECT_NAME}
)

# Required so Python code can use the generated messages
ament_package()
