cmake_minimum_required(VERSION 3.5)
project(ur_simulation_gz)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
    src/twist_servo_cal.py
    src/kdl_parser_py.py
    DESTINATION lib/${PROJECT_NAME}
)

install(FILES
    src/ur5_arm.urdf
    DESTINATION share/${PROJECT_NAME}
)


if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)

endif()

ament_package()
