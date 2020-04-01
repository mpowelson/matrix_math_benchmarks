cmake_minimum_required(VERSION 3.5)
project(matrix_math_benchmarks VERSION 0.1.0 LANGUAGES CXX)
include(cmake/cmake_macros.cmake)

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
find_package(ArrayFire REQUIRED)
find_package(benchmark REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Torch REQUIRED)

add_run_benchmarks_target()
add_benchmark(${PROJECT_NAME}_dense_multiplication src/dense_multiplication.cpp)


#############
## Install ##
#############

# Mark executables and/or libraries for installation
install(TARGETS
  ${PROJECT_NAME}_dense_multiplication
    EXPORT ${PROJECT_NAME}-targets
    DESTINATION lib)
install(EXPORT ${PROJECT_NAME}-targets
    NAMESPACE benchmarks:: DESTINATION lib/cmake/${PROJECT_NAME})

## Mark cpp header files for installation
#install(DIRECTORY include/${PROJECT_NAME}
#  DESTINATION include
#  FILES_MATCHING
#    PATTERN "*.h"
#    PATTERN "*.hpp"
#    PATTERN ".svn" EXCLUDE
# )

install(FILES package.xml DESTINATION share/${PROJECT_NAME})

# Create cmake config files
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/${PROJECT_NAME}-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
    INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
    NO_CHECK_REQUIRED_COMPONENTS_MACRO)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY ExactVersion)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
    DESTINATION lib/cmake/${PROJECT_NAME})

export(EXPORT ${PROJECT_NAME}-targets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-targets.cmake)








