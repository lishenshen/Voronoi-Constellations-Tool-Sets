cmake_minimum_required(VERSION 3.8)
project(VC)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(VC main.cpp)
target_link_libraries(VC ${OpenCV_LIBS})

message("-- CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER_ID}")
message("-- CMAKE_C_COMPILER: ${CMAKE_C_COMPILER_ID}")
message("-- CMAKE_C_COMPILER_VERSION: ${CMAKE_C_COMPILER_VERSION}")
message("-- CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")
