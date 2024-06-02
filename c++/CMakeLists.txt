cmake_minimum_required(VERSION 3.21)

project(myapp)

set (TENSORFLOW_DIR /opt/tensorflow)
find_package( OpenCV REQUIRED )

# threads
find_package(Threads)

# C++11(or C++14)
set (CMAKE_CXX_STANDARD 14)

# PIC
set (CMAKE_POSITION_INDEPENDENT_CODE ON)

set (CORE_SOURCE
    ${CMAKE_SOURCE_DIR}/src/main.cpp
)

link_directories(
    ${TENSORFLOW_DIR}/lib/
    )

add_executable( myapp
    ${CORE_SOURCE}
    )

target_include_directories(myapp
    PUBLIC ${TENSORFLOW_DIR}
    
    PUBLIC ${TENSORFLOW_DIR}/lib/include

    ${OpenCV_INCLUDE_DIRS}

    # for array_ops.h
    # PUBLIC ${TENSORFLOW_DIR}/bazel-genfiles

    # # headers for external packages
    # PUBLIC ${TENSORFLOW_EXTERNAL_DIR}/external/protobuf_archive/src
    # PUBLIC ${TENSORFLOW_EXTERNAL_DIR}/external/eigen_archive
    # PUBLIC ${TENSORFLOW_EXTERNAL_DIR}/external/nsync/public

    # this project
    PUBLIC ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries( myapp
    tensorflow_cc
    tensorflow_framework
    ${CMAKE_THREAD_LIBS_INIT}
    ${CMAKE_DL_LIBS}
    ${OpenCV_LIBS}
    )

