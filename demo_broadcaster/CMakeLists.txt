# cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

# project(demo_broadcaster)

# set(Torch_DIR /usr/local/libtorch/share/cmake/Torch)
# find_package(Torch REQUIRED)
# find_package(OpenCV REQUIRED)

# #SET( CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-pthread")

# include_directories(${OpenCV_INCLUED_DIRS})
# include_directories("/usr/local/toml11-master")
# add_executable(demo_broadcaster main_broadcaster.cpp Broadcaster.cpp)

# target_link_libraries(demo_broadcaster ${TORCH_LIBRARIES})
# target_link_libraries(demo_broadcaster pthread)
# target_link_libraries(demo_broadcaster avcodec avformat swresample avutil swscale)
# target_link_libraries(demo_broadcaster ${OpenCV_LIBS})

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.3/bin/nvcc)

project(demo_broadcaster)
find_package(PythonInterp REQUIRED)
set(Torch_DIR /usr/local/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

#SET( CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-pthread")

include_directories(${OpenCV_INCLUED_DIRS})
include_directories("/usr/local/toml11-master")
add_executable(demo_broadcaster main_broadcaster.cpp Broadcaster.cpp)

target_link_libraries(demo_broadcaster ${TORCH_LIBRARIES})
target_link_libraries(demo_broadcaster pthread)
target_link_libraries(demo_broadcaster avcodec avformat swresample avutil swscale)
target_link_libraries(demo_broadcaster ${OpenCV_LIBS})