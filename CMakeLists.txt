cmake_minimum_required(VERSION 3.10)
project(SAM3_Inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 启用 CUDA 语言支持
enable_language(CUDA)

# 现代 FindCUDAToolkit 方式用于库路径
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 75) # RTX 2060 对应的架构

# 选项:是否使用GPU版本的ONNX Runtime
option(USE_GPU "Use GPU version of ONNX Runtime" OFF)

# 定义3rdparty路径
set(3RDPARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
set(OPENCV_ROOT ${3RDPARTY_DIR}/opencv)

# 根据USE_GPU选项选择ONNX Runtime版本
if(USE_GPU)
    set(ORT_ROOT ${3RDPARTY_DIR}/onnxruntime-gpu)
    message(STATUS "Using GPU version of ONNX Runtime")
    add_definitions(-DUSE_CUDA)
else()
    set(ORT_ROOT ${3RDPARTY_DIR}/onnxruntime-cpu)
    message(STATUS "Using CPU version of ONNX Runtime")
endif()

# OpenCV设置
set(OpenCV_DIR ${OPENCV_ROOT}/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ONNX Runtime设置
set(ORT_INCLUDE_DIR ${ORT_ROOT}/include)
set(ORT_LIB_DIR ${ORT_ROOT}/lib)

include_directories(${ORT_INCLUDE_DIR})
link_directories(${ORT_LIB_DIR})

# 添加可执行文件
add_executable(sam3_inference 
    sam3_inference.cpp 
    SAM3Predictor.cpp 
    SimpleTokenizer.cpp 
    Preprocessing.cu
)

# 链接库及头文件
target_include_directories(sam3_inference PRIVATE ${OpenCV_INCLUDE_DIRS} ${ORT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
target_link_libraries(sam3_inference 
    ${OpenCV_LIBS}
    onnxruntime
    CUDA::cudart
)
