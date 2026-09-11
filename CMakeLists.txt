cmake_minimum_required(VERSION 3.14)
project(SAM3_Inference LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 选项:是否使用GPU版本的ONNX Runtime与CUDA加速
option(USE_GPU "Use GPU version of ONNX Runtime and CUDA preprocessing" OFF)

# 定义3rdparty路径
set(3RDPARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)

# 根据USE_GPU选项选择ONNX Runtime与CUDA
if(USE_GPU)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()
    set(ORT_ROOT ${3RDPARTY_DIR}/onnxruntime-gpu)
    message(STATUS "Using GPU version of ONNX Runtime with CUDA (${CMAKE_CUDA_ARCHITECTURES})")
    add_definitions(-DUSE_CUDA)
else()
    set(ORT_ROOT ${3RDPARTY_DIR}/onnxruntime-cpu)
    message(STATUS "Using CPU version of ONNX Runtime")
endif()

# OpenCV设置
set(OPENCV_ROOT ${3RDPARTY_DIR}/opencv CACHE PATH "OpenCV installation directory")
if(EXISTS ${OPENCV_ROOT}/lib/cmake/opencv4)
    set(OpenCV_DIR ${OPENCV_ROOT}/lib/cmake/opencv4)
endif()
find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV: ${OpenCV_VERSION}")

# ONNX Runtime设置
set(ORT_INCLUDE_DIR ${ORT_ROOT}/include)
set(ORT_LIB_DIR ${ORT_ROOT}/lib)
unset(ORT_LIB CACHE)
find_library(ORT_LIB NAMES onnxruntime PATHS ${ORT_LIB_DIR} NO_DEFAULT_PATH REQUIRED)
message(STATUS "Found ONNXRuntime: ${ORT_LIB}")

# 源文件列表
set(SRCS
    sam3_inference.cpp
    SAM3Predictor.cpp
    SimpleTokenizer.cpp
)

set(EXTRA_LIBS)
if(USE_GPU)
    list(APPEND SRCS Preprocessing.cu)
    list(APPEND EXTRA_LIBS CUDA::cudart)
endif()

# 添加可执行文件
add_executable(sam3_inference ${SRCS})

# 设置 RPATH，确保运行时无需手动配置 LD_LIBRARY_PATH
set_target_properties(sam3_inference PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${ORT_LIB_DIR};${OPENCV_ROOT}/lib;\$ORIGIN/../3rdparty/onnxruntime-gpu/lib;\$ORIGIN/../3rdparty/onnxruntime-cpu/lib;\$ORIGIN/../3rdparty/opencv/lib"
)

# 包含路径
target_include_directories(sam3_inference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${ORT_INCLUDE_DIR}
)
if(USE_GPU)
    target_include_directories(sam3_inference PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

# 链接库
target_link_libraries(sam3_inference PRIVATE
    ${OpenCV_LIBS}
    ${ORT_LIB}
    ${EXTRA_LIBS}
)
