cmake_minimum_required(VERSION 3.18)
project(mma_gemm LANGUAGES CXX CUDA)

# 设置构建类型（默认Release，可通过-DCMAKE_BUILD_TYPE=Debug切换为调试模式）
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 对 CUDA 链接启用 lld（需 CUDA 11.2+ 支持）
set(CMAKE_CUDA_LINK_FLAGS "${CMAKE_CUDA_LINK_FLAGS} -fuse-ld=lld")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=lld")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
# 查找 CUDA LibTorch
find_package(CUDAToolkit REQUIRED)
set(LIBTORCH ${CMAKE_CURRENT_SOURCE_DIR}/../libtorch)
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH}")
find_package(Torch REQUIRED)

file(GLOB_RECURSE CSRCS "csrc/*.cpp")
file(GLOB_RECURSE CUSRCS "csrc/*.cu")

set(TARGET_NAME ${PROJECT_NAME})

# 创建可执行目标
add_executable(${TARGET_NAME}
  ${CSRCS}
  ${CUSRCS}
)

target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)

# 链接 CUDA 运行时库
target_link_libraries(${TARGET_NAME} "${TORCH_LIBRARIES}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall -Wextra -arch=sm_89")

message("${CMAKE_BUILD_TYPE}")
# 区分 Release 和 Debug 模式的编译选项
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(${TARGET_NAME} PRIVATE
    # Debug：-O0 -G 以支持gdb和断点
    # $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0 -g -G> # -G是CUDA调试关键选项
    # $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-O0 -g>  # -g生成调试符号
    # Debug: -O2 -lineinfo 接近release行为
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O3 -g -lineinfo --use_fast_math>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-O3 -g>  # -g生成调试符号
  )
else()
  target_compile_options(${TARGET_NAME} PRIVATE
    # Release 模式：优化为主
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3 --use_fast_math>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3>
  )
endif()
