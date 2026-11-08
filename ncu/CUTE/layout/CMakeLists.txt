cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(operator LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
find_package(CUDAToolkit REQUIRED)
set(CUTLASS_PATH "/home/mikasa/hpc/cutlass/include/")
set(CUTLASS_UTIL_PATH "/home/mikasa/hpc/cutlass/tools/util/include")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

file(GLOB CUSRC "./*.cu")

get_filename_component(TARGET_NAME ${CUSRC} NAME_WE)
add_executable(${TARGET_NAME} ${CUSRC})

target_link_libraries(${TARGET_NAME} PRIVATE CUDA::cudart )
target_include_directories(${TARGET_NAME} PRIVATE
  ${CUTLASS_PATH}
  ${CUTLASS_UTIL_PATH}
)

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
