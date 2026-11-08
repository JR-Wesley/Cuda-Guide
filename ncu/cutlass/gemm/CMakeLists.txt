cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(operator LANGUAGES CXX CUDA)

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

set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(${TARGET_NAME} PRIVATE
    &><$<COMPILE_LANGUAGE:CUDA>:$<<CONFIG:Debug>:-O0 -G>>)
else()
  target_compile_options(${TARGET_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -arch=sm_89 -Iineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
endif()
