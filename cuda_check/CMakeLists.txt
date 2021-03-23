cmake_minimum_required(VERSION 3.15)
project(cuda_check C)
find_package(CUDA)
add_executable(cuda_check cuda_check.c)
target_include_directories(cuda_check PUBLIC /usr/local/cuda/include/)
target_link_libraries(cuda_check /usr/local/cuda/lib64/)
target_link_libraries(cuda_check cuda)
set_target_properties(cuda_check PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/)

