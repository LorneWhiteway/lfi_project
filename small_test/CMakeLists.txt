cmake_minimum_required(VERSION 3.17)

project(small_test C CXX)

find_package(HDF5 COMPONENTS C HL)
add_executable(small_test small_test.cpp)

target_include_directories(small_test PUBLIC ${HDF5_INCLUDE_DIRS})
target_link_libraries(small_test ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})

