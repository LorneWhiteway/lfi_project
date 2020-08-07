cmake_minimum_required(VERSION 3.17)

project(small_test C)


#link_directories(/usr/local/software/spack/spack-0.11.2/opt/spack/linux-rhel7-x86_64/gcc-5.4.0/hdf5-1.10.1-z7bre2kdm5mup3ooctobwo2aojtje7uo/lib/)

add_executable(small_test small_test.c)

#target_link_libraries(small_test /usr/local/software/spack/spack-0.11.2/opt/spack/linux-rhel7-x86_64/gcc-5.4.0/hdf5-1.10.1-z7bre2kdm5mup3ooctobwo2aojtje7uo/lib/libhdf5.so)
#target_link_libraries(small_test /usr/lib64/libhdf5.so)

#target_include_directories(small_test PUBLIC /usr/local/software/spack/spack-0.11.2/opt/spack/linux-rhel7-x86_64/gcc-5.4.0/hdf5-1.10.1-z7bre2kdm5mup3ooctobwo2aojtje7uo/include/)




find_package(HDF5 COMPONENTS C HL)

message(STATUS =============================================)
message(STATUS =============================================)
message(STATUS ${HDF5_LIBRARIES})
message(STATUS =============================================)
message(STATUS =============================================)

target_include_directories(small_test PUBLIC ${HDF5_INCLUDE_DIRS})
target_link_libraries(small_test ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})

