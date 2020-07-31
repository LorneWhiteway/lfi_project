cmake_minimum_required(VERSION 3.17)

project(small_test CXX)

set( CMAKE_VERBOSE_MAKEFILE on )

#find_package(HDF5 COMPONENTS C HL)
#link_directories(/usr/local/software/spack/spack-0.11.2/opt/spack/linux-rhel7-x86_64/gcc-5.4.0/hdf5-1.10.1-z7bre2kdm5mup3ooctobwo2aojtje7uo/lib/)
add_link_options(-v)
add_executable(small_test small_test.cpp)


if (HDF5_FOUND)
  target_include_directories(small_test PUBLIC ${HDF5_INCLUDE_DIRS})
  target_link_libraries(small_test ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})
  #target_link_libraries(small_test /usr/local/software/spack/spack-0.11.2/opt/spack/linux-rhel7-x86_64/gcc-5.4.0/hdf5-1.10.1-z7bre2kdm5mup3ooctobwo2aojtje7uo/lib/libhdf5.so)
  #target_link_libraries(small_test /usr/local/software/spack/spack-0.11.2/opt/spack/linux-rhel7-x86_64/gcc-5.4.0/hdf5-1.10.1-z7bre2kdm5mup3ooctobwo2aojtje7uo/lib/libhdf5_hl.so)
  #target_link_libraries(small_test /usr/local/software/spack/develop-290617/spack/opt/spack/linux-rhel7-x86_64/gcc-5.4.0/zlib-1.2.11-dmjwhmsgdbfx2wwsdipnswso6s7poyye/lib/libz.so)
  
  message(STATUS =============================================)
  message(STATUS =============================================)
  message(STATUS ${HDF5_LIBRARIES})
  message(STATUS =============================================)
  message(STATUS ${HDF5_HL_LIBRARIES})
  message(STATUS =============================================)
  message(STATUS =============================================)
endif(HDF5_FOUND)

