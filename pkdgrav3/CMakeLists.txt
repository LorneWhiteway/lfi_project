cmake_minimum_required(VERSION 3.1)
project(pkdgrav3 VERSION 3.0.4)
enable_language(Fortran)

# Uncomment this line for more information during building.
# set( CMAKE_VERBOSE_MAKEFILE on )

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/mdl2)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})

include(CheckFunctionExists)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Release build with debug info selected")
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

include(CheckCXXCompilerFlag)

set(TARGET_ARCHITECTURE "auto" CACHE STRING "CPU architecture to optimize for.")
if(TARGET_ARCHITECTURE STREQUAL "auto")
  CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
  if (COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
    add_compile_options(-march=native)
  endif()
  CHECK_CXX_COMPILER_FLAG("/arch:AVX" COMPILER_OPT_ARCH_AVX_SUPPORTED)
  if (COMPILER_OPT_ARCH_AVX_SUPPORTED)
    add_compile_options(/arch:AVX)
  endif()
else()
    add_compile_options(-march=${TARGET_ARCHITECTURE})
endif()
CHECK_CXX_COMPILER_FLAG("-Wall" COMPILER_OPT_WARN_ALL_SUPPORTED)
if (COMPILER_OPT_WARN_ALL_SUPPORTED)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
endif()

find_package(MPI REQUIRED)      # Added by LW to get tostd and psout able to compile on Wilkes ampere.
find_package(CUDA)
find_package(GSL REQUIRED)      # GNU Scientific Library
find_package(HDF5 COMPONENTS C HL)


#find_package(FFTW REQUIRED)
set(FFTW_FOUND TRUE)
set(FFTW_LIBRARIES ${FFTW_ROOT}/lib)
set(FFTW_INCLUDES ${FFTW_ROOT}/include)

#find_package(PythonLibs)
# _GNU_SOURCE gives us more options
INCLUDE(CheckCSourceCompiles)
check_c_source_compiles("
#include <features.h>
#ifdef __GNU_LIBRARY__
  int main() {return 0;} 
#endif
" _GNU_SOURCE)
if (_GNU_SOURCE)
  set(CMAKE_REQUIRED_DEFINITIONS -D_GNU_SOURCE)
endif()

# Check for restrict keyword
# Builds the macro A_C_RESTRICT form automake
foreach(ac_kw __restrict __restrict__ _Restrict restrict)
  check_c_source_compiles(
  "
  typedef int * int_ptr;
  int foo (int_ptr ${ac_kw} ip) {
    return ip[0];
  }
  int main(){
    int s[1];
    int * ${ac_kw} t = s;
    t[0] = 0;
    return foo(t);
  }
  "
  RESTRICT)
  if(RESTRICT)
    set(ac_cv_c_restrict ${ac_kw})
    break()
  endif()
endforeach()
if(RESTRICT)
  add_definitions("-Drestrict=${ac_cv_c_restrict}")
else()
  add_definitions("-Drestrict=")
endif()


INCLUDE (CheckIncludeFiles)
INCLUDE (CheckLibraryExists)
CHECK_INCLUDE_FILES (malloc.h HAVE_MALLOC_H)
CHECK_INCLUDE_FILES (signal.h HAVE_SIGNAL_H)
CHECK_INCLUDE_FILES (sys/time.h HAVE_SYS_TIME_H)
CHECK_INCLUDE_FILES (sys/stat.h HAVE_SYS_STAT_H)
CHECK_INCLUDE_FILES (sys/param.h HAVE_SYS_PARAM_H)
CHECK_INCLUDE_FILES (unistd.h HAVE_UNISTD_H)
CHECK_INCLUDE_FILES (inttypes.h HAVE_INTTYPES_H)
CHECK_INCLUDE_FILES(rpc/types.h HAVE_RPC_TYPES_H)
CHECK_INCLUDE_FILES(rpc/xdr.h HAVE_RPC_XDR_H)

INCLUDE(CheckSymbolExists)
check_symbol_exists(floor math.h HAVE_FLOOR)
check_symbol_exists(pow math.h HAVE_POW)
check_symbol_exists(sqrt math.h HAVE_SQRT)
check_symbol_exists(strchr string.h HAVE_STRCHR)
check_symbol_exists(strrchr string.h HAVE_STRRCHR)
check_symbol_exists(strdup string.h HAVE_STRDUP)
check_symbol_exists(strstr string.h HAVE_STRSTR)
check_symbol_exists(memmove string.h HAVE_MEMMOVE)
check_symbol_exists(memset string.h HAVE_MEMSET)
check_symbol_exists(gettimeofday sys/time.h HAVE_GETTIMEOFDAY)
check_symbol_exists(posix_memalign stdlib.h HAVE_POSIX_MEMALIGN)

check_symbol_exists(wordexp wordexp.h HAVE_WORDEXP)
check_symbol_exists(wordfree wordexp.h HAVE_WORDFREE)
check_symbol_exists(glob glob.h HAVE_GLOB)
check_symbol_exists(globfree glob.h HAVE_GLOBFREE)
check_symbol_exists(gethostname unistd.h HAVE_GETHOSTNAME)
check_symbol_exists(getpagesize unistd.h HAVE_GETPAGESIZE)
check_symbol_exists(mkdir sys/stat.h HAVE_MKDIR)
check_symbol_exists(strverscmp string.h HAVE_STRVERSCMP)

check_symbol_exists(backtrace execinfo.h USE_BT)

#AC_CHECK_FUNCS([gethrtime read_real_time time_base_to_time clock_gettime mach_absolute_time])
check_symbol_exists(atexit stdlib.h HAVE_ATEXIT)

add_subdirectory(blitz)
add_subdirectory(mdl2)

add_executable(${PROJECT_NAME} "")
target_compile_features(${PROJECT_NAME} PRIVATE cxx_auto_type cxx_lambdas)
set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY COMPILE_DEFINITIONS _LARGEFILE_SOURCE)
target_sources(${PROJECT_NAME} PRIVATE
	main.c cosmo.c master.c pst.c fio.c illinois.c param.c
	pkd.c analysis.c smooth.c smoothfcn.c outtype.c output.c
	walk2.c grav2.c ewald.cxx ic.cxx tree.cxx opening.cxx pp.cxx pc.cxx cl.c
	lst.c moments.c ilp.c ilc.c iomodule.c
	fof.c hop.c group.c groupstats.c RngStream.c listcomp.c healpix.c
	tinypy.c pkdtinypy.c
	gridinfo.cxx assignmass.cxx measurepk.cxx whitenoise.cxx pmforces.cxx
)
add_executable(tostd tostd.c fio.c)
add_executable(psout psout.c cosmo.c)

target_link_libraries(${PROJECT_NAME} m)
target_link_libraries(tostd m)
target_link_libraries(psout m)

if(USE_BT)
target_sources(${PROJECT_NAME} PRIVATE bt.c)
endif()

CHECK_INCLUDE_FILES(libaio.h HAVE_LIBAIO_H)
if (HAVE_LIBAIO_H)
  CHECK_LIBRARY_EXISTS(aio io_setup "" HAVE_LIBAIO)
  if(HAVE_LIBAIO)
    find_library(LIBAIO_LIBRARY aio)
    target_link_libraries(${PROJECT_NAME} ${LIBAIO_LIBRARY})
  endif()
endif()
CHECK_INCLUDE_FILES(aio.h HAVE_AIO_H)
if (HAVE_AIO_H)
  CHECK_LIBRARY_EXISTS(rt aio_read "" HAVE_RT)
  if(HAVE_RT)
    target_link_libraries(${PROJECT_NAME} rt)
  endif()
endif()

if (CUDA_FOUND)
  set(USE_CUDA TRUE)
  get_property(MDL_INCLUDES TARGET mdl2 PROPERTY INCLUDE_DIRECTORIES)
  get_property(OPENPA_INCLUDES TARGET openpa PROPERTY INCLUDE_DIRECTORIES)
#  target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
  CUDA_INCLUDE_DIRECTORIES(${MDL_INCLUDES} ${OPENPA_INCLUDES})
  CUDA_COMPILE(cuda_files cudaewald.cu cudapppc.cu cudautil.cu
	OPTIONS -arch compute_61
		-I${CMAKE_CURRENT_SOURCE_DIR}/mdl2/openpa
		-I${CMAKE_CURRENT_BINARY_DIR})
  target_sources(${PROJECT_NAME} PRIVATE ${cuda_files})
endif()
if (HDF5_FOUND)
  set(USE_HDF5 TRUE)
  target_include_directories(${PROJECT_NAME} PUBLIC ${HDF5_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME} ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})
  #set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY COMPILE_DEFINITIONS ${HDF5_DEFINITIONS})
  #set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY COMPILE_DEFINITIONS H5_USE_16_API)
  target_include_directories(tostd PUBLIC ${HDF5_INCLUDE_DIRS})
  target_link_libraries(tostd ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})
  target_include_directories(psout PUBLIC ${HDF5_INCLUDE_DIRS})
  target_link_libraries(psout ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})
endif(HDF5_FOUND)
if (PYTHONLIBS_FOUND)
  set(USE_PYTHON TRUE)
  target_include_directories(${PROJECT_NAME} PUBLIC ${PYTHON_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME} ${PYTHON_LIBRARIES})
  target_sources(${PROJECT_NAME} PRIVATE pkdpython.c)
endif(PYTHONLIBS_FOUND)

#if USE_SIMD
#if USE_SIMD_FMM
target_sources(${PROJECT_NAME} PRIVATE vmoments.cxx)
#endif
#if USE_SIMD_LC
target_sources(${PROJECT_NAME} PRIVATE lightcone.cxx)
#endif
#endif




foreach(flag_var
        CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
   if(${flag_var} MATCHES "-DNDEBUG")
      string(REPLACE "-DNDEBUG" "" ${flag_var} "${${flag_var}}")
   endif(${flag_var} MATCHES "-DNDEBUG")
endforeach(flag_var)



CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/pkd_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/pkd_config.h)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(${PROJECT_NAME} mdl2 openpa)
target_include_directories(${PROJECT_NAME} PRIVATE ${GSL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${GSL_LIBRARIES})
target_link_libraries(${PROJECT_NAME} blitz)
target_include_directories(tostd PRIVATE ${MPI_C_INCLUDE_PATH}) # Added by LW to get tostd and psout able to compile on Wilkes ampere.
target_include_directories(tostd PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(psout PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(psout PRIVATE ${GSL_INCLUDE_DIRS})
target_include_directories(psout PUBLIC ${MPI_C_INCLUDE_PATH}) # Added by LW to get tostd and psout able to compile on Wilkes ampere.
target_link_libraries(psout ${GSL_LIBRARIES})

install(TARGETS ${PROJECT_NAME} tostd DESTINATION "bin")
