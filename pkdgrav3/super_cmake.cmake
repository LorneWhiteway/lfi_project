
#---------------------------------------------------------------------------------------------------
/share/splinter/ucapwhi/lfi_project/pkdgrav3/CMakeLists.txt
#---------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.1)
project(pkdgrav3 VERSION 3.0.4)

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
    message(STATUS "Setting -march=${TARGET_ARCHITECTURE}")
    add_compile_options(-march=${TARGET_ARCHITECTURE})
endif()
CHECK_CXX_COMPILER_FLAG("-Wall" COMPILER_OPT_WARN_ALL_SUPPORTED)
if (COMPILER_OPT_WARN_ALL_SUPPORTED)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
endif()
find_package(CUDA) # An alternative to this would be to replace the 'project' line (at the top) with project(pkdgrav3 VERSION 3.0.4 LANGUAGES C CXX CUDA)
find_package(GSL REQUIRED)      # GNU Scientific Library
find_package(HDF5 COMPONENTS C HL)
find_package(FFTW REQUIRED)
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
	OPTIONS -arch compute_35
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

target_include_directories(tostd PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(psout PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(psout PRIVATE ${GSL_INCLUDE_DIRS})
target_link_libraries(psout ${GSL_LIBRARIES})

install(TARGETS ${PROJECT_NAME} tostd DESTINATION "bin")

#---------------------------------------------------------------------------------------------------
/share/splinter/ucapwhi/lfi_project/pkdgrav3/blitz/CMakeLists.txt
#---------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.1)
project(blitz)


set(BLITZ_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(BLITZ_HEADERS
    ${BLITZ_INCLUDE_DIR}/blitz/array.h
)

add_library(blitz STATIC globals.cpp)
#target_include_directories(blitz INTERFACE $<BUILD_INTERFACE:${BLITZ_INCLUDE_DIR}>
#					   $<INSTALL_INTERFACE:include>)
target_include_directories(blitz INTERFACE ${BLITZ_INCLUDE_DIR})
target_include_directories(blitz PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/bzconfig.h.in ${CMAKE_CURRENT_BINARY_DIR}/bzconfig.h)
install(FILES ${BLITZ_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/blitz)

#---------------------------------------------------------------------------------------------------
/share/splinter/ucapwhi/lfi_project/pkdgrav3/mdl2/CMakeLists.txt
#---------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.1)
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Release build with debug info selected")
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
project(mdl2)
add_subdirectory(openpa)	# Lockless queue support
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
include(FindPkgConfig)
find_package(FFTW)
if(FFTW_FOUND)
  set(MDL_FFTW ${FFTW_FOUND})
endif()
pkg_check_modules(HWLOC hwloc)
INCLUDE (CheckIncludeFiles)
CHECK_INCLUDE_FILES (malloc.h HAVE_MALLOC_H)
CHECK_INCLUDE_FILES (time.h HAVE_TIME_H)
CHECK_INCLUDE_FILES (sys/time.h HAVE_SYS_TIME_H)
CHECK_INCLUDE_FILES (unistd.h HAVE_UNISTD_H)
CHECK_INCLUDE_FILES (signal.h HAVE_SIGNAL_H)
CHECK_INCLUDE_FILES (inttypes.h HAVE_INTTYPES_H)
CHECK_INCLUDE_FILES (stdint.h HAVE_STDINT_H)

INCLUDE(CheckLibraryExists)
CHECK_LIBRARY_EXISTS(memkind hbw_posix_memalign "" HAVE_MEMKIND)

find_package(Threads REQUIRED)
find_package(MPI REQUIRED)	# MPI support
find_package(CUDA)

add_library(${PROJECT_NAME} STATIC "")
target_sources(${PROJECT_NAME}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/mpi/mdl.c
          ${CMAKE_CURRENT_SOURCE_DIR}/mdlbase.c
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/mpi/mdl.h
          ${CMAKE_CURRENT_SOURCE_DIR}/mdlbase.h
          ${CMAKE_CURRENT_BINARY_DIR}/mdl_config.h
)
if (HWLOC_FOUND)
  find_library(HWLOC_LIBRARY hwloc HINTS ${HWLOC_LIBDIR})
  CHECK_LIBRARY_EXISTS(hwloc hwloc_topology_init ${HWLOC_LIBDIR} HAVE_HWLOC)
  if (HAVE_HWLOC)
    set(USE_HWLOC TRUE)
    target_include_directories(${PROJECT_NAME} PUBLIC ${HWLOC_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${HWLOC_LIBRARY})
  else()
    message(WARNING
	" Found hwloc library but cannot link to it, so we won't use it.\n"
	" On Cray you can try: export CRAYPE_LINK_TYPE=dynamic")
  endif()
endif()
if (APPLE)
  target_sources(${PROJECT_NAME}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/mac/pthread_barrier.c
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/mac/pthread_barrier.h
  )
endif()
if (MSVC)
  target_sources(${PROJECT_NAME}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/windows/pthread.c
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/windows/pthread.h
  )
  target_link_libraries(${PROJECT_NAME} wsock32 ws2_32)
endif()
find_package(Threads REQUIRED)
find_package(MPI REQUIRED)	# MPI support
if (APPLE)
  target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/mac)
endif()
if (MSVC)
  target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/windows)
endif()
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/openpa)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/mpi)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC ${MPI_C_INCLUDE_PATH})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..) ############################
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/..)

target_link_libraries(${PROJECT_NAME} ${MPI_C_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} openpa)

if(FFTW_FOUND)
target_link_libraries(${PROJECT_NAME} ${FFTW_LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC ${FFTW_INCLUDES})
endif()
if(HAVE_MEMKIND)
find_library(MEMKIND_LIBRARY memkind)
target_link_libraries(${PROJECT_NAME} ${MEMKIND_LIBRARY})
endif()

if (CUDA_FOUND)
  set(USE_CUDA TRUE)
  get_property(MDL_INCLUDES TARGET mdl2 PROPERTY INCLUDE_DIRECTORIES)
  get_property(OPENPA_INCLUDES TARGET openpa PROPERTY INCLUDE_DIRECTORIES)
  CUDA_INCLUDE_DIRECTORIES(${MDL_INCLUDES} ${OPENPA_INCLUDES})
  target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
endif(CUDA_FOUND)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/mdl_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/mdl_config.h)
install(TARGETS ${PROJECT_NAME} DESTINATION "lib")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mdl_config.h mpi/mdl.h mdlbase.h DESTINATION "include")

#---------------------------------------------------------------------------------------------------
/share/splinter/ucapwhi/lfi_project/pkdgrav3/mdl2/FindFFTW.cmake
#---------------------------------------------------------------------------------------------------

# - Find the FFTW library
#
# Usage:
#   find_package(FFTW [REQUIRED] [QUIET] )
#     
# It sets the following variables:
#   FFTW_FOUND               ... true if fftw is found on the system
#   FFTW_LIBRARIES           ... full path to fftw library
#   FFTW_INCLUDES            ... fftw include directory
#
# The following variables will be checked by the function
#   FFTW_USE_STATIC_LIBS    ... if true, only static libraries are found
#   FFTW_ROOT               ... if set, the libraries are exclusively searched
#                               under this path
#   FFTW_LIBRARY            ... fftw library to use
#   FFTW_INCLUDE_DIR        ... fftw include directory
#

#If environment variable FFTWDIR is specified, it has same effect as FFTW_ROOT
if( NOT FFTW_ROOT AND ENV{FFTWDIR} )
  set( FFTW_ROOT $ENV{FFTWDIR} )
elseif( NOT FFTW_ROOT AND ENV{FFTW3_ROOT_DIR} )
  set( FFTW_ROOT $ENV{FFTW3_ROOT_DIR} )
endif()

# Check if we can use PkgConfig
find_package(PkgConfig)

#Determine from PKG
if( PKG_CONFIG_FOUND AND NOT FFTW_ROOT )
  pkg_check_modules( PKG_FFTW QUIET "fftw3" )
endif()

#Check whether to search static or dynamic libs
#set( CMAKE_FIND_LIBRARY_SUFFIXES_SAV ${CMAKE_FIND_LIBRARY_SUFFIXES} )

#if( ${FFTW_USE_STATIC_LIBS} )
#  set( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX} )
#else()
#  set( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_SHARED_LIBRARY_SUFFIX} )
#endif()

if( FFTW_ROOT )

  find_library(
    FFTWF_LIB
    NAMES "fftw3f"
    PATHS ${FFTW_ROOT}
    PATH_SUFFIXES "lib" "lib64"
    NO_DEFAULT_PATH
  )

  find_library(
    FFTWF_MPI_LIB
    NAMES "fftw3f_mpi"
    PATHS ${FFTW_ROOT}
    PATH_SUFFIXES "lib" "lib64"
    NO_DEFAULT_PATH
  )

   find_library(
    FFTWF_THREADS_LIB
    NAMES "fftw3f_threads"
    PATHS ${FFTW_ROOT}
    PATH_SUFFIXES "lib" "lib64"
    NO_DEFAULT_PATH
  )
 
  #find includes
  find_path(
    FFTW_INCLUDES
    NAMES "fftw3-mpi.h"
    PATHS ${FFTW_ROOT}
    PATH_SUFFIXES "include"
    NO_DEFAULT_PATH
  )

else()

  find_library(
    FFTWF_LIB
    NAMES "fftw3f"
    PATHS ${PKG_FFTW_LIBRARY_DIRS} $ENV{FFTW_DIR} ${LIB_INSTALL_DIR}
  )

   find_library(
    FFTWF_MPI_LIB
    NAMES "fftw3f_mpi"
    PATHS ${PKG_FFTW_LIBRARY_DIRS} $ENV{FFTW_DIR} ${LIB_INSTALL_DIR}
  )

   find_library(
    FFTWF_THREADS_LIB
    NAMES "fftw3f_threads"
    PATHS ${PKG_FFTW_LIBRARY_DIRS} $ENV{FFTW_DIR} ${LIB_INSTALL_DIR}
  )

  find_path(
    FFTW_INCLUDES
    NAMES "fftw3-mpi.h"
    PATHS ${PKG_FFTW_INCLUDE_DIRS} $ENV{FFTW_INC} ${INCLUDE_INSTALL_DIR}
  )
endif( FFTW_ROOT )

set(FFTW_LIBRARIES ${FFTWF_MPI_LIB} ${FFTWF_THREADS_LIB} ${FFTWF_LIB})

#set( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_SAV} )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW DEFAULT_MSG
                                  FFTW_INCLUDES FFTW_LIBRARIES)

mark_as_advanced(FFTW_INCLUDES FFTW_LIBRARIES FFTW_LIB FFTWF_LIB FFTWL_LIB FFTWF_THREADS_LIB FFTWF_MPI_LIB)

#---------------------------------------------------------------------------------------------------
/share/splinter/ucapwhi/lfi_project/pkdgrav3/mdl2/openpa/CMakeLists.txt
#---------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 2.4)
project(openpa)

INCLUDE(CheckTypeSize)
check_type_size ("int" OPA_SIZEOF_INT)

INCLUDE (CheckIncludeFiles)
CHECK_INCLUDE_FILES (pthread.h OPA_HAVE_PTHREAD_H)
CHECK_INCLUDE_FILES (stddef.h OPA_HAVE_STDDEF_H)

INCLUDE (CheckFunctionExists)
CHECK_FUNCTION_EXISTS(sched_yield OPA_HAVE_SCHED_YIELD)

INCLUDE (CheckCSourceCompiles)

CHECK_C_SOURCE_COMPILES("
int foo(char *,...) __attribute__ ((format(printf,1,2)));
int main() {return 0;}
" OPA_HAVE_GCC_ATTRIBUTE)

macro(CompilePrimitive HEADER NAME)
CHECK_C_SOURCE_COMPILES("
#define OPA_SIZEOF_INT ${OPA_SIZEOF_INT}
#define OPA_SIZEOF_VOID_P  ${CMAKE_SIZEOF_VOID_P}
#ifndef _opa_inline
#define _opa_inline inline
#endif
#ifndef _opa_restrict
#define _opa_restrict restrict
#endif
#ifndef _opa_const
#define _opa_const const
#endif
#ifdef HAVE_GCC_ATTRIBUTE
#define OPA_HAVE_GCC_ATTRIBUTE 1
#endif
#include \"${CMAKE_CURRENT_SOURCE_DIR}/opa_util.h\"
#include \"${CMAKE_CURRENT_SOURCE_DIR}/primitives/${HEADER}\"
int main() {
    OPA_int_t a, b;
    int c;

    OPA_store_int(&a, 0);
    OPA_store_int(&b, 1);
    c = OPA_load_int(&a);

    OPA_add_int(&a, 10);
    OPA_incr_int(&a);
    OPA_decr_int(&a);

    c = OPA_decr_and_test_int(&a);
    c = OPA_fetch_and_add_int(&a, 10);
    c = OPA_fetch_and_incr_int(&a);
    c = OPA_fetch_and_decr_int(&a);

    c = OPA_cas_int(&a, 10, 11);
    c = OPA_swap_int(&a, OPA_load_int(&b));

    OPA_write_barrier();
    OPA_read_barrier();
    OPA_read_write_barrier();
    return 0;
}" ${NAME})
endmacro(CompilePrimitive)

CompilePrimitive("opa_gcc_intel_32_64.h" OPA_HAVE_GCC_X86_32_64)
CompilePrimitive("opa_gcc_intel_32_64_p3.h" OPA_HAVE_GCC_X86_32_64_P3)
CompilePrimitive("opa_gcc_ia64.h" OPA_HAVE_GCC_AND_IA64_ASM)
CompilePrimitive("opa_gcc_ppc.h" OPA_HAVE_GCC_AND_POWERPC_ASM)
CompilePrimitive("opa_gcc_arm.h" OPA_HAVE_GCC_AND_ARM_ASM)
CompilePrimitive("opa_gcc_sicortex.h" OPA_HAVE_GCC_AND_SICORTEX_ASM)
CompilePrimitive("opa_gcc_intrinsics.h" OPA_HAVE_GCC_INTRINSIC_ATOMICS)
CompilePrimitive("opa_nt_intrinsics.h" OPA_HAVE_NT_INTRINSICS)
CompilePrimitive("opa_sun_atomic_ops.h" OPA_HAVE_SUN_ATOMIC_OPS)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/opa_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/opa_config.h)

add_library(${PROJECT_NAME} STATIC opa_primitives.c opa_queue.c)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

install(TARGETS ${PROJECT_NAME} DESTINATION "lib")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/opa_config.h opa_queue.h opa_primitives.h opa_util.h DESTINATION "include")
