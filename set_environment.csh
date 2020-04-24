#!/bin/csh

# One (optional) command line parameter that should be BUILD (to use when building PKDGRAV3) or USE (when using PKDGRAV3 and associated utilities).
# Default is USE.

set TRUE = 1
set FALSE = 0

set BUILD_MODE=$FALSE
if ( $# > 0) then
	if ($argv[1] == "BUILD") then
		set BUILD_MODE=$TRUE
	endif	
endif

if ($BUILD_MODE) then
	echo "Setting environment variables for BUILDING pkdgrav3..."
else
	echo "Setting environment variables for USING pkdgrav3..."
endif	

set MPI_MODULE                 = rocks-openmpi
set COMPILER_MODULE            = dev_tools/sep2019/gcc-7.4.0 # dev_tools/sep2019/gcc-9.2.0 is more up-to-date - but nvcc insists on gcc version <= 8
set CMAKE_MODULE               = dev_tools/sep2019/cmake-3.15.3
set GSL_MODULE                 = science/sep2019/gsl-2.6
set FFTW_MODULE                = fft/nov2019/fftw-3.3.4
set HDF5_MODULE                = dev_tools/may2018/hdf5-1.10.2 # hdf5 isn't in /usr/include/ on the CORE16 compute servers.
set PYTHON_MODULE              = dev_tools/oct2018/python-Anaconda-3-5.3.0

module purge

module load $MPI_MODULE
module load $COMPILER_MODULE
module load $GSL_MODULE
module load $FFTW_MODULE
module load $HDF5_MODULE
	
if ($BUILD_MODE) then
	module load $CMAKE_MODULE
else
	module load $PYTHON_MODULE
endif



