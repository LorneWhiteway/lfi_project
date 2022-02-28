#!/bin/bash

# Sets environment for building or using pkdgrav3 on Splinter.

# One (optional) command line parameter that should be BUILD (to use when building PKDGRAV3)
#   or USE (when using PKDGRAV3 and associated utilities). Default is USE.

# Example:
# > source ./set_environment.sh BUILD
# or
# > source ./set_environment.sh USE

TRUE=1
FALSE=0

BUILD_MODE=$FALSE
if [[ $# > 0 ]]; then
	if [[ $1 == "BUILD" ]]; then
		BUILD_MODE=$TRUE
	fi	
fi

if [[ $BUILD_MODE == $TRUE ]]; then
	echo "Setting environment variables for BUILDING pkdgrav3..."
else
	echo "Setting environment variables for USING pkdgrav3..."
fi	

MPI_MODULE=rocks-openmpi
COMPILER_MODULE=dev_tools/sep2019/gcc-7.4.0 # dev_tools/sep2019/gcc-9.2.0 is more up-to-date - but nvcc insists on gcc version <= 8
CMAKE_MODULE=dev_tools/sep2019/cmake-3.15.3
GSL_MODULE=science/sep2019/gsl-2.6
FFTW_MODULE=fft/nov2019/fftw-3.3.4
HDF5_MODULE=dev_tools/may2018/hdf5-1.10.2 # hdf5 isn't in /usr/include/ on the CORE16 compute servers.
PYTHON_MODULE=dev_tools/oct2018/python-Anaconda-3-5.3.0


module purge
module load $MPI_MODULE
module load $COMPILER_MODULE
module load $GSL_MODULE
module load $FFTW_MODULE
module load $HDF5_MODULE
	

if [[ $BUILD_MODE == $TRUE ]]; then
	module load $CMAKE_MODULE
else
	module load $PYTHON_MODULE
fi


