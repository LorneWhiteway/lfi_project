#!/bin/bash

# Sets environment for building or using pkdgrav3 on hypatia.

# One (optional) command line parameter that should be BUILD (to use when building PKDGRAV3)
#   or USE (when using PKDGRAV3 and associated utilities). Default is USE.

# Example:
# > source ./set_environment_hypatia.sh BUILD
# or
# > source ./set_environment_hypatia.sh USE

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

USEOWN=use.own
MPI_MODULE=rocks-openmpi
COMPILER_MODULE=gcc/10.1.0
GSL_MODULE=gsl/2.6
FFTW_MODULE=ucapwhi/fftw/3.3.10
HDF5_MODULE=hdf5/1.12.1
CMAKE_MODULE=cmake/3.17.3
#CMAKE_MODULE=cmake/3.7.1

module purge
module load $USEOWN
module load $MPI_MODULE
module load $COMPILER_MODULE
module load $GSL_MODULE
module load $FFTW_MODULE
module load $HDF5_MODULE

	
if [[ $BUILD_MODE == $TRUE ]]; then
	module load $CMAKE_MODULE
fi

# See README file entry 'Working with python on hypatia'
source env/bin/activate

# Set environment variables used by CMake
#export FFTW3_ROOT=/share/rcifdata/ucapwhi/lfi_project/fftw-3.3.10/
#export FFTW_INCLUDE_DIR=/share/rcifdata/ucapwhi/lfi_project/fftw-3.3.10/include/
#export FFTWDIR=/share/apps/fftw-3.3.5/
export FFTW_ROOT=/share/rcifdata/ucapwhi/lfi_project/fftw-3.3.10/


