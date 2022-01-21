#!/bin/bash

# Sets environment for building or using pkdgrav3 on Tursa ampere nodes.

# One (optional) command line parameter that should be BUILD (to use when building PKDGRAV3)
#   or USE (when using PKDGRAV3 and associated utilities). Default is USE.

# Example:
# > source ./set_environment_tursa.sh BUILD
# or
# > source ./set_environment_tursa.sh USE

TRUE=0
FALSE=1

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

BASE_MODULE=use.own
COMPILER_MODULE=gcc/9.3.0
MPI_MODULE=openmpi/4.1.1-cuda11.4.1
GSL_MODULE=lw-gsl-2.7
FFTW_MODULE=foo2
HDF5_MODULE=foo3
CMAKE_MODULE=lw-cmake-3.22.1


module purge
module load $BASE_MODULE
module load $COMPILER_MODULE
module load $MPI_MODULE
module load $GSL_MODULE
#module load $FFTW_MODULE
#module load $HDF5_MODULE

	
if [[ $BUILD_MODE == $TRUE ]]; then
	module load $CMAKE_MODULE
fi


