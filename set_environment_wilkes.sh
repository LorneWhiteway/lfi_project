#!/bin/bash

# Sets environment for building or using pkdgrav3 on Wilkes ampere nodes.

# One (optional) command line parameter that should be BUILD (to use when building PKDGRAV3)
#   or USE (when using PKDGRAV3 and associated utilities). Default is USE.

# Example:
# > source ./set_environment.sh BUILD
# or
# > source ./set_environment.sh USE

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

# Following modules (appropriate for Wilkes ampere nodes) suggested to
# LW by Kacper Kornet, Wilkes support, 22 Oct2021.
BASE_MODULE=rhel8/default-amp
COMPILER_MODULE=gcc/9.4.0
GSL_MODULE=gsl/2.7
FFTW_MODULE=fftw/3.3.9
HDF5_MODULE=hdf5/1.10.7
CMAKE_MODULE=cmake/3.21.3

module purge
module load $BASE_MODULE
module load $COMPILER_MODULE
module load $GSL_MODULE
module load $FFTW_MODULE
module load $HDF5_MODULE

	
if [[ $BUILD_MODE == $TRUE ]]; then
	module load $CMAKE_MODULE
fi

source env/bin/activate


