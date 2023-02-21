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


module purge
module load use.own
module load rocks-openmpi
module load gcc/10.1.0
module load gsl/2.6
module load pkdgrav3/fftw/3.3.10 # See README for more information about this module file.
module load hdf5/1.12.1

	
if [[ $BUILD_MODE == $TRUE ]]; then
	module load cmake/3.17.3
fi

# See README file entry 'Working with python on hypatia'
source env/bin/activate

