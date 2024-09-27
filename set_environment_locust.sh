#!/bin/bash

# Sets environment for building or using pkdgrav3 on locust system at UCL.

# One (optional) command line parameter that should be BUILD (to use when building PKDGRAV3)
#   or USE (when using PKDGRAV3 and associated utilities). Default is USE.

# Example:
# > source ./set_environment_locust.sh BUILD
# or
# > source ./set_environment_locust.sh USE

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

MODULE_AFTER_PURGE=/home/y07/shared/tursa-modules/setup-env
COMPILER_MODULE=/mnt/lustre/tursafs1/home/dp327/dp327/shared/lfi_project/tursa/modulefiles/lfi-gcc-9.3.0
CUDA_MODULE=cuda/12.3
MPI_MODULE=openmpi/4.1.5-cuda12.3
GSL_MODULE=/mnt/lustre/tursafs1/home/dp327/dp327/shared/lfi_project/tursa/modulefiles/lfi-gsl-2.7
FFTW_MODULE=/mnt/lustre/tursafs1/home/dp327/dp327/shared/lfi_project/tursa/modulefiles/lfi-fftw-3.3.10
HDF5_MODULE=/mnt/lustre/tursafs1/home/dp327/dp327/shared/lfi_project/tursa/modulefiles/lfi-hdf5-1.14.4-2
CMAKE_MODULE=cmake/3.27.4

module purge
module load $MODULE_AFTER_PURGE
module load $COMPILER_MODULE
module load $CUDA_MODULE
module load $MPI_MODULE
module load $GSL_MODULE
module load $FFTW_MODULE
module load $HDF5_MODULE

	
if [[ $BUILD_MODE == $TRUE ]]; then
	module load $CMAKE_MODULE
fi

source /data/gower_st/lfi_project/env/bin/activate

