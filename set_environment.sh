#!/bin/bash

# One (optional) command line parameter that should be BUILD (to use when building PKDGRAV3) or USE (when using PKDGRAV3 and associated utilities).
# Default is USE.

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

MPI_MODULE=openmpi-3.1.3-gcc-5.4.0-to5hchq
COMPILER_MODULE=gcc-5.4.0-gcc-4.8.5-fis24gg # On splinter nvcc insists on gcc version <= 8
export CC=/usr/local/software/spack/spack-0.11.2/opt/spack/linux-rhel7-x86_64/gcc-4.8.5/gcc-5.4.0-fis24ggupugiobii56fesif2y3qulpdr/bin/gcc
export CXX=/usr/local/software/spack/spack-0.11.2/opt/spack/linux-rhel7-x86_64/gcc-4.8.5/gcc-5.4.0-fis24ggupugiobii56fesif2y3qulpdr/bin/g++
export FC=/usr/local/software/spack/spack-0.11.2/opt/spack/linux-rhel7-x86_64/gcc-4.8.5/gcc-5.4.0-fis24ggupugiobii56fesif2y3qulpdr/bin/gfortran
CUDA_MODULE=cuda-9.0.176-gcc-5.4.0-n22ctry
CMAKE_MODULE=cmake/latest
GSL_MODULE=gsl-2.4-gcc-5.4.0-z4fspad
FFTW_MODULE=fftw-3.3.8-gcc-5.4.0-cypgfyw
# Don't use hdf5-1.10.1-gcc-5.4.0-3dwdydr as it doesn't have HF
HDF5_MODULE=hdf5-1.10.1-gcc-5.4.0-z7bre2k
PYTHON_MODULE=dev_tools/oct2018/python-Anaconda-3-5.3.0

####module purge

module load $MPI_MODULE
module load $COMPILER_MODULE
module load $CUDA_MODULE
module load $GSL_MODULE
module load $FFTW_MODULE
module load $HDF5_MODULE
	
if [[ $BUILD_MODE == $TRUE ]]; then
	module load $CMAKE_MODULE
else
	module load $PYTHON_MODULE
fi

# Added 18 October 2021 in attempt to allow direct compiling of cuda
if [[ $BUILD_MODE == $TRUE ]]; then
	# Got this from https://github.com/jetsonhacks/buildLibrealsense2TX/issues/13
	export CUDA_HOME=/usr/local/cuda
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
	export PATH=$PATH:$CUDA_HOME/bin
fi


