#!/bin/bash

# Sets environment for building or using pkdgrav3 on Splinter.
# Example:
# > source ./set_environment.sh


echo "Setting environment variables for building or using pkdgrav3..."

MPI_MODULE=rocks-openmpi
COMPILER_MODULE=dev_tools/sep2019/gcc-7.4.0 # dev_tools/sep2019/gcc-9.2.0 is more up-to-date - but nvcc insists on gcc version <= 8
CMAKE_MODULE=dev_tools/sep2019/cmake-3.15.3
GSL_MODULE=science/sep2019/gsl-2.6
FFTW_MODULE=fft/nov2019/fftw-3.3.4
HDF5_MODULE=dev_tools/may2018/hdf5-1.10.2 # hdf5 isn't in /usr/include/ on the CORE16 compute servers.


module purge
module load $MPI_MODULE
module load $COMPILER_MODULE
module load $CMAKE_MODULE
module load $GSL_MODULE
module load $FFTW_MODULE
module load $HDF5_MODULE


echo "Still to do - load Python environment."
