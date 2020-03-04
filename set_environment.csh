#!/bin/csh

# For building pkdgrav3. Perhaps also needed for running (haven't checked).
set MPI_MODULE                 = rocks-openmpi
set COMPILER_MODULE            = dev_tools/sep2019/gcc-9.2.0
set CMAKE_MODULE               = dev_tools/sep2019/cmake-3.15.3
set GSL_MODULE                 = science/sep2019/gsl-2.6
set FFTW_MODULE                = fft/nov2019/fftw-3.3.4
set HDF5_MODULE                = dev_tools/may2018/hdf5-1.10.2 # I added this because hdf5 isn't in /usr/include/ on the CORE16 compute servers.

# Needed for running the Python scripts.
set PYTHON_MODULE              = dev_tools/oct2018/python-Anaconda-3-5.3.0


module purge
module load $MPI_MODULE
module load $COMPILER_MODULE
module load $CMAKE_MODULE
module load $GSL_MODULE
module load $FFTW_MODULE
module load $HDF5_MODULE
module load $PYTHON_MODULE


