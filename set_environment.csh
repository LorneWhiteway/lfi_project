#!/bin/csh

set TRUE = 1
set FALSE = 0

set SZIP_REQUIRED    = $FALSE # Set to $TRUE when compiling on the SMP server; $FALSE when compiling on the login node. 
set PYTHON_REQUIRED  = $FALSE # Set to $TRUE if you want to use Python when _using_ PKDGRAV3. Set to $FALSE when _building_ PKDGRAV3 (at least on the CORES16 nodes).

module purge
#module use /share/apps/modulefiles/old

# For building pkdgrav3. A subset of these (e.g. at least gsl) is needed for running.
set MPI_MODULE                 = rocks-openmpi
set COMPILER_MODULE            = dev_tools/sep2019/gcc-7.4.0 # dev_tools/sep2019/gcc-9.2.0 is more up-to-date - but nvcc insists on gcc version <= 8
set CMAKE_MODULE               = dev_tools/sep2019/cmake-3.15.3
set GSL_MODULE                 = science/sep2019/gsl-2.6
set FFTW_MODULE                = fft/nov2019/fftw-3.3.4
set SZIP_MODULE                = utils/may2018/szip-2.1.1 # The SMP server doesn't seem to have libsz.so.2.
set HDF5_MODULE                = dev_tools/may2018/hdf5-1.10.2 # hdf5 isn't in /usr/include/ on the CORE16 compute servers.
set PYTHON_MODULE              = dev_tools/oct2018/python-Anaconda-3-5.3.0

module load $MPI_MODULE
module load $COMPILER_MODULE
module load $CMAKE_MODULE
module load $GSL_MODULE
module load $FFTW_MODULE
if $SZIP_REQUIRED then
	module load $SZIP_MODULE
endif
module load $HDF5_MODULE
if $PYTHON_REQUIRED then
    module load $PYTHON_MODULE
endif

