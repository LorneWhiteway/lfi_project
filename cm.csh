#!/usr/bin/tcsh

cd /share/splinter/ucapwhi/lfi_project/
source ./set_environment.csh
##git clone https://bitbucket.org/dpotter/pkdgrav3.git
cd ./pkdgrav3
rm -rf ./build
mkdir build
cd ./build
cmake -DFFTW_ROOT=$FFTW_ROOT ..
make
