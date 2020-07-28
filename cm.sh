#!/bin/bash

# Takes optional command-line argument to specify build directory name. Defaults to 'build'.

if [[ $# == 0 ]]; then
	BUILD_DIR="build"
else
	BUILD_DIR=$1
fi

echo "Build directory is " $BUILD_DIR
echo "Proceed? (y/n)"

read -n1 ans

if [[ $ans == "y" ]]; then
	cd /rds/user/dc-whit2/rds-dirac-dp153/lfi_project
	source ./set_environment.sh BUILD
	##git clone https://bitbucket.org/dpotter/pkdgrav3.git
	cd ./pkdgrav3
	rm -rf ./$BUILD_DIR
	mkdir $BUILD_DIR
	cd ./$BUILD_DIR
	#cmake -DFFTW_ROOT=$FFTW_ROOT ..
	cmake -DHDF5_ROOT=$HDF5_ROOT -DCMAKE_POLICY_DEFAULT_CMP0074=NEW ..
	make
fi
