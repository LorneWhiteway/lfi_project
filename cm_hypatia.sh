#!/bin/bash

# Runs cmake and then make to build pkdgrav3.
# Takes optional command-line argument to specify build directory name. Defaults to 'build'.

if [[ $# == 0 ]]; then
	BUILD_DIR="build"
else
	BUILD_DIR=$1
fi

echo "Build directory is " $BUILD_DIR
echo "Proceed? (y/n)"

read ans

if [[ $ans == "y" ]]; then
	cd ${LFI_PROJECT_DIRECTORY}
	source ./set_environment_hypatia.sh
	##git clone https://bitbucket.org/dpotter/pkdgrav3.git
	cd ./pkdgrav3
	rm -rf ./$BUILD_DIR
	mkdir $BUILD_DIR
	cd ./$BUILD_DIR
	cmake ..
	make -j 12
fi
