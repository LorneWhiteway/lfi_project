#!/bin/bash

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
	cd /data/gower_st/shared/lfi_project
	source ./set_environment_locust.sh BUILD
	##git clone https://bitbucket.org/dpotter/pkdgrav3.git
	cd ./pkdgrav3
	rm -rf ./$BUILD_DIR
	mkdir $BUILD_DIR
	cd ./$BUILD_DIR
	cmake ..
	make
fi
