#!/bin/bash

# Takes optional command-line argument to specify build directory name. Defaults to 'build_tursa'.

if [[ $# == 0 ]]; then
	BUILD_DIR="build_tursa"
else
	BUILD_DIR=$1
fi

echo "Build directory is " $BUILD_DIR
echo "Proceed? (y/n)"

read ans

if [[ $ans == "y" ]]; then
	cd /mnt/lustre/tursafs1/home/dp327/dp327/shared/lfi_project
	source ./set_environment_tursa.sh BUILD
	##git clone https://bitbucket.org/dpotter/pkdgrav3.git
	cd ./pkdgrav3
	rm -rf ./$BUILD_DIR
	mkdir $BUILD_DIR
	cd ./$BUILD_DIR
	cmake ..
	make -j 32
fi
