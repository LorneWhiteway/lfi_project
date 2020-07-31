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
	cd /rds/user/dc-whit2/rds-dirac-dp153/lfi_project/small_test/
	rm -rf ./$BUILD_DIR
	mkdir $BUILD_DIR
	cd ./$BUILD_DIR
	cmake ..
	make
fi
