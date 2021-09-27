#!/usr/bin/tcsh

if ( $# == 0) then
	set BUILD_DIR="build"
else
	set BUILD_DIR=$argv[1]
endif

echo "Build directory is " $BUILD_DIR
echo "Proceed? (y/n)"

if ($< == y) then
	cd /share/splinter/ucapwhi/lfi_project/
	source ./set_environment.csh BUILD
	cd ./cuda_check
	rm -rf ./$BUILD_DIR
	mkdir $BUILD_DIR
	cd ./$BUILD_DIR
	cmake -D CMAKE_CUDA_COMPILER=/usr/local/cuda ..
	make
endif
