#!/usr/bin/tcsh

# Takes optional command-line argument to specify build directory name. Defaults to 'build'.

if ( $# == 0) then
	set BUILD_DIR="build"
else
	set BUILD_DIR=$argv[1]
endif

echo "Build directory is " $BUILD_DIR
echo "Proceed? (y/n)"

if ($< == y) then
	cd /share/splinter/ucapwhi/lfi_project/
	source ./set_environment_splinter.csh BUILD
	##git clone https://bitbucket.org/dpotter/pkdgrav3.git
	cd ./pkdgrav3
	rm -rf ./$BUILD_DIR
	mkdir $BUILD_DIR
	cd ./$BUILD_DIR
	cmake -DFFTW_ROOT=$FFTW_ROOT ..
	make
endif
