#!/bin/bash

# Sets environment for building or using pkdgrav3 on hypatia.

# Example:
# > source ./set_environment_hypatia.sh

echo "Setting environment variables for building or using pkdgrav3..."

# Got this from https://stackoverflow.com/questions/59895; it's the directory in which this script resides.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export LFI_PROJECT_DIRECTORY=$SCRIPT_DIR

module purge
module load use.own
module load rocks-openmpi
module load gcc/10.1.0
module load gsl/2.6
module load pkdgrav3/fftw/3.3.10 # See README for more information about this module file.
module load hdf5/1.12.1
module load cmake/3.17.3


# See README file entry 'Working with python on hypatia'
source env/bin/activate

