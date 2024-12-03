#!/bin/bash

# Sets environment for building or using pkdgrav3 on locust system at UCL.

echo "Setting environment variables for pkdgrav3..."

source /opt/ucl/spack/share/spack/setup-env.sh
spack env activate gower_st2

source /data/gower_st/lfi_project/env/bin/activate
export PYTHONPATH="${PYTHONPATH}:/data/gower_st/lfi_project/env/lib64/python3.11/site-packages"

