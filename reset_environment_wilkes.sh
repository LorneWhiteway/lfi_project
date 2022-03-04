#!/bin/bash

# Resets to starting (login) environment on Wilkes ampere nodes.
# Puts us in a state in which we can use slurm.

module purge
module load dot
module load slurm
module load turbovnc/2.0.1
module load vgl/2.5.1/64
module load singularity/current
module load rhel7/global
module load cuda/8.0
module load gcc-5.4.0-gcc-4.8.5-fis24gg
module load openmpi-1.10.7-gcc-5.4.0-jdc7f4f
module load cmake/latest
module load rhel7/default-gpu
