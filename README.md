# lfi_project
Likelihood-free inference project for the [Dark Energy Survey](https://www.darkenergysurvey.org/) (project 330)

Contains files for using [pkdgrav3](https://bitbucket.org/dpotter/pkdgrav3/src).

## Changes made to pkdgrav3:

Original version of pkdgrav3 was 3.0.4, master branch commit d908272.

LW has made the following changes to this code:

1. Amended the top level CMakeLists.txt to turn on policy CMP0060: did this so that an incorrect old version of HDF5 library is not found on the Wilkes machine.

2. Amended cudautil.cu to fix calls to cudaSetDevice and to cudaGetDeviceProperties to handle the case in which iCore is negative. See my notes p. 69.

3. Added some helpful compile-time output to ewald.cxx so that we can see which SIMD compiler flags are set (which ones are set depends crucially on the chipset of the CPU on which the code is compiled).

4. Amended simd.h to fix a typo: changed a cast_fvec to cast_dvec for __SSE2__.

5. Amended lightcone.cxx to fix the 'holes in the lightcone' problem. This involved un-hard-coding an assumption that the SIMD data vector was of length 4 (as in some cases - e.g. the CPU that controls the V100 GPU on splinter - it is of length 2). Checked-in on 19 March 2021.

6. Amended master.c to suppress the creation of output snapshot files. (This can be amended if necessary to allow certain snapshots to be written).

## Information about working with Wilkes cluster

1. When logging on to Wilkes (<login-gpu.hpc.cam.ac.uk>), it seems to be necessary to type password from keyboard, rather than cutting-and-posting from password safe (despite PuTTY Window/Selection/Ctrl+Shift+{C,V} being set correctly). Not sure why.

2. Project directory is `/rds/user/dc-whit2/rds-dirac-dp153/lfi_project`. Can use shortcut `s` to get to the parent of this directory.

3. Can use WebDrive to connect to login-gpu.hpc.cam.ac.uk as (for example) the W drive. It is convenient to set up a symbolic link (suggest calling it `work`) in the home directory to point to the project directory.

4. To set the environment on Wilkes, use `set_environment.sh` (and not `set_environment.csh`, which is for splinter).


## How to interact with the GPUs on Wilkes

In what follows:
- `<partition>` refers to either `pascal` (the old partition with [V100](https://en.wikipedia.org/wiki/Volta_(microarchitecture)) GPUs) or `ampere` (the new partition with [A100](https://en.wikipedia.org/wiki/Ampere_(microarchitecture)) GPUs.
- `<PROJECT-CODE-GPU>` is the project code (contact LW or NJ to get this).
- `<experiment>` denotes an 'experiment' i.e. a subdirectory of `/rds/user/dc-whit2/rds-dirac-dp153/lfi_project/experiments/` containing a control file `control.par`. Example: `gpu_256_1024_900`.

### How to log on to one of the GPU nodes
```
sintr -A <PROJECT-CODE-GPU> -p <partition> -t 1:0:0 --exclusive
```
For more information on `sintr` see [here](https://docs.hpc.cam.ac.uk/hpc/user-guide/interactive.html#sintr); the progam has the same interface as `sbatch` (so the example above requests an interactive session for one hour).

### How to build pkdgrav3 for use by the Wilkes GPUs
- Log on to a GPU
- Go to `/rds/user/dc-whit2/rds-dirac-dp153/lfi_project`
- Run `./cm.sh build_<partition>`

### How to run one of the experiments on the Wilkes GPUs
- Log on to Wilkes (but don't log on to a GPU)
- Go to `/rds/user/dc-whit2/rds-dirac-dp153/lfi_project`
- `source set_environment.sh USE`
- Go to `/rds/user/dc-whit2/rds-dirac-dp153/lfi_project/experiments/<experiment>`
- `env EXPERIMENT=<experiment> sbatch ../../scripts/cuda_job_script_wilkes_<partition>`



