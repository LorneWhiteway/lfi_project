# lfi_project
Likelihood-free inference project for the [Dark Energy Survey](https://www.darkenergysurvey.org/) (project 330)

Contains files for using [pkdgrav3](https://bitbucket.org/dpotter/pkdgrav3/src).

## Changes made to pkdgrav3:

Original version of pkdgrav3 was 3.0.4, master branch commit d908272.

LW has made the following changes to this code:

1. Amended the top level CMakeLists.txt to allow tostd and psout to see MPI include files.

2. Amended cudautil.cu to fix calls to cudaSetDevice and to cudaGetDeviceProperties to handle the case in which iCore is negative. See my notes p. 69.

3. Added some helpful compile-time output to ewald.cxx so that we can see which SIMD compiler flags are set (which ones are set depends crucially on the chipset of the CPU on which the code is compiled).

4. Amended simd.h to fix a typo: changed a cast_fvec to cast_dvec for __SSE2__.

5. Amended lightcone.cxx to fix the 'holes in the lightcone' problem. This involved un-hard-coding an assumption that the SIMD data vector was of length 4 (as in some cases - e.g. the CPU that controls the V100 GPU on splinter - it is of length 2). Checked-in on 19 March 2021.

6. Amended master.c to suppress the creation of output snapshot files. (This can be amended if necessary to allow certain snapshots to be written).

7. Amended mdl2/mpi/mdl.c so that hwloc_topology_destroy is called only if bDedicated == 1 (as otherwise the topology object will not have been initialised and the call to hwloc_topology_destroy will cause a crash).

## Information about working with Wilkes cluster

1. Project directory is `/rds/user/dc-whit2/rds-dirac-dp153/lfi_project`. Can use shortcut `s` to get to the parent of this directory.

2. The following instructions are for using the ampere partition. To use the old pascal partition, contact LW.

3. In what follows:
- `<PROJECT-CODE-GPU>` is the project code (contact LW or NJ to get this).
- `<experiment>` denotes an 'experiment' i.e. a subdirectory of `/rds/user/dc-whit2/rds-dirac-dp153/lfi_project/experiments/` containing a control file `control.par`. Example: `gpu_256_1024_900`.

### How to log on to one of the Wilkes GPU nodes
```
sintr -A <PROJECT-CODE-GPU> -p ampere -t 1:0:0 --exclusive
```
For more information on `sintr` see [here](https://docs.hpc.cam.ac.uk/hpc/user-guide/interactive.html#sintr); the program has the same interface as `sbatch` (so the example above requests an interactive session for one hour).

### How to build pkdgrav3 for use by the Wilkes GPUs
- Log on to a GPU
- Go to `/rds/user/dc-whit2/rds-dirac-dp153/lfi_project`
- Run `./cm.sh build_wilkes`. You will need to type `y` to confirm the build directory name.

### How to run one of the experiments on the Wilkes GPUs
- Log on to Wilkes (but don't log on to a GPU)
- Go to `/rds/user/dc-whit2/rds-dirac-dp153/lfi_project/experiments/<experiment>`
- `sbatch ../../scripts/cuda_job_script_wilkes`

### Information about Wilkes specifically for LW

1. When logging on to Wilkes (<login-gpu.hpc.cam.ac.uk>), it seems to be necessary to type password from keyboard, rather than cutting-and-posting from password safe (despite PuTTY Window/Selection/Ctrl+Shift+{C,V} being set correctly). This seems to be related to the version of PuTTY.

2. LW has set up the the shortcut `s` to get to the parent of the project directory.

3. Use WebDrive to connect to login-gpu.hpc.cam.ac.uk and map it in Windows as the W drive. I have set up a symbolic link `work` in my home directory to point to the project directory.
