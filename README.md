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

## Information about working with the splinter cluster

1. Project directory is `/share/splinter/ucapwhi/lfi_project`.

2. In what follows, `<GPU_NAME>` hould be replaced by one of `v100` or `k80`, depending on which GPU you wish to use.

### How to log on to one of the splinter GPU nodes
```
srun -p GPU --gres=gpu:<GPU_NAME>:1 --pty tcsh
```

### How to build pkdgrav3 for use by a splinter GPU
- Log on to a GPU and go to the project directory.
- Run `./cm.csh build_splinter_<GPU_NAME>`. You will need to type `y` to confirm the build directory name.

### How to run one of the experiments on a splinter GPU
- Log on to splinter (but don't log on to a GPU)
- Go to the project directory, and from there to `/experiments/<experiment>`
- `sbatch --export=ALL,experiment_name='<experiment>' ../../scripts/cuda_job_script_splinter_<GPU_NAME>`


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
- Log on to a GPU and go to the project directory.
- Run `./cm.sh build_wilkes`. You will need to type `y` to confirm the build directory name.

### How to run one of the experiments on the Wilkes GPUs
- Log on to Wilkes (but don't log on to a GPU)
- Go to the project directory, and from there to `/experiments/<experiment>`
- `sbatch ../../scripts/cuda_job_script_wilkes`

### Working with python on the Wilkes cluster

I have set up a virtual Python environment in the subdirectory `env` of the project directory. This is convenient as we can install our own software there.

To enter the virtual environment:
- Go to the project directory
- `module load python/3.8`
- `source env/bin/activate`

To leave the virtual environment:
- `deactivate`

For reference (e.g. in case it needs to be repeated), here's how the virtual environment was created:
- Go to the project directory
- `module load python/3.8`
- `python3 -m venv env`
- `source env/bin/activate`
- `cd env`
- `pip install --upgrade pip`
- `pip install healpy`

### Information about Wilkes specifically for LW

1. When logging on to Wilkes (<login-gpu.hpc.cam.ac.uk>), it seems to be necessary to type password from keyboard, rather than cutting-and-posting from password safe (despite PuTTY Window/Selection/Ctrl+Shift+{C,V} being set correctly). This seems to be related to the version of PuTTY.

2. I have set up the the shortcut `s` to get to the parent of the project directory.

3. Use WebDrive to connect to login-gpu.hpc.cam.ac.uk and map it in Windows as the W drive. I have set up a symbolic link `work` in my home directory to point to the project directory.

## Python scripts

### pkdgrav3_postprocess.py

This is the main script for post-processing PKDGRAV3 output to create full lightcones and image files. Recall that PKDGRAV3 outputs each lightcone in several sections, one section per file. These files need to be concatenated (and trimmed) to get the full lightcone; that is what this script does. The script is 'lazy' in that it will not try to create files that it can see already exist (but this can be overridden using the `--force` option).

Syntax: 
```
pkdgrav3_postprocess.py [options] directory
```
| Options | Meaning |
| --- | --- |
| -l or --lightcone | create full lightcone files |
| -m or --mollview | create image files in mollview format |
| -o or --orthview | create image files in ortho format |
| -z or --zfile | create a text file specifying the redshift ranges for the lightcones |
| -s or --status | print a summary of the files in the directory |
| -a or --all | all of the above |
| -f or --force | create output files even if they are present already |
| -h or --help | print this help message, then exit |

Here is further information about output format. Here XXXXX stands for a 5 character left-zero-padded tomographic slice identifier e.g. 00076.
| Type | Name format | Comments |
| --- | --- | --- |
| Lightcone files | run.XXXXX.lightcone.npy | Numpy file (i.e. can be opened with [np.load](https://numpy.org/doc/stable/reference/generated/numpy.load.html)) containing healpix array (in the default ordering i.e. ring) of object counts per pixel. The NSIDE is as specified when PKDGRAV3 was run. The data-type used in the files is [uint16](https://numpy.org/doc/stable/reference/arrays.scalars.html#sized-aliases) (unless the values are too large to allow this). |
| Image files | run.XXXXX.lightcone.mollview.png or run.XXXXX.lightcone.orthview.png | Image file (in png format) showing the lightcone. The plotted quantity is the base-10 logarithm of the 'normalised object count' (by 'normalised object count' we mean the pixel object count divided by the average object count across all pixels). The plots are in [mollview](https://healpy.readthedocs.io/en/latest/generated/healpy.visufunc.mollview.html) or [orthview](https://healpy.readthedocs.io/en/latest/generated/healpy.visufunc.orthview.html) format as requested. |
| Redshift ranges | z_values.txt | Text file describing the redshift ranges for the tomographic slices. The file has one header row, refer to this for details of the columns in the file. Each slice is described by the redshift, by the comoving distance (in Mpc/h) and by the comoving distance (in units of the box length) for the far endpoint and for the near endpoint of the slice; the width of the slice (in the same terms) is also given. | 

