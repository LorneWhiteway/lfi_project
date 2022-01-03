# lfi_project

This repository contains code used in the likelihood-free inference project for the [Dark Energy Survey](https://www.darkenergysurvey.org/) (project 330).

Its primary purpose is to make it easy to run the N-body simulation code [pkdgrav3](https://bitbucket.org/dpotter/pkdgrav3/src), and to post-process the output from this program.

This repository contains a version of pkdgrav3 in which several bugs have been fixed.

pkdgrav3 runs most efficiently using GPU hardware, and we provide job files to run pkdgrav3 efficiently on the 'splinter' cluster at UCL and on the DiRAC 'Wilkes3' cluster at Cambridge.

The primary goal of the simulation is to create a sequence of snapshots (at a discrete sequence of time points) of the positions in 3 dimensional space of a collection of particles interacting via gravity. The sequence of snapshots is therefore a 4 dimensional object (3 spatial and 1 temporal dimension). For our purposes we are interested only in the lightcone, i.e. the 3 dimensional submanifold consisting of events that we (as observers at the centre of the simulation box) can actually observe. Within the simulation the lightcone thus becomes a sequence of snapshots (each representing a tomographic slice bounded by two adjacent time points, or equivalently two adjacent distances, or equivalently two adjacent redshifts) of the positions of the particles on the celestial sphere. These positions are then binned into (heal-)pixels to give pixel-by-pixel object counts. In short, the lightcone is a sequence of healpix maps.

pkdgrav3 is able to output lightcone files; as these are our only interest we have amended the pkdgrav3 code to _only_ output these files (i.e. the 3 dimensional snapshot files are not output). Each healpix map is output in several separate pieces (each in one file) and this repository provides the functionality to reconstitute the full healpix map from these parts. We also provide functionality to create images of the lightcone files, and to create a summary of the redshift boundaries of each tomographic slice.

pkdgrav3 creates lightcone snapshots by extracting objects from a thin shell of the appropriate radius within a 'superbox' consisting of 218 repeats of the simulation box in a 6x6x6 array (and we as observers are at the centre of the superbox). At early times the lightcone shell lies completely outside this box and hence is empty; in this case the post-processing script does not create a full healpix map (instead of creating an empty map).


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

2. In what follows, `<GPU_NAME>` should be replaced by one of `v100` or `k80`, depending on which GPU you wish to use.

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


## Information about working with the Wilkes cluster

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
- Then for nbodykit: (see https://nbodykit.readthedocs.io/en/latest/getting-started/install.html#installing-nbodykit-with-pip)
- `pip install cython`
- `pip install mpi4py`
- `pip install nbodykit`

### Information about Wilkes specifically for LW

1. When logging on to Wilkes (<login-gpu.hpc.cam.ac.uk>), it seems to be necessary to type password from keyboard, rather than cutting-and-posting from password safe (despite PuTTY Window/Selection/Ctrl+Shift+{C,V} being set correctly). This seems to be related to the version of PuTTY.

2. I have set up the the shortcut `s` to get to the parent of the project directory.

3. Use WebDrive to connect to login-gpu.hpc.cam.ac.uk and map it in Windows as the W drive. I have set up a symbolic link `work` in my home directory to point to the project directory.

## Python scripts

Scripts (in [Python3](https://www.python.org/)) are located in the `scripts` subdirectory of the project directory.

### pkdgrav3_postprocess.py

This is the main script for post-processing PKDGRAV3 output to create full healpix lightcone files and image files. Recall that PKDGRAV3 outputs each lightcone healpix array in several sections, one section per file. These files need to be concatenated (and trimmed) to get the full healpoix map; that is what this script does. The script is 'lazy' in that it will not try to create files that it can see already exist (but this can be overridden using the `--force` option).

Syntax: 
```
pkdgrav3_postprocess.py [options] directory
```
| Options | Meaning |
| --- | --- |
| -l or --lightcone | create full lightcone files |
| -d or --delete | delete the raw, partial lightcone files (and any _stub_ simulation box outputs) |
| -m or --mollview | create image files in mollview format |
| -o or --orthview | create image files in ortho format |
| -z or --zfile | create a text file specifying the redshift ranges for the lightcones |
| -s or --status | print a summary of the files in the directory |
| -a or --all | all of the above |
| -f or --force | create output files even if they are present already |
| -h or --help | print this help message, then exit |

Here is further information about output format. The slices are sequentially numbered from 1 (highest redshift) to nSteps (lowest redshift); in what follows, XXXXX stands for the 5 character left-zero-padded slice number e.g. 00076.
| Type | Name format | Comments |
| --- | --- | --- |
| Lightcone files | run.XXXXX.lightcone.npy | Numpy file (i.e. can be opened with [np.load](https://numpy.org/doc/stable/reference/generated/numpy.load.html)) containing healpix array (in the default ordering i.e. ring) of object counts per pixel. The NSIDE is as specified when PKDGRAV3 was run. The data-type used in the files is [uint16](https://numpy.org/doc/stable/reference/arrays.scalars.html#sized-aliases) (unless the values are too large to allow this). |
| Incomplete lightcone files | run.XXXXX.incomplete.npy | Same details as the lightcone files, except that this file is not to be used as this tomographic slice lies partly outside the superbox (see above for definition of superbox). |
| Image files | run.XXXXX.lightcone.mollview.png or run.XXXXX.lightcone.orthview.png | Image file (in png format) showing the lightcone. The plotted quantity is the base-10 logarithm of the 'normalised object count' (by 'normalised object count' we mean the pixel object count divided by the average object count across all pixels). The plots are in [mollview](https://healpy.readthedocs.io/en/latest/generated/healpy.visufunc.mollview.html) or [orthview](https://healpy.readthedocs.io/en/latest/generated/healpy.visufunc.orthview.html) format as requested. |
| Redshift ranges | z_values.txt | Text file describing the redshift ranges for the tomographic slices. The file has one header row; refer to this for details of the columns in the file. Each slice is described by the redshift, and by the comoving distance (calculated using a cosmology with the same matter density as that used in the simulation, and quoted both in Mpc/h and in box-length units), for the far endpoint and for the near endpoint of the slice; the width of the slice (in the same terms) is also given. |
| Object counts | object_count.pkl | A Python [pickle](https://docs.python.org/3/library/pickle.html) file containing a dictionary where each key is a tomographic slice number (1-based; 1 is the furthest slice) and the associated value is the number of objects in that slice. |

The first tomographic slice of the lightcone that pkdgrav3 populates is the one whose _near_ boundary is within the superbox (defined above); this slice has a _far_ boundary that is still outside the superbox and so this slice is only partially filled. The script therefore creates an 'incomplete' lightcone file for this slice.

## Transfer functions

pkdgrav3 requires an input file containing a transfer function (to specify the initial power spectrum). Such files can be created using the Python module [nbodykit](https://nbodykit.readthedocs.io/en/latest/index.html).

The script 'utility.py' has a function called 'make_specific_cosmology_transfer_function' that may be used to create transfer function files.

We are currently putting transfer function files in the same directory as the control file for that run.

### nbodykit on splinter

#### Installing nbodykit on splinter

I used the instructions at https://nbodykit.readthedocs.io/en/latest/getting-started/install.html#installing-nbodykit-with-anaconda, together with information from the splinter user manual.

To install on splinter:
```
bash
eval "$(/share/apps/anaconda/3-2019.07/bin/conda shell.bash hook)"
conda create --prefix /share/splinter/ucapwhi/lfi_project/nbodykit python=3.7 pip
conda install -c bccp nbodykit
conda install matplotlib
```

Note that `/home/ucapwhi/.conda` is a symbolic link to `/share/splinter/ucapwhi/.conda` - this is to avoid the 2 Gb disk quota in `/home/ucapwhi`.

#### Using nbodykit on splinter

To use nbodykit, do:
- Go to the project directory
- `bash`
- `source ./set_environment_splinter_nbodycode.sh`


### nbodykit on Wilkes3

#### Installing nbodykit on Wilkes3

This is included in the instructions (elsewhere in this README) about working with Python on Wilkes3.

#### Using nbodykit on Wilkes3

- Go to the project directory
- `module load python/3.8`
- `source env/bin/activate`


### Example code

If nbodykit is installed properly then the following Python3 code should work:
```
from nbodykit.lab import *
k = 0.00001
t = cosmology.power.transfers.CLASS(cosmology.Planck15,0.0)(k)
print(t) # Should be 1.0
```

