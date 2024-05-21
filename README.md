# lfi_project

This repository contains code used in the likelihood-free inference project for the [Dark Energy Survey](https://www.darkenergysurvey.org/) (project 330).

Its primary purpose is to make it easy to run the N-body simulation code [pkdgrav3](https://bitbucket.org/dpotter/pkdgrav3/src), and to post-process the output from this program.

This repository contains a version of pkdgrav3 in which several bugs have been fixed.

pkdgrav3 runs most efficiently using GPU hardware, and we provide job files to run pkdgrav3 efficiently on the 'splinter' and 'hypatia' clusters at UCL, the DiRAC 'Wilkes3' cluster at Cambridge, and the DiRAC 'Tursa' cluster at Edinburgh.

The primary goal of the simulation is to create a sequence of snapshots (at a discrete sequence of time points) of the positions in 3 dimensional space of a collection of particles interacting via gravity. The sequence of snapshots is therefore a 4 dimensional object (3 spatial and 1 temporal dimension). For our purposes we are interested only in the lightcone, i.e. the 3 dimensional submanifold consisting of events that we (as observers at the centre of the simulation box) can actually observe. Within the simulation the lightcone thus becomes a sequence of snapshots (each representing a tomographic slice bounded by two adjacent time points, or equivalently two adjacent distances, or equivalently two adjacent redshifts) of the positions of the particles on the celestial sphere. These positions are then binned into (heal-)pixels to give pixel-by-pixel object counts. In short, the lightcone is a sequence of healpix maps.

pkdgrav3 is able to output lightcone files; as these are our only interest we have amended the pkdgrav3 code to _only_ output these files (i.e. the 3 dimensional snapshot files are not output). Each healpix map is output in several separate pieces (each in one file) and this repository provides the functionality to reconstitute the full healpix map from these parts. We also provide functionality to create images of the lightcone files, and to create a summary of the redshift boundaries of each tomographic slice.

pkdgrav3 creates lightcone snapshots by extracting objects from a thin shell of the appropriate radius within a 'superbox' consisting of 218 repeats of the simulation box in a 6x6x6 array (and we as observers are at the centre of the superbox). At early times the lightcone shell lies completely outside this box and hence is empty; in this case the post-processing script does not create a full healpix map (instead of creating an empty map).

The paper [Jeffery et al. 2024](https://arxiv.org/abs/2403.02314) has further information. Lightcone snapshots created by this project have been made publically available as the [Gower Street Sims]( http://www.star.ucl.ac.uk/GowerStreetSims/). 


## Changes made to pkdgrav3:

Original version of pkdgrav3 was 3.0.4, master branch commit [d908272](https://bitbucket.org/dpotter/pkdgrav3/commits/d908272f2eb8be32ab1061d498dc8122e70dc7c9).

LW has made the following changes to this code:

1. Amended the top level CMakeLists.txt to allow tostd and psout to see MPI include files.

2. Amended cudautil.cu to fix calls to cudaSetDevice and to cudaGetDeviceProperties to handle the case in which iCore is negative. See LW's notes p. 69.

3. Added some helpful compile-time output to ewald.cxx so that we can see which SIMD compiler flags are set (which ones are set depends crucially on the chipset of the CPU on which the code is compiled).

4. Amended simd.h to fix a typo: changed a cast_fvec to cast_dvec for __SSE2__.

5. Amended lightcone.cxx to fix the 'holes in the lightcone' problem. This involved un-hard-coding an assumption that the SIMD data vector was of length 4 (as in some cases - e.g. the CPU that controls the V100 GPU on splinter - it is of length 2). Checked-in on 19 March 2021.

6. Amended master.c to suppress the creation of output snapshot files. (This can be amended if necessary to allow certain snapshots to be written).

7. Amended mdl2/mpi/mdl.c so that hwloc_topology_destroy is called only if bDedicated == 1 (as otherwise the topology object will not have been initialised and the call to hwloc_topology_destroy will cause a crash).

8. Increased number of replications of the simulation box at 'lightcone creation' time from 6^3 to 20^3. This change incorporates code made available by Janis Fluri (https://cosmo-gitlab.phys.ethz.ch/jafluri/pkdgrav3_dev) - many thanks to Janis! This change affected lightcone.cxx, pkd.c and pkd.h.

9. Amended pkdgrav3/mdl2/CMakeLists.txt to change the cmake minimum version from 3.1 to 3.12 and to set cmake policy 0074 to NEW behaviour. This is needed to ensure that the environment variable FFTW_ROOT is paid attention to by the FindFFTW.cmake module (which is located in \pkdgrav3\mdl2\).

## How to install the software

The main high-level steps to install the software are:
1. Checkout this repository using `git clone https://github.com/LorneWhiteway/lfi_project.git`. This will create a directory called 'lfi_project' which we will refer to as the 'project directory'. The 'git clone' command should be run in the parent directory of what will be the project directory. 
2. Install (where necessary) the dependent modules.
3. Build pkdgrav3.
4. Install the python environment and nbodykit.
Details of each step will depend on which cluster is being used - see details below.

## How to use the software

1. Go to the project directory
2. `source ./set_environment_XXX.sh` where 'XXX' refers to the cluster being used ('splinter', 'hypatia', 'wilkes', or 'tursa'.)

## Tursa cluster

1. Documentation for tursa is at https://epcced.github.io/dirac-docs/tursa-user-guide/.
2. Project directory is `/mnt/lustre/tursafs1/home/dp327/dp327/shared/lfi_project`.

### How to log on to one of the Tursa GPU nodes
```
srun --partition gpu --gres=gpu:1 --account=DP327 --qos=standard --nodes=1 --time=0:30:00 --pty bash
```
However, it probably isn't necessary to use this (as you can build pkdgrav3 using the login node).

### How to build pkdgrav3 for use by the Tursa GPUs

- Go to the project directory.
- Run `./cm_tursa.sh build_tursa`. You will need to type `y` to confirm the build directory name.


### How to install dependent libraries on Tursa

Tursa didn't have any of the dependent libraries installed, so I installed them myself. Each library is installed in a sibling directory to the project directory.

#### GSL

1. cd /mnt/lustre/tursafs1/home/dp327/dp327/shared/
2. wget https://ftp.nluug.nl/pub/gnu/gsl/gsl-2.7.tar.gz
3. tar -xvf gsl-2.7.tar.gz
4. cd gsl-2.7
5. ./configure --prefix=/mnt/lustre/tursafs1/home/dp327/dp327/shared/gsl-2.7
6. make
7. make install

#### FFTW

1. cd /mnt/lustre/tursafs1/home/dp327/dp327/shared/
2. wget https://www.fftw.org/fftw-3.3.10.tar.gz
3. tar -xvf fftw-3.3.10.tar.gz
4. cd fftw-3.3.10
5. ./configure --enable-float --enable-shared --enable-threads --enable-mpi --prefix=/mnt/lustre/tursafs1/home/dp327/dp327/shared/fftw-3.3.10
6. make
7. make install

#### HDF5

Here I put the source code 'one level down' as otherwise there were problems at 'make install' time.

1. cd /mnt/lustre/tursafs1/home/dp327/dp327/shared/
2. mkdir hdf5-1.14.4-2
3. cd hdf5-1.14.4-2
4. wget https://github.com/HDFGroup/hdf5/releases/download/hdf5_1.14.4.2/hdf5-1.14.4-2.tar.gz
5. tar -xvf hdf5-1.14.4-2.tar.gz
6. cd hdf5-1.14.4-2
7. ./configure --prefix=/mnt/lustre/tursafs1/home/dp327/dp327/shared/hdf5-1.14.4-2/
8. make
9. make install

#### Further Tursa configuration changes

1. Add `opal_cuda_support=0` to `$HOME/.openmpi/mca-params.conf` to suppress a CUDA warning when running nbodykit. You will still get an OenFabric warning, though...


## UCL hypatia cluster

Project directory is `/share/rcifdata/ucapwhi/lfi_project`.


### Installing dependent modules of hypatia
The hypatia installation needs its own copy of FFTW 3.3.10, compiled with all the necessary options. This goes in a subdirectory /fftw-3.3.10 of the project directory. To create this:
1. Go to the project directory
2. wget https://www.fftw.org/fftw-3.3.10.tar.gz
3. tar -xvf fftw-3.3.10.tar.gz
4. cd ./fftw-3.3.10
5. ./configure --enable-float --enable-shared --enable-threads --enable-mpi --prefix=(project directory)/fftw-3.3.10
6. make
7. make install

### How to build pkdgrav3 for use by the hypatia GPU
- Log on to a GPU via `srun -p GPU --gres=gpu:a100:1 --pty bash`.
- Go to the project directory.
- Run `./cm_hypatia.sh build_hypatia`. You will need to type `y` to confirm the build directory name.
- Then `exit` from the GPU.


## Wilkes cluster

1. Two possible addresses: `login-icelake.hpc.cam.ac.uk` (but note that with this address you can't run nbodykit) or `login-gpu.hpc.cam.ac.uk` (but note that with this address you can't run slurm).
2. Project directory is `/rds/user/dc-whit2/rds-dirac-dp153/lfi_project`.


### How to build pkdgrav3 for use by the Wilkes GPUs
- Log on to a GPU using `sintr -A DIRAC-DP153-GPU -p ampere -t 1:0:0 --exclusive` and go to the project directory.
- Run `./cm_wilkes.sh build_wilkes`. You will need to type `y` to confirm the build directory name.
- Then `exit` from the GPU.
For more information on `sintr` see [here](https://docs.hpc.cam.ac.uk/hpc/user-guide/interactive.html#sintr); the program has the same interface as `sbatch` (so the example above requests an interactive session for one hour).


### Information about Wilkes specifically for LW

1. Recall that when pasting passwords from KeePass into PuTTY, you should use Shift+Insert (and not Ctrl+V).
2. I have set up the the shortcut `s` to get to the project directory.
3. Use WebDrive to connect to login-gpu.hpc.cam.ac.uk and map it in Windows as the W drive. I have set up a symbolic link `work` in my home directory to point to the project directory.



## Information about working with the UCL splinter cluster

1. Project directory is `/share/splinter/ucapwhi/lfi_project`.

2. In what follows, you can replace `v100` with `k80` to use the other splinter GPU.

### How to log on to one of the splinter GPU nodes
```
srun -p GPU --gres=gpu:v100:1 --pty tcsh
```

### How to build pkdgrav3 for use by a splinter GPU
- Log on to a GPU and go to the project directory.
- Run `./cm.csh build_splinter_v100`. You will need to type `y` to confirm the build directory name.
- Then `exit` from the GPU.


## How to install a Python environment and nbodykit
To set up a virtual Python environment in the subdirectory `env` of the project directory:
- Go to the project directory
- If necessary, load a module file for Python: `module load python/3.6.4` on Hypatia, `module load python/3.8` on Wilkes; not needed on Tursa.
- `python3 -m venv env`
- `source env/bin/activate`
- `cd env`
- `pip install --upgrade pip`
- `pip install healpy==1.15.0`
- Then for nbodykit: (see https://nbodykit.readthedocs.io/en/latest/getting-started/install.html#installing-nbodykit-with-pip):
-    `pip install cython==0.29.33`
-    `pip install mpi4py==3.1.4`
-    `pip install nbodykit==0.3.15`
-    `pip install "dask[array]" --upgrade`

It is convenient to have this environment as we can install our own software there. The software versions listed are not the latest, but they are known to be compatible.

The environment can be activated using `source env/bin/activate` in the project directory; this is done automatically when `source ./set_environment_XXX.sh` is called.

If nbodykit is installed properly then the following Python3 code should work:
```
from nbodykit.lab import *
k = 0.00001
t = cosmology.power.transfers.CLASS(cosmology.Planck15,0.0)(k)
print(t) # Should be 1.0
```

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
| Redshift ranges | z_values.txt | Text file describing the redshift ranges for the tomographic slices. The file has one header row; refer to this for details of the columns in the file. Each slice is described by the redshift, and by the comoving distance (calculated using a cosmology with the same matter density as that used in the simulation, and quoted both in Mpc/h and in box-length units), for the far endpoint and for the near endpoint of the slice; the width and volume of the slice (in the same terms) are also given. |
| Object counts | object_count.pkl | A Python [pickle](https://docs.python.org/3/library/pickle.html) file containing a dictionary where each key is a tomographic slice number (1-based; 1 is the furthest slice) and the associated value is the number of objects in that slice. |

The first tomographic slice of the lightcone that pkdgrav3 populates is the one whose _near_ boundary is within the superbox (defined above); this slice has a _far_ boundary that is still outside the superbox and so this slice is only partially filled. The script therefore creates an 'incomplete' lightcone file for this slice.

### monitor.py

This script does 'on-the-fly' post-processing (with the goal of reducing peak disk usage). It runs in a permanent loop, repeatedly doing the same tasks as would be done by pkdgrav3_postprocess.py with options '-l -d -f'. Usage:

Syntax: 
```
monitor.py directory
```
The script operates on 'directory' as well as recursively on all subdirectories of 'directory'. The script periodically sleeps, upon awakening it searches for a file called 'monitor_stop.txt' in 'directory', and it stops if this file exists. On addition, at startup if the file 'monitor_wait.txt' exists in 'directory' then the script will pause, only continuing when this file no longer exists.


### expand_shell_script.py

This script creates shell scripts. Specify the name of a 'template' shell script; this script may contain the special string "{}" in various places, in which case a new shell script will be created containing repeats of the template with "{}" replaced by specified integers (zero padded to length three). 
Syntax: 
```
expand_shell_script.py original_shell_script_file_name new_shell_script_file_name list_of_jobs...
```

Example:
```
expand_shell_script.py copy_template.sh copy.sh 1-8,10 x3 x5
```
If in this example copy_template.py contained
```
cp ./run{}/run.log ./run{}.log
```
then copy.sh would be created and would contain
```
cp ./run001/run.log ./run001.log
cp ./run002/run.log ./run002.log
cp ./run004/run.log ./run004.log
cp ./run006/run.log ./run006.log
cp ./run007/run.log ./run007.log
cp ./run008/run.log ./run008.log
cp ./run010/run.log ./run010.log
```

Note that 'x' stands for 'exclude'. More complicated strings are possible, such as
```
1-32 64-128 140 141,142 x23-30 x75,76 26
```

### utility.py

This script contains various functions that may be called 'by hand' particularly to set up and run large numbers of pkdgrav3 jobs (with postprocessing).

| Function | Purpose |
| --- | --- |
| create_input_files_for_multiple_runs | Creates a set of input files for a number of runs. For each run the input files include a pkdgrav3 control file (control.par), a transfer function file (transfer_function.txt), a text file listing the cosmological parameters used in the transfer function file (transfer_function_cosmology.txt), a bash script to run pkdgrav3 and then do postprocessing (pkdgrav3_and_post_process.sh), and a SLURM job script (cuda_job_script_wilkes). |



## Transfer functions

pkdgrav3 requires an input file containing a transfer function (to specify the initial power spectrum). Such files can be created using the Python module [nbodykit](https://nbodykit.readthedocs.io/en/latest/index.html).

The script 'utility.py' has a function called 'make_specific_cosmology_transfer_function' that may be used to create transfer function files.

Transfer function files are in the same directory as the control file for that run.

Updated info:
- For runs C and E (also known as runs 1 and 2) we used a transfer function file. You will find that in each run directory (e.g. .runsC/run001) there will be a file 'transfer_function.txt' that stores the transfer function information and a file 'transfer_function_cosmology.txt' that stores the cosmology used to create the transfer function. The transfer function file was created using the routine 'make_specific_cosmology_transfer_function' in ./scripts/utility.py. The first column in the transfer function file has the wavenumber in units of h/Mpc.
- For the remaining runs (I through S i.e. 3 through 13) we instead used a 'Concept' HDF5 file, which Niall produced (such files are needed to allow pkdgrav3 to better understand neutrinos - NJ: is this correct?). In each run directory (e.g. ./runsR/run012) you will find a file with a name such as 'class_processed_batch12_012.hdf5' (the name will vary from run to run - you can find the correct name for each run by checking the item achClassFilename in control.par). Niall will have more info about how these files are created. Alas I don't have any information about the structure of these files, or how to use the information contained therein.


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
- `source ./set_environment_splinter_nbodycode.sh`


## Storage of output files
Output files are stored:
- on splinter, at /share/testde/ucapwhi/GowerStreetSims/. Recall that when manipulating files on testde, it is best to ssh to nas-0-1 (e.g. ssh ucapwhi@nas-0-1). This has been made public, with webaddress http://www.star.ucl.ac.uk/GowerStreetSims/.
- on NERSC, at /global/cfs/cdirs/des/dirac_sims/original_files.


## Google document with summary of which runs have been done
Google document [here](https://docs.google.com/spreadsheets/d/1lfNehrNxl7ggto7P6gNRRfGNA4YKRRgE8VzDfqTWOqQ/edit#gid=0).


