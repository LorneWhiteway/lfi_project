This directory contains utilities for interacting with `nbodykit`.

## Introduction

See https://nbodykit.readthedocs.io/en/latest/index.html for info on nbodykit.

Python code that interacts with nbodykit is located in `nbody_utility.py`. Currently there is a function called `make_specific_cosmology_transfer_function_caller()` that can be used to build a transfer function for a specified cosmology.


## nbodykit on splinter

### Installing nbodykit on splinter

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

### Using nbodykit on splinter

To use nbodykit, do:
```
bash
source ./set_environment.sh
```

## nbodykit on Wilkes3

### Installing nbodykit on Wilkes3

See the description in the section on Python in the top-level README.

### Using nbodykit on Wilkes3

- Go to the project directory
- `module load python/3.8`
- `source env/bin/activate`


## Example code

If nbodykit is installed properly then the following Python3 code should work:
```
from nbodykit.lab import *
k = 0.00001
t = cosmology.power.transfers.CLASS(cosmology.Planck15,0.0)(k)
print(t) # Should be 1.0
```

