This directory contains utilities for interacting with `nbodykit`.

See https://nbodykit.readthedocs.io/en/latest/index.html for info on nbodykit.

Installation used the instructions at https://nbodykit.readthedocs.io/en/latest/getting-started/install.html#installing-nbodykit-with-anaconda, together with information from the splinter user manual.

To install on splinter, I did this:
```
bash
eval "$(/share/apps/anaconda/3-2019.07/bin/conda shell.bash hook)"
conda create --prefix /share/splinter/ucapwhi/lfi_project/nbodykit python=3.7 pip
conda install -c bccp nbodykit
conda install matplotlib
```

To use, do this:
```
bash
source ./set_environment.sh
```

You should then be able to e.g.
```
from nbodykit.lab import *
```
in Python.

Note that /home/ucapwhi/.conda is a symbolic link to /share/splinter/ucapwhi/.conda - this is to avoid the 2 Gb disk quota in /home/ucapwhi.

Example:
```
from nbodykit.lab import *
k = 0.00001
t = cosmology.power.transfers.CLASS(cosmology.Planck15,0.0)(k)
print(t) # Should be 1.0
```

Python code that interacts with nbodykit is located in `nbody_utility.py`. Currently there is a function called `make_specific_cosmology_transfer_function_caller()` that can be used to build a transfer function for a specified cosmology.
