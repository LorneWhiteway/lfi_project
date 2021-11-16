This directory contains output lightcone files from a high-resolution run of PKDGRAV3.

For more information, contact Lorne Whiteway on lorne.whiteway@star.ucl.ac.uk.

Basic settings: 1024^3 particles, nside = 4096, box = (1070 Mpc/h)^3, 800 time slices.

Lightcone files begin at step 237 (before that the lightcone was too far away to fit into the 'superbox' i.e. the box formed from 6^3 repeats of the basic simulation box). Lightcones are saved in numpy format (so load the files using numpy.load).

Redshift ranges for each lightcone are as specified in z_values.txt.

Code used was a modified version of PKDGRAV3 - https://github.com/LorneWhiteway/lfi_project/tree/master/pkdgrav3 SHA1=854d879b7.

Control parameters for PKDGRAV3 are as given in control.par.

Transfer function as specified in specific_cosmology_01_transfer_function.dat. This was generated using nbodykit.cosmology.power.transfers.CLASS (see https://nbodykit.readthedocs.io/en/latest/api/_autosummary/nbodykit.cosmology.power.transfers.html) and using the following settings:
======= Parameters for specific_cosmology_01 =======
output = vTk dTk mPk
extra metric transfer functions = y
N_ur = 2.0328
gauge = synchronous
n_s = 0.9649
k_pivot = 0.05
tau_reio = 0.066
T_cmb = 2.7255
Omega_k = 0.0
N_ncdm = 1
P_k_max_h/Mpc = 10.0
z_max_pk = 100.0
h = 0.6736
Omega_cdm = 0.2107
Omega_b = 0.0493
A_s = 3.0466874250500465e-09
m_ncdm = [0.06]
sigma8 = 0.8400000000000004
======= End of parameters for specific_cosmology_01 =======

