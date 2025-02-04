# V14 Test Runs

Directory V14 has been used for several test runs. Most of these runs are variants of (the somewhat randomly chosen) runsV/run014.

The base cosmological parameters were (see also params_run_gs2_V14_3.txt):

| Parameter | Value |
| --- | --- |
| Omega_m | 0.3  |
| sigma_8 | 0.8 |
| w | -0.9 |
| Omega_b | 0.050677810676866969 |
| little_h | 0.673057851239669436 |
| n_s | 0.926574394463667850 |
| m_nu | 0.065418397586654659 |

Here is a description of the various runs:

| Run | Comment |
| --- | --- |
| 001 | As runsV/run014, but with num_particles = 1500^3 instead of 1350^3. Output simulation box at steps 63 and 100.|
| 002 | As runsV/run014, but with num_particles = 1500^3 instead of 1350^3, and 125 time steps instead of 100. Output simulation box at steps 78 and 125. |
| 003 | As runsV/run014, but with 150 time steps instead of 100. Output simulation box at steps 94 and 150. |
| 004 | As runsV/run014. Output simulation box at steps 63 and 100. |
| 005 | As runsV/run014, but with high-resolution input CONCEPT file. Output simulation box at steps 63 and 100. |
| 006 | As runsV/run014, with input files from _before_ changes to the 'Python code that creates input files' to optionally run without CONCEPT. Never actually run; used simply to allow comparison of input files. |
| 007 | As runsV/run014, with input files from _after_ changes to the 'Python code that creates input files' to optionally run without CONCEPT. |
| 008 | As runsV/run014, with input files that do not require CONCEPT. Output simulation box at steps 63 and 100. Random seed changed from 1920 to 2426. |
| 009 | As run008, but with changes to the jobfile: partition changed from 'gpu-a100-80' to 'gpu' and 'cpus-per-task' changed from 48 to 32. These changes will allow running on a wider range of hardware. Run was not successful (out of time). |
| 010 | As run004, but with changes to the control file: 'achLinSpecies    = "g+ncdm[0]+fld+metric"' changed to 'achLinSpecies    = "g+ncdm[0]+fld". |
| 011 | As run008, but with the random seed changed back to 1920 (same as for runsV/run014). |
| 012 | As run010, but with the HDF5 file amended so that delta_metric has been scaled by 1e-4. |
| 013 | As run004, but with special HDF5 file (dakin_test2), A_s=2.1e-9 (so sigma_8 in params file is incorrect) and n_s = 0.96. |
| 014 | As run004, but with special HDF5 file (flagship_test2), A_s=2.1e-9 (so sigma_8 in params file is incorrect) and n_s = 0.96. |
| 015 | As run004, but with special HDF5 file (dakin_test3), A_s=2.1e-9 (so sigma_8 in params file is incorrect) and n_s = 0.96. |
| 016 | As run004, but with special HDF5 file (flagship_test3), A_s=2.1e-9 (so sigma_8 in params file is incorrect) and n_s = 0.96. Output simulation boxes at steps 31, 63 and 100 (corresponding approximately to z = 1.5, 0.5 and 0. |
| 017 | As run016, but non-CONCEPT. Used m_nu=0.0595968, h=0.67, Omega_m=0.319 , omega_b=0.049, A_s=2.1e-9 (obtained using sigma8=0.8134341419869286), n_s = 0.96, and w=-1. |
| 018 | As run004, but with the simulation box output only for step 1. |
