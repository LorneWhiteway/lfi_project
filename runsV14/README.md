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
| 005 | As runsV/run014, but with high-resolution inpout CONCEPT file. Output simulation box at steps 63 and 100. |
| 006 | As runsV/run014, with input files from _before_ changes to the 'Python code that creates input files' to optionally run without CONCEPT. Never actually run; used simply to allow comparison of input files. |
| 007 | As runsV/run014, with input files from _after_ changes to the 'Python code that creates input files' to optionally run without CONCEPT. |
| 008 | As runsV/run014, with input files that do not require CONCEPT. Output simulation box at steps 63 and 100.|
| 009 | As run008, but with changes to the jobfile: partition changed from 'gpu-a100-80' to 'gpu' and 'cpus-per-task' changed from 48 to 32. These changes will allow running on a wider range of hardware. |


