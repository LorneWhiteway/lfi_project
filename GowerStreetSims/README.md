# Gower Street Sims

This is the public directory containing the Gower Street Simulations, a suite of approximately 800 n-body cosmological simulations created using Pkdgrav3 (https://bitbucket.org/dpotter/pkdgrav3/src). The simulations are described in section II.B.1 of https://arxiv.org/abs/2310.17557; a more detailed paper describing the simulations is in preparation.

We do not provide the full four-dimensional simulation output (i.e. 3D positions of particles for a series of times slices). Rather, we provide the 'lightcone' data; this shows where on the sky the central observer would currently see each simulation object, which this information binned into redshift ranges i.e. tomographic bins.

The simulations are stored in several directories, with names such as 'runsC'. Within each directory there are several TAR archive files, with names such as 'run111.tar.gz'. Each such file contains one simulation.

Detailed information about the simulations is available at https://docs.google.com/spreadsheets/d/1lfNehrNxl7ggto7P6gNRRfGNA4YKRRgE8VzDfqTWOqQ/edit#gid=0. Sheet 1 of this workbook describes the directories, while sheet 2 describes the individual simulations (including information about the cosmological parameters used for each simulation).

Pkdgrav3 creates lightcone snapshots by extracting objects from thin shells of the appropriate radius within a 'superbox' consisting of 218 repeats of the simulation box in a 6x6x6 array (and we as observers are at the centre of the superbox). At early times some or all of the lightcone shell lies completely outside this superbox and in this case no lightcone file is saved. For runsI and later runs, the superbox used 8000 repeats of the simulation box in a 20x20x20 array.

Each TAR archive file contains approximately 100 files, as follows:
a) Lightcone files. These have names of the form run.XXXXX.lightcone.npy where XXXXX is a serial number (left padded with zeroes to have five digits) e.g. 'run.00068.lightcone.npy'. Each lightcone file refers to a particular redshift range i.e. tomographic bin, and contains a pixelised description of the simulation objects visible within that redshift range. The larger the number, the closer the bin to the observer. Each lightcone file is in numpy format i.e. maybe opened by the numpy command 'load' (https://numpy.org/doc/stable/reference/generated/numpy.load.html#numpy.load) in Python. Each file contains a single Healpix array with NSIDE=2048, in RING format; the contents of the array are, per pixel, the number of objects visible in that pixel for that file's redshift range.
b) Incomplete lightcone files, which have names such as 'run.00027.incomplete.npy'. These are lightcone files describe the nearest redshift range that is not completely enclosed in the simulation super-box; such files are incomplete and should not be used.
c) z_values.txt. This is a text file describing the redshift ranges for the tomographic slices. The file has one header row; refer to this for details of the columns in the file. Each slice is described by the redshift, and by the comoving distance (calculated using a cosmology with the same matter density as that used in the simulation, and quoted both in Mpc/h and in box-length units), for the far endpoint and for the near endpoint of the slice; the width and volume of the slice (in the same terms) are also given.
d) control.par. This is the control file (i.e. the ini file) for Pkdgrav3 for this run.
e) output.txt. This is a text file that captured the output that pkdgrav3 wrote to the screen while it was running.
f) run.log. This is the log file that Pkdgrav3 created while it was running. Refer to the Pkdgrav3 documentation for the structure of this file.
g) All other files are for internal use; their contents should not be relied on.


The Gower Street Simulations are named after the London street on which University College London is located. More information about this street here: https://en.wikipedia.org/wiki/Gower_Street,_London.

