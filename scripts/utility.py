#!/usr/bin/env python

""" 
    Routines for analyzing PKDGRAV3 output.
    Author: Lorne Whiteway.
"""

import glob
import numpy as np
import matplotlib
matplotlib.use('Agg') # See http://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined
import matplotlib.pyplot as plt
import healpy as hp
from astropy.cosmology import FlatLambdaCDM
import datetime
import os
import contextlib
import sys

# ======================== Start of code for reading control file ========================


# Returns a string; caller is then responsible for casting this to the desired type.
def get_from_control_file(control_file_name, key):
    with open(control_file_name, "r") as f:
        for line in f:
            if key == line.split("#")[0].split("=")[0].strip():
                return line.split("#")[0].split("=")[1].strip().strip('\"')
            
    raise SystemError("Could not find key {} in ini file {}".format(key, control_file_name))



def get_from_control_file_test_harness():
    control_file_name = "/rds/user/dc-whit2/rds-dirac-dp153/lfi_project/experiments/fast/control.par"
    key = "dBoxSize"
    print(float(get_from_control_file(control_file_name, key)))
    
# ======================== End of code for reading control file ========================





# ======================== Start of code for reading partial lightcone files ========================


# basefilename will be something like "/share/splinter/ucapwhi/lfi_project/experiments/simple/run.00001.hpb"
# The actual files to be read will then have names like "/share/splinter/ucapwhi/lfi_project/experiments/simple/run.00001.hpb.0"
def one_healpix_map_from_basefilename(basefilename, nside):
    
    hpb_type = np.dtype([('grouped', '=i4'),('ungrouped', '=i4'), ('potential', '=f4')])
    
    array_list = []
    
    # A sorted list of integers appearing as suffixes to basefilename
    suffixes = sorted([int(f.split(".")[-1]) for f in glob.glob(basefilename + ".*")])
    
    for file in [basefilename + "." + str(i) for i in suffixes]:
        with open(file,'rb') as hpb:
            data = np.fromfile(hpb,dtype=hpb_type)
        array_list.append(data['grouped'] + data['ungrouped'])
        
    healpix_map = np.concatenate(array_list)
    
    
    n_pixels = hp.nside2npix(nside)
    assert len(healpix_map) >= n_pixels, "{} yields {} elements, which is too few for nside{}".format(basefilename, len(healpix_map), nside)
    # Has padding at end; ensure that this padding is empty
    assert np.all(healpix_map[n_pixels:] == 0), "{} yields non-zero elements in the padding area".format(basefilename)
    
    return healpix_map[:n_pixels]
    
    

# filespec will be something like "/share/splinter/ucapwhi/lfi_project/experiments/simple/run.*.hpb"
# Output will be a list like ["/share/splinter/ucapwhi/lfi_project/experiments/simple/run.00001.hpb", etc.]
def basefilename_list_from_filespec(filespec):
    return [f[:-2] for f in sorted(glob.glob(filespec + ".0"))]


# Example filespec: "/share/splinter/ucapwhi/lfi_project/experiments/gpu_1000_1024_1000/run.*.hpb"
def save_all_lightcone_files(filespec, nside, delete_hpb_files_when_done):

    # Somewhat unfortunately, pkdgrav3 appears to create the first (i.e most distant) lightcone for the furthest tomographic
    # bin for which the _near_ boundary is less than or equal to 3*boxlength. But for this bin the _far_ boundary is more than
    # 3*boxlength, so this lightcone will be partially outside the 3*boxlength box. It will therefore be incomplete, and must
    # not be used. The variable have_already_encountered_first_populated_lightcone is part of the logic to ensure this.
    have_already_encountered_first_populated_lightcone = False
    
    
    # basefilename_list_from_filespec returns a sorted list, so we are certainly stepping through the files in the correct order.
    for b in basefilename_list_from_filespec(filespec):
    
        # b will be something like '.../run.00001.hpb'
        
        map_t = one_healpix_map_from_basefilename(b, nside)
        
        # Here is where you could change the NSIDE, if desired.
        new_nside = nside
        if new_nside != nside:
            map_t = hp.ud_grade(map_t, new_nside)
        
        output_file_name = b.replace(".hpb", ".lightcone.npy")
        max_pixel_value = np.max(map_t)
        if max_pixel_value > 0:
            if have_already_encountered_first_populated_lightcone:
                print("Writing file {}...".format(output_file_name))
                # We write in uint16 format if possible so as to get smaller files.
                np.save(output_file_name, map_t.astype(np.uint16) if (max_pixel_value < 65535) else map_t)
            else:
                print("Not writing file {} as it would be incomplete.".format(output_file_name))
                have_already_encountered_first_populated_lightcone = True
        else:
            print("Not writing file {} as it would have no objects.".format(output_file_name))
            
        if delete_hpb_files_when_done:
            filespec_to_delete = b + ".*"
            print("Deleting files matching{}...".format(filespec_to_delete))
            for f in glob.glob(filespec_to_delete):
                with contextlib.suppress(FileNotFoundError):
                    os.remove(f)
            # For tidiness also delete the simulation box file (with name such as '.../run.00001') - 
            # provided the file has size 32 bytes (i.e. is a stub file).
            simulation_box_file_name = b[:-4]
            with contextlib.suppress(FileNotFoundError):
                if os.stat(simulation_box_file_name).st_size == 32:
                    print("Deleting {}".format(simulation_box_file_name))
                    os.remove(simulation_box_file_name)
                
                    
    
def plot_lightcone_files(list_of_npy_filenames, do_show = True, do_save = False, mollview_format = None):

    if mollview_format is None:
        mollview_format = False
    
    for filename in list_of_npy_filenames:
        
        map_t = np.load(filename)
        num_objects = np.sum(map_t)
        
        # Histogram of pixel values
        if False:
            plt.hist(map_t, bins=np.max(map_t)+1)
            plt.yscale('log', nonposy='clip') # See https://stackoverflow.com/questions/17952279/logarithmic-y-axis-bins-in-python
            plt.show()
        
        if True:
            map_t = np.log10(map_t) - np.log10(np.mean(map_t)) # log10(1+\delta)
        
        cmap=plt.get_cmap('magma')
        
        
        if mollview_format:
            hp.mollview(map_t, title=filename.replace("/share/splinter/ucapwhi/lfi_project/experiments/", ""), cbar=True, cmap=cmap)
            #hp.graticule(dpar=30.0)
            save_file_extension = "mollview.png"
        else:
            rot = (50.0, -40.0, 0.0)
            hp.orthview(map_t, rot=rot, title=filename.replace("/share/splinter/ucapwhi/lfi_project/experiments/", ""), cbar=True, cmap=cmap)
            save_file_extension = "orthview.png"

        if do_save:
            save_file_name = filename.replace("npy", save_file_extension)
            print("From {} created image file {} with {} objects".format(filename, save_file_name, num_objects))    
            plt.savefig(save_file_name)

        
    if do_show:
        plt.show()
        
    
    
def save_all_lightcone_image_files(directory, mollview_format):
    file_list = glob.glob(os.path.join(directory, "*.npy"))
    file_list.sort()
    plot_lightcone_files(file_list, False, True, mollview_format)
    


# ======================== End of code for reading lightcone files ========================







# ======================== Start of code for reading boxes ========================

def get_header_type():

    return np.dtype([('time','>f8'),('N','>i4'),('Dims','>i4'),('Ngas','>i4'),('Ndark', '>i4'),('Nstar','>i4'),('pad','>i4')])

def get_dark_type():

    return np.dtype([('mass','>f4'),('x','>f4'),('y','>f4'),('z','>f4'),('vx','>f4'),('vy','>f4'),('vz','>f4'),('eps','>f4'),('phi','>f4')])


# file_name will be something like "/share/splinter/ucapwhi/lfi_project/experiments/simple/run.00100"
# This is an edited version the code in the readtipsy.py file distributed with PKDGRAV3.
# Returns an array with elements of type get_dark_type()
def read_one_box(file_name):

    

    with open(file_name, 'rb') as in_file:

        header = np.fromfile(in_file, dtype=get_header_type(), count=1)
        header = dict(zip(get_header_type().names, header[0]))
        print(header)
        
        return np.fromfile(in_file, dtype=get_dark_type(), count=header['Ndark']) # Just dark matter (not gas or stars)
    
    


def read_one_box_example():
    file_name = "/rds/user/dc-whit2/rds-dirac-dp153/lfi_project/experiments/fast/run.00071"
    print("Reading file {}...".format(file_name))
    print(datetime.datetime.now().time())
    d = read_one_box(file_name)
    print("Finished reading file.")
    print(datetime.datetime.now().time())
    
    d_x = d['x']
    d_y = d['y']
    d_z = d['z']
    
    
    fig, ax = plt.subplots()
    
    if False:
        p = 0.0
        slice_thickness = 1e-5
        print("using p = {}".format(p))
        
        z_filter = np.where(np.logical_and(d_z > p-slice_thickness, d_z <= p))
        plt.scatter(d_x[z_filter], d_y[z_filter], s=1)
        
    else:
        h = ax.hist2d(d_x, d_z, 50)
        fig.colorbar(h[3], ax=ax)

    ax.set_aspect('equal')
    plt.savefig("/rds/user/dc-whit2/rds-dirac-dp153/lfi_project/experiments/fast/foo.png")
    #print(d_x.shape[0])
    #print(d_x[z_filter].shape[0])
    
    

# File_name will be something like "/share/splinter/ucapwhi/lfi_project/experiments/simple/run.00100"
def show_one_shell(file_name, shell_low, shell_high, nside):

    d = read_one_box(file_name)
    ra_list = []
    dec_list = []
    offsets = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5] # See my notes p. LFI24
    index_dict = {}
    for offset_x in offsets:
        d_x = d['x']-offset_x
        d_x_2 = d_x**2
        for offset_y in offsets:
            d_y = d['y']-offset_y
            d_y_2 = d_y**2
            for offset_z in offsets:
                d_z = d['z']-offset_z
                d_z_2 = d_z**2
                dist = np.sqrt(d_x_2 + d_y_2 + d_z_2)
                for (i, dd, xx, yy, zz) in zip(range(len(d_x)), dist, d_x, d_y, d_z):
                
                    if shell_low < dd and dd < shell_high:
                        ra_list.append(np.degrees(np.arctan2(yy, xx)))
                        dec_list.append(np.degrees(np.arcsin(zz/dd)))
                        # Also register i so that we can see if it gets reused
                        if i not in index_dict:
                            index_dict[i] = 0
                        index_dict[i] += 1
    ra = np.array(ra_list)
    dec = np.array(dec_list)
    
    num_healpixels = hp.nside2npix(nside)
    map = np.zeros(num_healpixels)
    ids = hp.ang2pix(nside, ra, dec, False, lonlat=True)
    for id in ids:
        map[id] += 1.0
    hp.mollview(map, title="", xsize=400, badcolor="grey")
    hp.graticule(dpar=30.0)
    plt.show()
    
    print(index_dict)

    
def show_one_shell_example():

    file_name = "/share/splinter/ucapwhi/lfi_project/experiments/simple/run.00099"
    shell_low = 0.1137
    shell_high = 0.228
    nside = 128
    show_one_shell(file_name, shell_low, shell_high, nside)
    
    
def match_points_between_boxes():

    file_names = ["/share/splinter/ucapwhi/lfi_project/experiments/simple/run.00001", "/share/splinter/ucapwhi/lfi_project/experiments/simple/run.00100"]

    d0 = read_one_box(file_names[0])
    d1 = read_one_box(file_names[1])
	
    
    #dist = np.sqrt((d0['x']-d1['x'])**2 + (d0['y']-d1['y'])**2 + (d0['z']-d1['z'])**2)
    
    #print(np.mean(dist))
    
    #for i in range(len(d0['x'])):
    #    print(d0['x'][i], d0['y'][i], d0['y'][i], d1['x'][i], d1['y'][i], d1['y'][i], (d0['x'][i] - d1['x'][i]), (d0['y'][i] - d1['y'][i]), (d0['z'][i] - d1['z'][i]))
    
    #for s in ['x', 'y', 'z']:
    #    data = d0[s] - d1[s]
    #    print(np.mean(data), np.std(data))
        
    #plt.plot(d0['x'])
    #plt.plot(d0['y'])
    #plt.plot(d0['z'])
    #plt.show()
    
    plt.scatter(d0['x'], d0['y'], s=1, c='red')
    plt.scatter(d1['x'], d1['y'], s=1, c='blue')
    plt.show()
    
    plt.scatter(d0['x'], d0['z'], s=1, c='red')
    plt.scatter(d1['x'], d1['z'], s=1, c='blue')
    plt.show()
    
    
# ======================== End of code for reading boxes ========================








# ======================== Start of code for reading PKDGRAV3 output ========================

# The file to be read by this routine can be created by running PKDGRAV3 and piping the output to file.
def read_one_output_file(filename):

    step_list = []
    time_list = []
    z_list = []
    
    with open(filename, 'r') as infile:
        for el in infile:
            step_string = "Writing output for step "
            if step_string in el:
                step = int(el.split(step_string)[1])
                step_list.append(step)
            if "Time:" in el and "Redshift:" in el and not "Expansion factor" in el:
                t = float(el.split()[0].split(":")[1]) # Time
                time_list.append(t)
                z = float(el.split()[1].split(":")[1]) # redshift
                z_list.append(z)
                
    return (np.array(step_list), np.array(time_list), np.array(z_list))
            
    
def build_z_values_file(directory):

    input_filename = directory + "/output.txt"
    control_file_name = directory + "/control.par"
    output_filename = directory + "/z_values.txt"
    
    print("Writing data to {}...".format(output_filename))
    
    
    (s_arr, t_arr, z_arr) = read_one_output_file(input_filename)
    
    Om0 = float(get_from_control_file(control_file_name, "dOmega0"))
    cosmo = FlatLambdaCDM(H0=100.0, Om0=Om0)
    
    box_size = float(get_from_control_file(control_file_name, "dBoxSize"))

    # Prepand step 0 values
    starting_z = float(get_from_control_file(control_file_name, "dRedFrom"))
    s_arr = np.concatenate(([0], s_arr))
    z_arr = np.concatenate(([starting_z], z_arr))
    
    cmd_arr = cosmo.comoving_distance(z_arr).value # In Mpc/h
    cmd_over_box_arr = cmd_arr / box_size # Unitless
    
    
    header = "Step,z_far,z_near,delta_z,cmd_far(Mpc/h),cmd_near(Mpc/h),delta_cmd(Mpc/h),cmd/box_far,cmd/box_near,delta_cmd/box"
    
    
    np.savetxt(output_filename, np.column_stack((s_arr[1:], z_arr[:-1], z_arr[1:], (z_arr[:-1]-z_arr[1:]), cmd_arr[:-1], cmd_arr[1:], (cmd_arr[:-1]-cmd_arr[1:]), cmd_over_box_arr[:-1], cmd_over_box_arr[1:], (cmd_over_box_arr[:-1]-cmd_over_box_arr[1:]))), fmt=["%i", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f"], delimiter=",", header=header)
    


    

def file_spec_has_files(file_spec):
    return bool(glob.glob(file_spec))


# Run this after PKDGRAV3 has finished to do all the postprocessing.
# do_delete means that the raw, partial lightcone files (output by PKDGRAV) will be deleted.
# If do_delete is set then do_lightcone_files must be set as well (assertion failure if not) - this prevents this program from deleting
# raw, partial lightcone files that have not yet been processed into full lightcone files.
def pkdgrav3_postprocess(directory, do_lightcone_files, do_delete, do_mollview_images, do_ortho_images, do_z_file, do_status, do_force):

    assert(do_lightcone_files or not do_delete)

    control_file_name = os.path.join(directory, "control.par")
    
    nside = int(get_from_control_file(control_file_name, "nSideHealpix"))
    
    print("Processing {}".format(directory))
    
    outName = get_from_control_file(control_file_name, "achOutName") # Standard value is 'run'.
    
    
    if do_lightcone_files and (do_force or not file_spec_has_files(os.path.join(directory, outName + ".*.lightcone.npy"))):
        save_all_lightcone_files(os.path.join(directory, outName + ".*.hpb"), nside, do_delete)
        
    if do_mollview_images and (do_force or not file_spec_has_files(os.path.join(directory, outName + ".*.lightcone.mollview.png"))):
        save_all_lightcone_image_files(directory, True)
    
    if do_ortho_images and (do_force or not file_spec_has_files(os.path.join(directory, outName + ".*.lightcone.orthview.png"))):
        save_all_lightcone_image_files(directory, False)
    
    if do_z_file and (do_force or not file_spec_has_files(os.path.join(directory, "z_values.txt"))):
        build_z_values_file(directory)
        
    if do_status:
        status(directory)
        
    
    
# ======================== End of code for reading PKDGRAV3 output ========================    


# ======================== Start of other utilities ========================    

### Good code, but no longer needed.
#import cosmic_web_utilities as cwu 
#from numpy import random
#def intersection_of_shell_and_cells():
#
#    radius = 2.418
#    nside = 128
#    npix = hp.nside2npix(nside)
#    
#    (ra, dec) = hp.pix2ang(nside, np.arange(npix), False, lonlat=True)
#    (x, y, z) = cwu.spherical_to_cartesian(ra, dec, distance = radius)
#    
#    
#    to_be_sorted = np.column_stack([np.random.uniform(size=216),np.arange(216)])
#    random_indices = to_be_sorted[to_be_sorted[:,0].argsort()].astype(int)[:,1]
#    
#    
#        
#    if True:
#        # Box number
#        hp_map = random_indices[(np.floor(x).astype(int) + 3) * 36 + (np.floor(y).astype(int) + 3) * 6 + (np.floor(z).astype(int) + 3)]
#    else:
#        # Shell number
#        hp_map = np.floor(np.maximum.reduce([np.abs(x), np.abs(y), np.abs(z)]))
#    
#    rot = (50.0, -40.0, 0.0)
#    cmap=plt.get_cmap('magma')
#    if False:
#        hp.mollview(hp_map, cmap=cmap)
#    else:
#        hp.orthview(hp_map, rot=rot, cmap=cmap)
#    
#    plt.show()
    
    
def compare_two_lightcones_by_power_spectra():
    file_name_array = ["/share/splinter/ucapwhi/lfi_project/experiments/v100_1024_4096_900/run.00067.lightcone.npy", "/share/splinter/ucapwhi/lfi_project/experiments/v100_1024_4096_900/run.00130.lightcone.npy"]
    
    delta_array = [(m/np.mean(m))-1.0 for m in [np.load(f) for f in file_name_array]]

    lmax = 1000
    
    af = [hp.anafast(d, lmax=lmax, pol=False) for d in delta_array]
    
    if False:
        plt.plot(af[0][1:])
        plt.plot(af[1][1:])
        plt.yscale("log")
    else:
        plt.plot(af[1][1:]/af[0][1:])
    
    
    
    plt.savefig("/share/splinter/ucapwhi/lfi_project/scripts/foo.png")
    
    
def compare_two_time_spacings():

    file_1 = "/share/splinter/ucapwhi/lfi_project/experiments/v100_freqtimeslicing/z_values.txt"
    file_2 = "/share/splinter/ucapwhi/lfi_project/mice/steps.N4096.L3072.zrange.dat"
    
    d_1 = np.loadtxt(file_1, skiprows=1, delimiter=",")
    z_avg_1 = (d_1[:,1] + d_1[:,2]) * 0.5
    delta_z_1 = d_1[:,3]
    
    d_2 = np.loadtxt(file_2, skiprows=1, delimiter=",")
    z_avg_2 = (d_2[:,1] + d_2[:,2]) * 0.5
    delta_z_2 = d_2[:,2] - d_2[:,1]
    
    plt.yscale("log")
    
    plt.scatter(z_avg_1, delta_z_1, c="b", s=1, label="PKDGRAV3")
    plt.scatter(z_avg_2, delta_z_2, c="r", s=1, label="MICE")
    plt.scatter(z_avg_2, delta_z_2*0.5, c="g", s=1, label="MICE*0.5")
    
    
    plt.legend(loc="upper left")
    
    plt.xlabel('z')
    plt.ylabel('$\Delta z$')
    
    plt.xlim(0.0, 10.0)
    
    
    if False:
        plt.show()
    else:
        num_steps = int(get_from_control_file(file_1.replace("z_values.txt", "control.par"), "nSteps"))
        save_file_name = "/share/splinter/ucapwhi/lfi_project/experiments/v100_freqtimeslicing/z_values_comp_{}.png".format(num_steps)
        print("Saving {}...".format(save_file_name))    
        plt.savefig(save_file_name)


def create_dummy_output_file():
    # create dmmy output files to reserve disk space
    directory = "/share/splinter/ucapwhi/lfi_project/experiments/gpu_probtest"
    control_file_name = directory + "/control.par"
    nside = int(get_from_control_file(control_file_name, "nSideHealpix"))
    num_steps = int(get_from_control_file(control_file_name, "nSteps"))
    
    n_pixels = hp.nside2npix(nside)
    empty_map = np.zeros(n_pixels)
    
    print("Writing {} dummy output files each with {} pixels".format(num_steps, n_pixels))
    
    for i in range(num_steps):
        output_file_name = directory + "/dummy.{}.npy".format(str(i+1).zfill(5))
        print("Writing {}".format(output_file_name))
        hp.write_map(output_file_name, empty_map, overwrite=True)
    

# ======================== End of other utilities ========================    


# ======================== Start of code for reporting on the status of directory containing pkdgrav3 output ========================

# Helper functions

# Returns True iff the file exists.
def report_whether_file_exists(file_description, file_name):
    file_exists = os.path.isfile(file_name)
    print("File {} {} '{}'".format(("exists:" if file_exists else "DOES NOT exist: no"), file_description, file_name))
    return file_exists
    
# Returns True iff files exist.
def report_whether_several_files_exist(file_description, filespec):
    num_files = len(glob.glob(filespec))
    if num_files > 0:
        print("Files exist: {} {} file{}".format(num_files, file_description, plural_suffix(num_files)))
        return True
    else:
        print("Files DO NOT exist: no {} files".format(file_description))
        return False

def plural_suffix(count):
    return ("" if count==1 else "s")
    
def npy_file_data_type(file_name):
    d = np.load(file_name)
    return d.dtype.name
    

def status(directory):

    # Deal recursively with subdirectories
    dir_name_list = glob.glob(os.path.join(directory, "*/"))
    dir_name_list.sort()
    for d in dir_name_list:
        status(d)

    # Then deal with this directory
    print("========================================================================")
    print("Status of {} as of {}".format(directory, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    control_file_name = os.path.abspath(os.path.join(directory, "control.par"))
    out_name = ""
    if report_whether_file_exists("control file", control_file_name):
        out_name = get_from_control_file(control_file_name, "achOutName") # Typical value is 'run'.
    report_whether_file_exists("log file", os.path.abspath(os.path.join(directory, out_name + ".log")))
    report_whether_file_exists("output file", os.path.abspath(os.path.join(directory, "output.txt")))
    report_whether_several_files_exist("raw PKDGRAV3 output", os.path.join(directory, "*.hpb*"))
    if report_whether_several_files_exist("full lightcone", os.path.join(directory, "*.npy")):
        print("   Type of full lightcone files is {}".format(npy_file_data_type(glob.glob(os.path.join(directory, "*.npy"))[0])))
    report_whether_several_files_exist("lightcone image (mollview)", os.path.join(directory, "*.lightcone.mollview.png"))
    report_whether_several_files_exist("lightcone image (orthview)", os.path.join(directory, "*.lightcone.orthview.png"))
    report_whether_file_exists("z_values file", os.path.abspath(os.path.join(directory, "z_values.txt")))


# ======================== End of code for reporting on the status of an 'experiments' directory ========================



if __name__ == '__main__':
    
    #show_one_shell_example()
    #match_points_between_boxes()
    #read_one_box_example()
    #intersection_of_shell_and_cells()
    #save_all_lightcone_image_files("/share/splinter/ucapwhi/lfi_project/experiments/k80_1024_4096_900/", True)
    #compare_two_lightcones_by_power_spectra()
    #get_float_from_control_file_test_harness()
    #compare_two_time_spacings()
    #create_dummy_output_file()
    pass
    
    

    