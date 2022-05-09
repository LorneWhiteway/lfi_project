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
from shutil import copyfile
import math
import stat
import time
import pickle


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


# ======================== Start of code for reading log file ========================

# Typical input file name will be 'run.log'. Output is inclusive at both ends
# (i.e. includes both starting redshift and z=0).
def get_array_of_z_values_from_log_file(log_file_name):
    # Return the second column (thanks to NJ for pointing out that the
    # data could be found here...)
    return np.loadtxt(log_file_name, delimiter=" ")[:,1]
    
    
# Typical input file name will be 'run.log'.
def get_parameter_from_log_file(log_file_name, parameter_name):
    with open(log_file_name, "r") as f:
        for line in f:
            if parameter_name in line:
                tokenised_string = line.split(" ")
                for (token, i) in zip(tokenised_string, range(len(tokenised_string))):
                    if token == parameter_name or token == (parameter_name + ":"):
                        if i < len(tokenised_string) - 1:
                            return float(tokenised_string[i + 1])
    raise SystemError("Could not find parameter {} in log file {}".format(parameter_name, log_file_name))
    
    
def get_parameter_from_log_file_test_harness():
    log_file_name = "/rds/user/dc-whit2/rds-dirac-dp153/lfi_project/runsI/run001/run.log"
    parameter_name = "dOmega0"
    print(get_parameter_from_log_file(log_file_name, parameter_name))
    


# ======================== End of code for reading log file ========================


# ======================== Start of code for reading raw lightcone files ========================


# Example lightcone_file_name: "/share/splinter/ucapwhi/lfi_project/experiments/gpu_1000_1024_1000/run.00072.lightcone.npy"
# in which case we return the integer 72
def tomographic_slice_number_from_lightcone_file_name(lightcone_file_name):
    base_name = os.path.basename(lightcone_file_name) # Example: run.00072.lightcone.npy
    return int(base_name[-19:-14])

#def tomographic_slice_number_from_lightcone_file_name_test_harness():
#    lightcone_file_name = "/share/splinter/ucapwhi/lfi_project/experiments/gpu_1000_1024_1000/run.12345.lightcone.npy"
#    print(tomographic_slice_number_from_lightcone_file_name(lightcone_file_name))


# Updates a pickle file that stores a dictionary linking the tomographic slice number and the object count
# From https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file
def update_object_count_file(object_count_file_name, tomographic_slice_number, count):
    d = {}
    if os.path.isfile(object_count_file_name):
        with open(object_count_file_name, 'rb') as f:
            d = pickle.load(f)
    d[tomographic_slice_number] = count
    with open(object_count_file_name, 'wb') as f:
        pickle.dump(d, f)
        


def get_object_count_from_object_count_file(object_count_file_name, tomographic_slice_number):
    d = {}
    if os.path.isfile(object_count_file_name):
        with open(object_count_file_name, 'rb') as f:
            d = pickle.load(f)
        if tomographic_slice_number in d:
            return d[tomographic_slice_number]
            
    return 0
    
def show_object_count_file(object_count_file_name):
    d = {}
    if os.path.isfile(object_count_file_name):
        with open(object_count_file_name, 'rb') as f:
            d = pickle.load(f)
            
    for key in d:
        print("{}: {}".format(key, d[key]))
    
        

#def object_count_file_test_harness():
#    object_count_file_name = '/rds/user/dc-whit2/rds-dirac-dp153/lfi_project/scripts/obj_count.pkl'
#    lightcone_file_name = "/share/splinter/ucapwhi/lfi_project/experiments/gpu_1000_1024_1000/run.12345.lightcone.npy"
#    tomographic_slice_number = tomographic_slice_number_from_lightcone_file_name(lightcone_file_name)
#    count = 34557
#    update_object_count_file(object_count_file_name, tomographic_slice_number, count)
#    print(get_object_count_from_object_count_file(object_count_file_name, 12345))
#    print(get_object_count_from_object_count_file(object_count_file_name, 12346))
#    print(get_object_count_from_object_count_file(object_count_file_name, 72))
#    print(get_object_count_from_object_count_file(object_count_file_name, 0))
#    lightcone_file_name = "/share/splinter/ucapwhi/lfi_project/experiments/gpu_1000_1024_1000/run.12346.lightcone.npy"
#    tomographic_slice_number = tomographic_slice_number_from_lightcone_file_name(lightcone_file_name)
#    count = 98765
#    update_object_count_file(object_count_file_name, tomographic_slice_number, count)
#    print(get_object_count_from_object_count_file(object_count_file_name, 12345))
#    print(get_object_count_from_object_count_file(object_count_file_name, 12346))
#    print(get_object_count_from_object_count_file(object_count_file_name, 72))
#    print(get_object_count_from_object_count_file(object_count_file_name, 0))
    




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
    

def file_is_recent(file_name, cutoff_time_in_seconds):
    return (time.time() - os.path.getmtime(file_name)) < cutoff_time_in_seconds

    

# filespec will be something like "/share/splinter/ucapwhi/lfi_project/experiments/simple/run.*.hpb"
# Output will be a list like ["/share/splinter/ucapwhi/lfi_project/experiments/simple/run.00001.hpb", etc.]
def basefilename_list_from_filespec(filespec):
    return [f[:-2] for f in sorted(glob.glob(filespec + ".0"))]


# Example filespec: "/share/splinter/ucapwhi/lfi_project/experiments/gpu_1000_1024_1000/run.*.hpb"
# Set 'on_the_fly' to True if you are doing post-processing while the pkdgrav run is on-going. In this case:
# a) no processing is done if files for step '97' exist (as in this case the run is almost finished and regular 
# post-processing will do the rest); b) a step is post-processed only if it is more than one hour old.
def save_all_lightcone_files(filespec, nside, delete_hpb_files_when_done, on_the_fly):

    
    # This will be a pickle file storing a dictionary linking the tomographic slice number to the object count.
    object_count_file_name = os.path.join(os.path.dirname(filespec), "object_count.pkl")
    
    if on_the_fly:
        if os.path.isfile(filespec.replace("*", "00097")+".0") or os.path.isfile(filespec.replace("*", "00097").replace(".hpb", ".lightcone.npy")):
            # Don't do anything as normal post-processing will happen soon (or is underway - hence the test for .lightcone.npy)
            # and we don't want to interfere.
            print("save_all_lightcone_files is doing nothing due to presence of 00097 file.")
            return
        
    
    # basefilename_list_from_filespec returns a sorted list, so we are certainly stepping through the files in the correct order.
    for b in basefilename_list_from_filespec(filespec):
    
        # b will be something like '.../run.00001.hpb'
        
        cutoff_time_in_seconds = 3600
        if on_the_fly and file_is_recent(b + '.0', cutoff_time_in_seconds):
            # This slice is too recent (and might still be being written out) - so don't
            # process it. Also we can quit completely as all remaining files will be newer.
            print("save_all_lightcone_files is quitting as files are less than {} seconds old.".format(cutoff_time_in_seconds))
            return
        
        map_t = one_healpix_map_from_basefilename(b, nside)
        
        # Here is where you could change the NSIDE, if desired.
        new_nside = nside
        if new_nside != nside:
            map_t = hp.ud_grade(map_t, new_nside)
        
        # This will be something like "/share/splinter/ucapwhi/lfi_project/experiments/gpu_1000_1024_1000/run.00001.lightcone.npy"
        lightcone_file_name = b.replace(".hpb", ".lightcone.npy")
        
        update_object_count_file(object_count_file_name, tomographic_slice_number_from_lightcone_file_name(lightcone_file_name), np.sum(map_t))
        
        max_pixel_value = np.max(map_t)
        if max_pixel_value > 0:
            # Somewhat unfortunately, pkdgrav3 appears to create the first (i.e most distant) lightcone for the furthest tomographic
            # bin for which the _near_ boundary is less than or equal to 3*boxlength. But for this bin the _far_ boundary is more than
            # 3*boxlength, so this lightcone will be partially outside the 3*boxlength box. It will therefore be incomplete, and must
            # not be used. So test to see if the _previous_ tomographic slice had any objects; if it didn't, then we know that
            # the current file will be incomplete. The object_count_file exists to allow this logic.
            
            if get_object_count_from_object_count_file(object_count_file_name, tomographic_slice_number_from_lightcone_file_name(lightcone_file_name)-1) > 0:
                print("Writing file {}...".format(lightcone_file_name))
                # We write in uint16 format if possible so as to get smaller files.
                np.save(lightcone_file_name, map_t.astype(np.uint16) if (max_pixel_value < 65535) else map_t)
            else:
                # For safety, save the data under a new name.
                lightcone_file_name_amended = lightcone_file_name.replace('lightcone', 'incomplete')
                print("Writing incomplete file {}.".format(lightcone_file_name_amended))
                # We write in uint16 format if possible so as to get smaller files.
                np.save(lightcone_file_name_amended, map_t.astype(np.uint16) if (max_pixel_value < 65535) else map_t)
                
        else:
            print("Not writing file {} as it would have no objects.".format(lightcone_file_name))
            
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
    


# ======================== End of code for reading raw lightcone files ========================







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



    
def build_z_values_file(directory, out_name):

    log_file_name = os.path.join(directory, "{}.log".format(out_name))
    control_file_name = os.path.join(directory,  "control.par")
    output_filename = os.path.join(directory, "z_values.txt")
    
    print("Writing data to {}...".format(output_filename))
    
    z_arr = get_array_of_z_values_from_log_file(log_file_name)
    s_arr = range(z_arr.shape[0])
    
    Om0 = get_parameter_from_log_file(log_file_name, "dOmega0")
    cosmo = FlatLambdaCDM(H0=100.0, Om0=Om0) # Will only be approximately correct.
    
    box_size = float(get_from_control_file(control_file_name, "dBoxSize"))
    
    cmd_arr = cosmo.comoving_distance(z_arr).value # In Mpc/h
    cmd_over_box_arr = cmd_arr / box_size # Unitless
    
    slice_volume = (4.0 / 3.0) * math.pi * (cmd_arr[:-1]**3 - cmd_arr[1:]**3) # In (Mpc/h)^3
    slice_volume_over_box_volume = slice_volume / box_size**3
    
    
    header = "Step,z_far,z_near,delta_z,cmd_far(Mpc/h),cmd_near(Mpc/h),delta_cmd(Mpc/h),cmd/box_far,cmd/box_near,delta_cmd/box,cmvolume,cmvolume/boxvolume"
    
    
    np.savetxt(output_filename, np.column_stack((s_arr[1:], z_arr[:-1], z_arr[1:], (z_arr[:-1]-z_arr[1:]), cmd_arr[:-1], cmd_arr[1:], (cmd_arr[:-1]-cmd_arr[1:]), cmd_over_box_arr[:-1], cmd_over_box_arr[1:], (cmd_over_box_arr[:-1]-cmd_over_box_arr[1:]), slice_volume, slice_volume_over_box_volume)), fmt=["%i", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.2f", "%.6f"], delimiter=",", header=header)
    


    

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
    
    out_name = get_from_control_file(control_file_name, "achOutName") # Standard value is 'run'.
    
    
    if do_lightcone_files and (do_force or not file_spec_has_files(os.path.join(directory, out_name + ".*.lightcone.npy"))):
        save_all_lightcone_files(os.path.join(directory, out_name + ".*.hpb"), nside, do_delete, False)
        
    if do_mollview_images and (do_force or not file_spec_has_files(os.path.join(directory, out_name + ".*.lightcone.mollview.png"))):
        save_all_lightcone_image_files(directory, True)
    
    if do_ortho_images and (do_force or not file_spec_has_files(os.path.join(directory, out_name + ".*.lightcone.orthview.png"))):
        save_all_lightcone_image_files(directory, False)
    
    if do_z_file and (do_force or not file_spec_has_files(os.path.join(directory, "z_values.txt"))):
        build_z_values_file(directory, out_name)
        
    if do_status:
        status(directory)
        


def monitor_core(directory):
    print("Processing {}".format(directory))
    
    control_file_name = os.path.abspath(os.path.join(directory, "control.par"))
    if os.path.isfile(control_file_name):
        
        outName = get_from_control_file(control_file_name, "achOutName") # Standard value is 'run'.
        nside = int(get_from_control_file(control_file_name, "nSideHealpix"))
        do_delete = True
        
        save_all_lightcone_files(os.path.join(directory, outName + ".*.hpb"), nside, do_delete, True)
        sys.stdout.flush()
    
    



def monitor(directory):

    sleep_time_in_seconds = 300
    wait_file_name = os.path.join(directory, "monitor_wait.txt")
    stop_file_name = os.path.join(directory, "monitor_stop.txt")
    
    while os.path.isfile(wait_file_name):
        print("Monitor has encountered wait file {} and will wait {} seconds before checking again".format(wait_file_name, sleep_time_in_seconds))
        sys.stdout.flush()
        time.sleep(sleep_time_in_seconds)
        
    
    while True:
    
        # Deal recursively with subdirectories
        dir_name_list = glob.glob(os.path.join(directory, "*/"))
        dir_name_list.sort()
        for d in dir_name_list:
            monitor_core(d)

        # Then deal with this directory
        monitor_core(directory)
        
            
        print("Time now: {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        print("Monitor will now sleep for {} seconds".format(sleep_time_in_seconds))
        print("When it wakes, it will check for {} and quit if this file exists".format(stop_file_name))
        sys.stdout.flush()
        time.sleep(sleep_time_in_seconds)
        if os.path.isfile(stop_file_name):
            print("Monitor is quitting due to detection of stop file {}".format(stop_file_name))
            sys.stdout.flush()
            return
        
        
    
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


## Good code, but no longer needed
#def create_dummy_output_file():
#    # create dummy output files to reserve disk space
#    directory = "/share/splinter/ucapwhi/lfi_project/experiments/gpu_probtest"
#    control_file_name = directory + "/control.par"
#    nside = int(get_from_control_file(control_file_name, "nSideHealpix"))
#    num_steps = int(get_from_control_file(control_file_name, "nSteps"))
#    
#    n_pixels = hp.nside2npix(nside)
#    empty_map = np.zeros(n_pixels)
#    
#    print("Writing {} dummy output files each with {} pixels".format(num_steps, n_pixels))
#    
#    for i in range(num_steps):
#        output_file_name = directory + "/dummy.{}.npy".format(str(i+1).zfill(5))
#        print("Writing {}".format(output_file_name))
#        hp.write_map(output_file_name, empty_map, overwrite=True)
    

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


# ======================== End of code for reporting on the status of directory containing pkdgrav3 output ========================



# ======================== Start of code for handling transfer functions ========================


def cosmology_summary(cosmo):
    summary = []
    cosmo_dict = dict(cosmo)
    for key in cosmo_dict:
        summary.append("{} = {}".format(key, cosmo_dict[key]))
    summary.append("sigma8 = {}".format(cosmo.sigma8))
    summary.append("Omega0_k = {}".format(cosmo.Omega0_k))
    summary.append("Omega_fld = {}".format(cosmo.Ode0))
    summary.append("Omega0_cb = {}".format(cosmo.Omega0_cb))
    summary.append("Omega0_m = {}".format(cosmo.Omega0_m))
    return summary
    
    
# Omega0_m is the sum of the energy densities for b, cdm and nu.
# ncdm is the total neutrino mass in eV.
def make_specific_cosmology(directory, Omega0_m, sigma8, w, Omega0_b, h, n_s, ncdm, P_k_max):
    from nbodykit.lab import cosmology
    
    # Inferred neutrino energy density
    Omega0_nu = ncdm / (93.14 * h * h)
    # Implied cdm energy density
    Omega0_cdm = Omega0_m - Omega0_b - Omega0_nu
    cosmology_object = (cosmology.Planck15).clone(h=h, Omega0_b=Omega0_b, Omega0_cdm=Omega0_cdm, w0_fld=w, n_s=n_s, m_ncdm=ncdm, P_k_max=P_k_max).match(sigma8=sigma8)
    
    # Check that all this yields the expected Omega0_m:
    assert abs(cosmology_object.Omega0_m/Omega0_m - 1.0) < 2e-6, "Failed to match input Omega0_m when creating cosmology object: target = {}, actual = {}".format(Omega0_m, cosmology_object.Omega0_m)
    
    cosmology_parameters_file_name = os.path.join(directory, "nbodykit_cosmology.txt")
    np.savetxt(cosmology_parameters_file_name, cosmology_summary(cosmology_object), fmt = '%s')
    return cosmology_object



def trim_rows_containing_nan(a):
    # Credit: https://note.nkmk.me/en/python-numpy-nan-remove/
    return a[~np.isnan(a).any(axis=1), :]
    
    
# This set of values was obtained from /data/euclid_z0_transfer_combined.dat    
def set_of_wavenumbers():
    return np.array([ \
    1.05433E-05,1.16423E-05,1.28559E-05,1.41959E-05,1.56756E-05,1.73096E-05,1.91139E-05,2.11063E-05,
    2.33063E-05,2.57357E-05,2.84183E-05,3.13805E-05,3.46515E-05,3.82634E-05,4.22518E-05,4.66560E-05,
    5.15193E-05,5.68894E-05,6.28194E-05,6.93674E-05,7.65980E-05,8.45823E-05,9.33988E-05,1.03134E-04,
    1.13885E-04,1.25756E-04,1.38864E-04,1.53339E-04,1.69322E-04,1.86972E-04,2.06461E-04,2.27981E-04,
    2.51745E-04,2.77986E-04,3.06963E-04,3.38959E-04,3.74291E-04,4.13306E-04,4.56387E-04,5.03959E-04,
    5.56490E-04,6.14497E-04,6.78549E-04,7.49279E-04,8.27381E-04,9.13624E-04,1.00886E-03,1.11402E-03,
    1.23014E-03,1.35836E-03,1.49995E-03,1.65630E-03,1.82895E-03,2.01959E-03,2.23011E-03,2.46256E-03,
    2.71925E-03,3.00270E-03,3.31569E-03,3.66130E-03,4.04294E-03,4.46436E-03,4.92971E-03,5.44357E-03,
    6.01098E-03,6.63755E-03,7.32942E-03,8.09341E-03,8.93704E-03,9.86860E-03,1.08973E-02,1.20332E-02,
    1.32874E-02,1.46725E-02,1.62019E-02,1.78907E-02,1.97556E-02,2.18148E-02,2.40887E-02,2.65996E-02,
    2.93723E-02,3.24339E-02,3.58147E-02,3.95479E-02,4.36702E-02,4.80373E-02,5.24043E-02,5.67713E-02,
    6.11383E-02,6.55053E-02,6.98724E-02,7.42394E-02,7.86064E-02,8.29734E-02,8.73405E-02,9.17075E-02,
    9.71317E-02,1.02556E-01,1.07980E-01,1.13404E-01,1.18828E-01,1.24253E-01,1.29677E-01,1.35101E-01,
    1.40525E-01,1.45949E-01,1.51373E-01,1.56798E-01,1.62222E-01,1.67646E-01,1.73070E-01,1.78494E-01,
    1.83918E-01,1.89343E-01,1.94767E-01,2.00191E-01,2.05615E-01,2.11039E-01,2.16463E-01,2.21888E-01,
    2.27312E-01,2.32736E-01,2.38160E-01,2.43584E-01,2.49008E-01,2.54433E-01,2.59857E-01,2.65281E-01,
    2.70705E-01,2.76129E-01,2.81553E-01,2.86978E-01,2.92402E-01,2.97826E-01,3.03250E-01,3.08674E-01,
    3.14098E-01,3.19523E-01,3.24947E-01,3.30371E-01,3.35795E-01,3.41219E-01,3.46643E-01,3.52068E-01,
    3.57492E-01,3.62916E-01,3.68340E-01,3.73764E-01,3.79189E-01,3.84613E-01,3.90037E-01,3.95461E-01,
    4.00885E-01,4.06309E-01,4.11734E-01,4.17158E-01,4.22582E-01,4.28006E-01,4.33430E-01,4.38854E-01,
    4.44279E-01,4.49703E-01,4.55127E-01,4.60551E-01,4.65975E-01,4.71399E-01,4.76824E-01,4.82248E-01,
    4.87672E-01,4.93096E-01,4.98520E-01,5.03944E-01,5.09369E-01,5.14793E-01,5.20217E-01,5.25641E-01,
    5.31065E-01,5.36489E-01,5.41914E-01,5.47338E-01,5.52762E-01,5.58186E-01,5.63610E-01,5.69034E-01,
    5.74459E-01,5.79883E-01,6.04325E-01,6.40940E-01,6.79774E-01,9.48700E-01,1.32402E+00,1.84782E+00,
    2.57884E+00,3.59905E+00,5.02288E+00,7.01000E+00,9.78324E+00,1.36536E+01,1.90552E+01,2.65936E+01,
    3.71144E+01,5.17973E+01,7.22889E+01,1.00887E+02,1.40800E+02,1.96502E+02,2.74240E+02,3.82733E+02,
    5.34147E+02,7.45462E+02,1.04038E+03,1.45196E+03,2.02638E+03,2.82804E+03,3.94684E+03,5.50826E+03,
    7.68740E+03,1.07286E+04])
    
   
# ncdm is the total neutrino mass in eV.
def make_specific_cosmology_transfer_function(directory, Omega0_m, sigma8, w, Omega0_b, h, n_s, ncdm, P_k_max):

    from nbodykit.lab import cosmology

    transfer_function_file_name = os.path.join(directory, "transfer_function.txt")
    
    k = set_of_wavenumbers()
    cosmology_object = make_specific_cosmology(directory, Omega0_m, sigma8, w, Omega0_b, h, n_s, ncdm, P_k_max)
    transfer_function = np.column_stack([k, cosmology.power.transfers.CLASS(cosmology_object, 0.0)(k)])
    transfer_function = trim_rows_containing_nan(transfer_function)
    np.savetxt(transfer_function_file_name, transfer_function, delimiter = " ", fmt = "%10.5E")
        
    linear_power_file_name = os.path.join(directory, "linear_power.txt")
    linear_power = np.column_stack([k, cosmology.power.linear.LinearPower(cosmology_object, 0.0, transfer='CLASS')(k)])
    linear_power = trim_rows_containing_nan(linear_power)
    np.savetxt(linear_power_file_name, linear_power, delimiter = " ", fmt = "%10.5E")
    
    
    if False:
        print("Saved transfer function in {}".format(transfer_function_file_name))
        print("Parameters:")
        for line in cosmology_summary(cosmology_object):
            print(line)
            
    return cosmology_object


def make_specific_cosmology_transfer_function_caller():
    make_specific_cosmology_transfer_function("/rds/user/dc-whit2/rds-dirac-dp153/lfi_project/foo",
        Omega0_m = 0.249628758416763685,
        sigma8 = 0.950496160083634467,
        w = -0.792417404234305733,
        Omega0_b = 0.043508831007317443,
        h = 0.716547375449984592,
        n_s = 0.951311068780816615,
        ncdm = 0.06,
        P_k_max = 100.0)
    
    

    
# ======================== End of code for handling transfer functions ========================


# ======================== Start of code for creating input files for multiple runs ========================

def double_quoted_string(s):
    return '"' + s + '"'
    
def change_one_value_in_ini_file(file_name, key, new_value):
    list_of_lines = []
    
    key_was_found = False
    with open(file_name, 'r') as infile:
        for line in infile:
            if line.find(key) == 0:
                new_line = key + new_value + "\n"
                list_of_lines.append(new_line)
                key_was_found = True
            else:
                list_of_lines.append(line)
            
    if not key_was_found:
        raise RuntimeError("Key {} not found in {}".format(key, file_name))
    
    with open(file_name, 'w') as outfile:
        for line in list_of_lines:
            outfile.write(line)
    

# See https://stackoverflow.com/questions/606561/how-to-get-filename-of-the-main-module-in-python
def get_exec_path():
    return os.path.dirname(os.path.abspath(sys.modules['__main__'].__file__))

# location can be 'wilkes', 'tursa', 'splinter' or 'current'
def project_directory(location):
    if location == 'wilkes':
        return "/rds/project/dirac_vol5/rds-dirac-dp153/lfi_project"
    elif location == 'tursa':
        return "/mnt/lustre/tursafs1/home/dp153/dp153/shared/lfi_project"
    elif location == 'splinter':
        return "/share/splinter/ucapwhi/lfi_project"
    elif location == 'current':
        # Return parent directory of directory containing script.
        # See also https://stackoverflow.com/questions/2860153/how-do-i-get-the-parent-directory-in-python
        # for alternative solutions.
        script_directory = os.path.dirname(os.path.abspath(sys.modules['__main__'].__file__))
        return os.path.dirname(script_directory)
    
    
def zfilled_run_num(run_num):
    return str(run_num).zfill(3)


def job_script_file_name_no_path(location):
    return 'cuda_job_script_{}'.format(location)
    
    
    
def make_file_executable(file_name):
    st = os.stat(file_name)
    os.chmod(file_name, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH) # See https://stackoverflow.com/questions/12791997/
    
    
# Note that the lines in list_of_set_environment_commands should be '\n' terminated.
def write_run_script(location, runs_letter, run_string, run_script_file_name, list_of_set_environment_commands):
    with open(run_script_file_name, 'w') as out_file:
        out_file.write("#!/usr/bin/env bash\n") # See https://stackoverflow.com/questions/10376206/what-is-the-preferred-bash-shebang for the full discussion...
        for e in list_of_set_environment_commands:
            out_file.write(e)
        # Go to the run directory
        out_file.write("cd {}/runs{}/run{}/\n".format(project_directory(location), runs_letter, run_string))
        # Delete any residual 'stop' file that would prevent the monitor from running.
        out_file.write("rm -f {}/runs{}/run{}/monitor_stop.txt\n".format(project_directory(location), runs_letter, run_string))
        # Start the monitor running in the background.
        out_file.write("{}/scripts/monitor.py {}/runs{}/run{}/ > {}/runs{}/run{}/monitor_output.txt &\n".format(project_directory(location),  project_directory(location), runs_letter, run_string, project_directory(location), runs_letter, run_string))
        # Run pkdgrav3
        out_file.write("{}/pkdgrav3/build_{}/pkdgrav3 ./control.par > ./output.txt\n".format(project_directory(location), location))
        # Create a 'stop' file to stop the monitor program gracefully. This might take up to 5 minutes to have an effect, but that's OK
        # as the remaining steps in this batch file will probably take longer. And it's no big deal of the monitor program stops
        # ungracefully.
        out_file.write("echo stop > {}/runs{}/run{}/monitor_stop.txt\n".format(project_directory(location), runs_letter, run_string))
        # Do the post-processing of pkdgrav3 output
        out_file.write("python3 {}/scripts/pkdgrav3_postprocess.py -l -d -z -f . >> ./output.txt\n".format(project_directory(location)))
        # Go to the parent directory
        out_file.write("cd {}/runs{}/\n".format(format(project_directory(location)), runs_letter))
        # Zip up all the files
        out_file.write("tar czvf run{}.tar.gz ./run{}/\n".format(run_string, run_string))
        # If the zip worked OK, then delete most of the files in the run directory
        out_file.write("test -f ./run{}.tar.gz && rm ./run{}/run*\n".format(run_string, run_string))
    make_file_executable(run_script_file_name)
    
    
def write_run_script_test_harness():

    location = "wilkes"
    runs_letter = "Z"
    run_string = "001"
    run_script_file_name = "./foo.sh"
    list_of_set_environment_commands = ["source something\n"]
    write_run_script(location, runs_letter, run_string, run_script_file_name, list_of_set_environment_commands)
    
    


def create_input_files_for_multiple_runs(runs_letter):

    runs_directory = os.path.join(project_directory("current"), "runs{}".format(runs_letter))
    
    batch_number_dict = {'I' : 3, 'J' : 4, 'K' : 5, 'L' : 6, 'M' : 7, 'N' : 8}
    batch_number = batch_number_dict[runs_letter]

    cosmo_params_for_all_runs_file_name = os.path.join(runs_directory, "params_run_{}.txt".format(batch_number))
    cosmo_params_for_all_runs = np.loadtxt(cosmo_params_for_all_runs_file_name, delimiter=',').reshape([-1,7]) # The 'reshape' handles the num_runs=1 case.
    num_runs = cosmo_params_for_all_runs.shape[0]
    
    tursa_numa_wrapper_file_name_no_path = "wrapper.sh"
    
    original_wilkes_job_script_file_name = os.path.join(runs_directory, job_script_file_name_no_path("wilkes"))
    original_tursa_job_script_file_name = os.path.join(runs_directory, job_script_file_name_no_path("tursa"))
    original_numa_wrapper_file_name = os.path.join(runs_directory, tursa_numa_wrapper_file_name_no_path)
    
    control_file_name_no_path = 'control.par'
    original_control_file_name = os.path.join(runs_directory, control_file_name_no_path)
    
    random_seed_offset_dict = {'C' : 0, 'E' : 128, 'I' : 192, 'J' : 320, 'K' : 384, 'L' : 401, 'M' : 530, 'N' : 658}
    random_seed_offset = random_seed_offset_dict[runs_letter]
    
    
    
    for run_num_zero_based in range(num_runs):
        run_num_one_based = run_num_zero_based + 1
        
        # Amend the code here to restrict to just certain directories.
        if (True):
        
            print("{} of {}".format(run_num_one_based, num_runs))
            
            run_string = zfilled_run_num(run_num_one_based)
            
            this_run_directory = os.path.join(runs_directory, "run" + run_string)
            this_wilkes_job_script_file_name = os.path.join(this_run_directory, job_script_file_name_no_path("wilkes"))
            this_tursa_job_script_file_name = os.path.join(this_run_directory, job_script_file_name_no_path("tursa"))
            this_control_file_name = os.path.join(this_run_directory, control_file_name_no_path)
            run_script_name_wilkes = os.path.join(this_run_directory, "pkdgrav3_and_post_process_wilkes.sh")
            run_script_name_tursa = os.path.join(this_run_directory, "pkdgrav3_and_post_process_tursa.sh")
            
        
            # Make directory
            os.makedirs(this_run_directory, exist_ok = True)
            
            # Wilkes job script
            copyfile(original_wilkes_job_script_file_name, this_wilkes_job_script_file_name)
            change_one_value_in_ini_file(this_wilkes_job_script_file_name, '#SBATCH --time=', '35:59:00')
            change_one_value_in_ini_file(this_wilkes_job_script_file_name, '#SBATCH -J ', 'pgr3_{}'.format(run_string))
            change_one_value_in_ini_file(this_wilkes_job_script_file_name, 'application=', double_quoted_string(run_script_name_wilkes))
            
            # Tursa job script
            copyfile(original_tursa_job_script_file_name, this_tursa_job_script_file_name)
            change_one_value_in_ini_file(this_tursa_job_script_file_name, '#SBATCH --time=', '47:59:00')
            change_one_value_in_ini_file(this_tursa_job_script_file_name, '#SBATCH --job-name=', 'pgr3_{}'.format(run_string))
            change_one_value_in_ini_file(this_tursa_job_script_file_name, 'application=', double_quoted_string(run_script_name_tursa))
            
            # Cosmology object
            cosmology_object = make_specific_cosmology(this_run_directory,
                Omega0_m = cosmo_params_for_all_runs[run_num_zero_based, 0],
                sigma8 = cosmo_params_for_all_runs[run_num_zero_based, 1],
                w = cosmo_params_for_all_runs[run_num_zero_based, 2],
                Omega0_b = cosmo_params_for_all_runs[run_num_zero_based, 3],
                h = cosmo_params_for_all_runs[run_num_zero_based, 4],
                n_s = cosmo_params_for_all_runs[run_num_zero_based, 5],
                ncdm = cosmo_params_for_all_runs[run_num_zero_based, 6],
                P_k_max=100.0)
                
            ###OmegaDE = cosmology_object.Ode0

            # Control file
            copyfile(original_control_file_name, this_control_file_name)
            ###change_one_value_in_ini_file(this_control_file_name, 'achTfFile       = ', '"./transfer_function.txt"')
            ###change_one_value_in_ini_file(this_control_file_name, 'dOmega0         = ', str(1.0-OmegaDE) + "    # 1-dOmegaDE")
            ###change_one_value_in_ini_file(this_control_file_name, 'dOmegaDE        = ', str(OmegaDE) + "    # Equal to Omega_fld in transfer function")
            ###change_one_value_in_ini_file(this_control_file_name, 'dSigma8         = ', str(cosmo_params_for_all_runs[run_num_zero_based, 1]))
            #### Work around pkdgrav3 ini file parsing bug - doesn't like negative numbers.
            ###acos_w_string = "2.0*math.cos({})  # {}".format(math.acos(cosmo_params_for_all_runs[run_num_zero_based, 2] / 2.0), cosmo_params_for_all_runs[run_num_zero_based, 2])
            ###change_one_value_in_ini_file(this_control_file_name, 'w0              = ', acos_w_string)
            ###change_one_value_in_ini_file(this_control_file_name, 'h               = ', str(cosmo_params_for_all_runs[run_num_zero_based, 4]))
            change_one_value_in_ini_file(this_control_file_name, 'dSpectral        = ', str(cosmo_params_for_all_runs[run_num_zero_based, 5]))
            change_one_value_in_ini_file(this_control_file_name, 'dNormalization   = ', "{} # calculated from sigma_8 = {}".format(cosmology_object.A_s, cosmology_object.sigma8))
            
            hdf5_file_name = '"./class_processed_batch{}_{}.hdf5"'.format(str(batch_number), run_string)
            change_one_value_in_ini_file(this_control_file_name, 'achClassFilename = ', hdf5_file_name)
            
            change_one_value_in_ini_file(this_control_file_name, 'iSeed           = ', str(run_num_one_based + random_seed_offset) + "        # Random seed")
            
            change_one_value_in_ini_file(this_control_file_name, 'dBoxSize        = ', "1250       # Mpc/h")
            change_one_value_in_ini_file(this_control_file_name, 'nGrid           = ', "1080       # Simulation has nGrid^3 particles")
            change_one_value_in_ini_file(this_control_file_name, 'nSideHealpix    = ', "2048 # NSide for output lightcone healpix maps.")
            
            
            # Wilkes run script
            wilkes_set_environment_commands = ["module load python/3.8\n", "source {}/env/bin/activate\n".format(project_directory("wilkes"))]
            write_run_script("wilkes", runs_letter, run_string, run_script_name_wilkes, wilkes_set_environment_commands)

            
            # Tursa run script
            tursa_set_environment_commands = ["source {}/set_environment_tursa.sh\n".format(project_directory("tursa"))]
            write_run_script("tursa", runs_letter, run_string, run_script_name_tursa, tursa_set_environment_commands)
            
            # Tursa numa wrapper
            this_numa_wrapper_file_name = os.path.join(this_run_directory, tursa_numa_wrapper_file_name_no_path)
            copyfile(original_numa_wrapper_file_name, this_numa_wrapper_file_name)
            make_file_executable(this_numa_wrapper_file_name)


def start_time_end_time(directory):
    file_list = glob.glob(os.path.join(directory, "machine.file.*"))
    if not file_list:
        return 0
    start_time = os.path.getmtime(file_list[0])
    file_list = glob.glob(os.path.join(directory, "slurm*"))
    if not file_list:
        return 0
    end_time = os.path.getmtime(file_list[0])
    return (start_time, end_time)



def calculate_each_run_time_and_show_Gantt_chart():

    y = []
    width = []
    left = []
    
    for run_num_zero_based in range(128):
        run_num_one_based = run_num_zero_based + 1
        run_string = zfilled_run_num(run_num_one_based)
        this_directory = "/rds/user/dc-whit2/rds-dirac-dp153/lfi_project/runs{}/run{}/".format(runs_letter(), run_string)
        (start_time, end_time) = start_time_end_time(this_directory)
        run_time_in_seconds = end_time - start_time
        print(this_directory, "{:.3f}".format(run_time_in_seconds/3600))
        y.append(run_num_one_based)
        width.append(run_time_in_seconds)
        left.append(start_time)
    plt.barh(y=y, width=width, left=left)
    plt.savefig("/rds/user/dc-whit2/rds-dirac-dp153/lfi_project/scripts/gantt.png")
        



def show_last_unprocessed_file():
    print(datetime.datetime.now().time())
    for run_num_zero_based in range(129):
        run_num_one_based = run_num_zero_based + 1
        run_string = zfilled_run_num(run_num_one_based)
        this_directory = "/rds/user/dc-whit2/rds-dirac-dp153/lfi_project/runs{}/run{}/".format('L', run_string)
        these_files = sorted(glob.glob(os.path.join(this_directory, "*.0")))
        if these_files:
            print(these_files[-1])


# ======================== End of code for creating input files for multiple runs ========================


# ======================== Start of code for expand_shell_script ========================

# This functionality is exposed by the module expand_shell_script.py.

# Converts a string such as "1-128 x14,15,26,53" into a list of integers
def decode_list_of_jobs_string(s):

    res_set = set()
    for token in s.split():
        # token might be for example 'x1-5,7' or '5,6,8,9,'
        include = True
        if token[0].lower() == 'x':
            token = token[1:]
            include = False
        
        for subtoken in token.split(","):
            # subtoken might be for example '1-3' or '7'
            parsed_subtoken = subtoken.split('-')
            vals = []
            if len(parsed_subtoken) == 1:
                vals.append(int(parsed_subtoken[0]))
            elif len(parsed_subtoken) == 2:
                vals.extend(range(int(parsed_subtoken[0]), int(parsed_subtoken[1]) + 1))
            else:
                raise AssertionError("Could not parse expression {}".format(subtoken))
            for v in vals:
                if include:
                    res_set.add(v)
                else:
                    res_set.discard(v)
                
    return sorted(list(res_set))
    
    
def expand_shell_script(original_shell_script_file_name, new_shell_script_file_name, list_of_jobs_string):

    print(original_shell_script_file_name)
    print(new_shell_script_file_name)
    print(list_of_jobs_string)
    print(decode_list_of_jobs_string(list_of_jobs_string))

    with open(new_shell_script_file_name, 'w') as out_file:
        for job in decode_list_of_jobs_string(list_of_jobs_string):
            with open(original_shell_script_file_name, 'r') as in_file:
                for line in in_file:
                    out_file.write(line.replace("{}", str(job).zfill(3)))
    make_file_executable(new_shell_script_file_name)

        
# ======================== End of code for expand_shell_script ========================

if __name__ == '__main__':
    
    #show_one_shell_example()
    #match_points_between_boxes()
    #read_one_box_example()
    #save_all_lightcone_image_files("/share/splinter/ucapwhi/lfi_project/experiments/k80_1024_4096_900/", True)
    #compare_two_lightcones_by_power_spectra()
    #get_float_from_control_file_test_harness()
    #compare_two_time_spacings()
    #make_specific_cosmology_transfer_function_caller()
    #monitor()
    #tomographic_slice_number_from_lightcone_file_name_test_harness()
    #object_count_file_test_harness()
    create_input_files_for_multiple_runs('N')
    #calculate_each_run_time_and_show_Gantt_chart()
    #show_last_unprocessed_file()
    #write_run_script_test_harness()
    #get_parameter_from_log_file_test_harness()
     
    pass
    
