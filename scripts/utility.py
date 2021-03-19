#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
    Routines for analyzing PKDGRAV3 output.
    Author: Lorne Whiteway.
"""

import glob
import numpy as np
import matplotlib
#matplotlib.use('Agg') # See http://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined
import matplotlib.pyplot as plt
import healpy as hp
from astropy.cosmology import FlatLambdaCDM
import datetime
import re
from numpy import random
import cosmic_web_utilities as cwu 
import configparser



# ======================== Start of code for reading lightcone files ========================


# basefilename will be something like "/share/splinter/ucapwhi/lfi_project/experiments/simple/example.00001.hpb"
# The actual files to be read will then have names like "/share/splinter/ucapwhi/lfi_project/experiments/simple/example.00001.hpb.0"
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
    


# filespec will be something like "/share/splinter/ucapwhi/lfi_project/experiments/simple/example.{}.hpb"
# Output will be a list like ["/share/splinter/ucapwhi/lfi_project/experiments/simple/example.00001.hpb", etc.]
def basefilename_list_from_filespec(filespec):
    return [f[:-2] for f in sorted(glob.glob(filespec.replace("{}", "*") + ".0"))]

    
def num_objects_in_lightcones():


    filespec = "/share/splinter/ucapwhi/lfi_project/experiments/gpu_512_4096_900/example.{}.hpb"
    nside = 4096
    
    total_num_objects = 0
    for b in basefilename_list_from_filespec(filespec):
        map_b = one_healpix_map_from_basefilename(b, nside)
        num_objects_b = np.sum(map_b)
        print(b, num_objects_b)
        total_num_objects += num_objects_b
    print("======")
    print(total_num_objects)



    

# Example filespec: "/share/splinter/ucapwhi/lfi_project/experiments/gpu_1000_1024_1000/example.{}.hpb"
def save_all_lightcone_files_caller_core(filespec, nside, plot_one_example, new_nside = None):

    if new_nside is None:
        new_nside = nside

    for b in basefilename_list_from_filespec(filespec):
        map_t = one_healpix_map_from_basefilename(b, nside)
        if new_nside != nside:
            map_t = hp.ud_grade(map_t, new_nside)
        output_file_name = b.replace(".hpb", ".lightcone.npy")
        if np.sum(map_t) > 0:
            print("Writing file {}...".format(output_file_name))
            np.save(output_file_name, map_t)
        else:
            print("Not writing file {} as it would have no objects.".format(output_file_name))
    
    if plot_one_example:
        filename = filespec.replace(".{}.hpb", ".089.lightcone.npy")
        plot_lightcone_files([filename])


def save_all_lightcone_files():
    filespec = "/share/splinter/ucapwhi/lfi_project/experiments/gpu_1024_4096_900/example.{}.hpb"
    nside = 16
    save_all_lightcone_files_caller_core(filespec, nside, False)

    
def plot_lightcone_files(list_of_npy_filenames, do_show = True, do_save = False):
    
    for filename in list_of_npy_filenames:
        print("Using file {}".format(filename))
        
        map_t = np.load(filename)
        
        print("Number of objects = {}".format(np.sum(map_t)))
        print("Number of pixels = {}".format(map_t.shape))
        
        # Histogram of pixel values
        if False:
            plt.hist(map_t, bins=np.max(map_t)+1)
            plt.yscale('log', nonposy='clip') # See https://stackoverflow.com/questions/17952279/logarithmic-y-axis-bins-in-python
            plt.show()
        
        if True:
            map_t = np.log10(map_t) - np.log10(np.mean(map_t)) # log10(1+\delta)
        
        cmap=plt.get_cmap('magma')
        
        
        if True:
            hp.mollview(map_t, title=filename.replace("/share/splinter/ucapwhi/lfi_project/experiments/", ""), cbar=True, cmap=cmap)
            #hp.graticule(dpar=30.0)
            save_file_extension = "mollview.png"
        elif False:
            rot = (50.0, -40.0, 0.0)
            hp.orthview(map_t, rot=rot, title=filename.replace("/share/splinter/ucapwhi/lfi_project/experiments/", ""), cbar=True, cmap=cmap)
            save_file_extension = "png"
        else:
            rot = (0.0, 0.0, 0.0)
            hp.gnomview(map_t, rot=rot, reso=0.4, xsize=500, cbar=True, cmap=cmap)
            save_file_extension = "gnomview.png"

        if do_save:
            save_file_name = filename.replace("npy", save_file_extension)
            print("Saving {}...".format(save_file_name))    
            plt.savefig(save_file_name)

        
    if do_show:
        plt.show()
        
    
            
def show_one_lightcone():

    filename = "/share/splinter/ucapwhi/lfi_project/experiments/gpu_1024_1024_900/example.00115.lightcone.npy"
    plot_lightcone_files([filename,])
    
def save_one_lightcone():
    filename = "/share/splinter/ucapwhi/lfi_project/experiments/gpu_1024_1024_1536/example.00115.lightcone.npy"
    plot_lightcone_files([filename,], False, True)

    
def show_two_lightcones():

    filenames = ["/share/splinter/ucapwhi/lfi_project/experiments/gpu_fast/example.00093.lightcone.npy",
        "/share/splinter/ucapwhi/lfi_project/experiments/gpu_fast_amendedtransfer/example.00093.lightcone.npy"]
    plot_lightcone_files(filenames)
    #maps = [np.load(f) for f in filenames]
    #for (i, c0, c1) in zip(range(len(maps[0])), maps[0], maps[1]):
    #    if c0 != c1:
    #        print(i, c0, c1)
        
    
def save_all_lightcone_image_files(directory):
    file_list = glob.glob(directory + "*.npy")
    plot_lightcone_files(file_list, False, True)
    

def save_all_lightcone_image_files_caller():
    directory = "/share/splinter/ucapwhi/lfi_project/experiments/gpu_1024_1024_1536/"
    save_all_lightcone_image_files(directory)
    
    

# From https://stackoverflow.com/questions/4666973/how-to-extract-the-substring-between-two-markers
def string_between_strings(s_to_search, left_s, right_s):
    try:
        ret = re.search(left_s + '(.+?)' + right_s, s_to_search).group(1)
    except AttributeError:
        ret = ''
    return ret
    

def string_between_strings_test_harness():
    print(string_between_strings("ABC", "A", "C"))
    print(string_between_strings("xxx123zzz", "xxx", "zzz"))
    
    
    
    
def count_objects_in_many_lightcone_files():
    directory = "/share/splinter/ucapwhi/lfi_project/experiments/gpu_512_1024_1000/"
    file_list = glob.glob(directory + "*.lightcone.npy")
    file_list.sort()
    
    file_num_list = []
    num_objects_list = []
    for el in file_list:
    
        file_num = int(string_between_strings(el, "example.", ".lightcone.npy"))
        num_objects = np.load(el).sum()
        
        file_num_list.append(file_num)
        num_objects_list.append(num_objects)
        
        print(file_num, num_objects)
        
    file_num_array = np.array(file_num_list)
    num_objects_array = np.array(num_objects_list)
    
    array_file_name = directory + "object_count.txt"
    print("Saving array to {}".format(array_file_name))
    np.save(array_file_name, np.column_stack([file_num_array, num_objects_array]))
    
    
    plt.plot(file_num_array, num_objects_array)
    plot_file_name = directory + "object_count.png"
    print("Saving plot to {}".format(plot_file_name))
    plt.savefig(plot_file_name)
    
        
        
    
    
    


# ======================== End of code for reading lightcone files ========================







# ======================== Start of code for reading boxes ========================

def get_header_type():

    return np.dtype([('time','>f8'),('N','>i4'),('Dims','>i4'),('Ngas','>i4'),('Ndark', '>i4'),('Nstar','>i4'),('pad','>i4')])

def get_dark_type():

    return np.dtype([('mass','>f4'),('x','>f4'),('y','>f4'),('z','>f4'),('vx','>f4'),('vy','>f4'),('vz','>f4'),('eps','>f4'),('phi','>f4')])


# file_name will be something like "/share/splinter/ucapwhi/lfi_project/experiments/simple/example.00100"
# This is an edited version the code in the readtipsy.py file distributed with PKDGRAV3.
# Returns an array with elements of type get_dark_type()
def read_one_box(file_name):

    

    with open(file_name, 'rb') as in_file:

        header = np.fromfile(in_file, dtype=get_header_type(), count=1)
        header = dict(zip(get_header_type().names, header[0]))
        print(header)
        
        return np.fromfile(in_file, dtype=get_dark_type(), count=header['Ndark']) # Just dark matter (not gas or stars)
    
    


def read_one_box_example():
    #file_name = "/share/splinter/ucapwhi/lfi_project/experiments/gpu_1024_4096_900/example.00071"
    file_name = "/share/splinter/ucapwhi/lfi_project/experiments/simple_64_64/example.00071"
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
    plt.savefig("/share/splinter/ucapwhi/lfi_project/scripts/foo.png")
    #print(d_x.shape[0])
    #print(d_x[z_filter].shape[0])
    
    

# File_name will be something like "/share/splinter/ucapwhi/lfi_project/experiments/simple/example.00100"
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

    file_name = "/share/splinter/ucapwhi/lfi_project/experiments/simple/example.00099"
    shell_low = 0.1137
    shell_high = 0.228
    nside = 128
    show_one_shell(file_name, shell_low, shell_high, nside)
    
    
def match_points_between_boxes():

    file_names = ["/share/splinter/ucapwhi/lfi_project/experiments/simple/example.00001", "/share/splinter/ucapwhi/lfi_project/experiments/simple/example.00100"]

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


            
def show_one_output_file(filename):

    (s_arr, t_arr, z_arr) = read_one_output_file(filename)
    
    hubble_constant = 0.67
    
    cosmo = FlatLambdaCDM(H0=100*hubble_constant, Om0=0.32)
    c_arr = cosmo.comoving_distance(z_arr).value # In Mpc

    if True:
        c_arr_diffs = c_arr[:-1] - c_arr[1:] #This way around as the arrays go from high z to low z
        z_arr_diffs = z_arr[:-1] - z_arr[1:] #This way around as the arrays go from high z to low z
        if False:
            plt.scatter(z_arr[:-1], c_arr_diffs[:]*hubble_constant, s=5)
            plt.ylabel('Shell thickness (Mpc/h)')
        else:
            plt.scatter(z_arr[:-1], z_arr_diffs[:], s=5)
            plt.ylabel('Shell thickness (delta redshift)')
        
        plt.xlabel('Redshift')
        plt.show()
    
    
    print("Step\tTime\tRedshift\tComoving r(Mpc)\tComoving r(Mpc/h)")
    for (s, t, z, c) in zip(s_arr, t_arr, z_arr, c_arr):
        print(s,t,z,c,c*hubble_constant)
        pass
    
def show_one_output_file_example():
    filename = "/share/splinter/ucapwhi/lfi_project/experiments/gpu_32_512_900/example_output.txt"
    show_one_output_file(filename)
    
    
def build_z_values_file(directory):

    input_filename = directory + "/example_output.txt"
    output_filename = directory + "/z_values.txt"
    
    (s_arr, t_arr, z_arr) = read_one_output_file(input_filename)
    
    header = "Step,z,t,a,1/a,d(Mpc/h),d/BoxSize"
    
    a_arr = 1.0 / (1.0 + z_arr)

    config = configparser.ConfigParser(inline_comment_prefixes="#", comment_prefixes=("#", "import"))
    # See https://stackoverflow.com/questions/2885190/using-configparser-to-read-a-file-without-section-name
    with open(directory + "/control.par") as stream:
        config.read_string("[top]\n" + stream.read())
    Om0 = config['top']["dOmega0"]
    cosmo = FlatLambdaCDM(H0=100.0, Om0=Om0)
    c_arr = cosmo.comoving_distance(z_arr).value # In Mpc/h
    
    box_size = float(config['top']["dBoxSize"])
    
    np.savetxt(output_filename, np.column_stack((s_arr, z_arr, t_arr, a_arr, 1.0/a_arr, c_arr, c_arr/box_size)), fmt=["%i", "%f6", "%f6", "%f6", "%f6", "%f6", "%f6"], delimiter=",", header=header)
    


def build_z_values_file_caller():
    directory = "/share/splinter/ucapwhi/lfi_project/experiments/gpu_1024_4096_900"
    build_z_values_file(directory)
    


def display_z_values_file(directory):
    
    input_filename = directory + "/z_values.txt"
    
    d = np.loadtxt(input_filename, delimiter=",", skiprows=1)
    s_arr = d[:,0]
    z_arr = d[:,1]
    t_arr = d[:,2]
    a_arr = d[:,3]
    inverse_a_arr = d[:,4]
    d_arr = d[:,5]
    d_over_boxsize_arr = d[:,6]
    
    
    plt.scatter(s_arr[0:], inverse_a_arr[0:])
    plt.show()
    

# Run this after PKDGRAV3 has finished to do all the postprocessing
def post_run_process():
    
    # Set directory and nside to the appropriate values...
    directory = "/share/splinter/ucapwhi/lfi_project/experiments/gpu_probtest/"
    nside = 64
    new_nside = 64
    
    print("Processing {} with nside {}".format(directory, nside))
    
    filespec = directory + "example.{}.hpb"
    save_all_lightcone_files_caller_core(filespec, nside, False, new_nside)
    
    save_all_lightcone_image_files(directory)
    
    build_z_values_file(directory)
    
    #display_z_values_file(directory)


    
    
# ======================== End of code for reading PKDGRAV3 output ========================    


# ======================== Start of other utilities ========================    

def intersection_of_shell_and_cells():

    radius = 2.418
    nside = 128
    npix = hp.nside2npix(nside)
    
    (ra, dec) = hp.pix2ang(nside, np.arange(npix), False, lonlat=True)
    (x, y, z) = cwu.spherical_to_cartesian(ra, dec, distance = radius)
    
    
    to_be_sorted = np.column_stack([np.random.uniform(size=216),np.arange(216)])
    random_indices = to_be_sorted[to_be_sorted[:,0].argsort()].astype(int)[:,1]
    
    
        
    if True:
        # Box number
        hp_map = random_indices[(np.floor(x).astype(int) + 3) * 36 + (np.floor(y).astype(int) + 3) * 6 + (np.floor(z).astype(int) + 3)]
    else:
        # Shell number
        hp_map = np.floor(np.maximum.reduce([np.abs(x), np.abs(y), np.abs(z)]))
    
    rot = (50.0, -40.0, 0.0)
    cmap=plt.get_cmap('magma')
    if False:
        hp.mollview(hp_map, cmap=cmap)
    else:
        hp.orthview(hp_map, rot=rot, cmap=cmap)
    
    plt.show()
    

                        


# ======================== End of other utilities ========================    



if __name__ == '__main__':
    
    #show_one_output_file_example()
    #show_one_shell_example()
    #match_points_between_boxes()
    #save_all_lightcone_files()
    #show_one_lightcone()
    #show_two_lightcones()
    #num_objects_in_lightcones()
    #display_z_values_file("/share/splinter/ucapwhi/lfi_project/experiments/gpu_1024_4096_900/")
    #post_run_process()
    #read_one_box_example()
    #count_objects_in_many_lightcone_files()
    #string_between_strings_test_harness()
    #save_all_lightcone_image_files_caller()
    intersection_of_shell_and_cells()
    #build_z_values_file_caller()
