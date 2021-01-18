#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
    Routines for analyzing PKDGRAV3 output.
    Author: Lorne Whiteway.
"""

import glob
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9 as cosmo



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
    assert len(healpix_map) >= n_pixels, "{} yields {} elements, which is too few for nside{}".format(filespec, len(healpix_map), nside)
    # Has padding at end; ensure that this padding is empty
    assert np.all(healpix_map[n_pixels:] == 0), "{} yields non-zero elements in the padding area".format(filespec)
    
    return healpix_map[:n_pixels]
    


# filespec will be something like "/share/splinter/ucapwhi/lfi_project/experiments/simple/example.{}.hpb"
# Output will be a list like ["/share/splinter/ucapwhi/lfi_project/experiments/simple/example.00001.hpb", etc.]
def basefilename_list_from_filespec(filespec):
    return [f[:-2] for f in sorted(glob.glob(filespec.replace("{}", "*") + ".0"))]

    
def num_objects_in_lightcones():


    filespec = "/share/splinter/ucapwhi/lfi_project/experiments/gpu_1024_1024_1000/example.{}.hpb"
    nside = 1024
    
    total_num_objects = 0
    for b in basefilename_list_from_filespec(filespec):
        map_b = one_healpix_map_from_basefilename(b, nside)
        num_objects_b = np.sum(map_b)
        print(b, num_objects_b)
        total_num_objects += num_objects_b
    print("======")
    print(total_num_objects)



    

# Example filespec: "/share/splinter/ucapwhi/lfi_project/experiments/gpu_1000_1024_1000/example.{}.hpb"    
def save_all_lightcone_files_caller_core(filespec, nside, plot_one_example):

    for b in basefilename_list_from_filespec(filespec):
        map_t = one_healpix_map_from_basefilename(b, nside)
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
    filespec = "/share/splinter/ucapwhi/lfi_project/experiments/gpu_768_900_2048/example.{}.hpb"
    nside = 2048
    save_all_lightcone_files_caller_core(filespec, nside, False)

    
def plot_lightcone_files(list_of_npy_filenames):
    

    
    for filename in list_of_npy_filenames:
        map_t = np.load(filename)
        if True:
            map_t = np.log10(map_t) - np.log10(np.mean(map_t)) # log10(1+\delta)
        cmap=plt.get_cmap('magma')
        if False:
            hp.mollview(map_t, title=filename.replace("/share/splinter/ucapwhi/lfi_project/experiments/", ""), cbar=True, cmap=cmap)
            hp.graticule(dpar=30.0)
        else:
            rot = (0.0, 0.0, 0.0)
            hp.gnomview(map_t, rot=rot, reso=0.4, xsize=500, cbar=True, cmap=cmap)
        
    plt.show()
            
def show_one_lightcone():

    filename = "/share/splinter/ucapwhi/lfi_project/experiments/gpu_1024_1024_1000/example.068.lightcone.npy"
    plot_lightcone_files([filename,])
    
    
def show_two_lightcones():

    filenames = ["/share/splinter/ucapwhi/lfi_project/experiments/simple_32_32/example.088.lightcone.npy",
        "/share/splinter/ucapwhi/lfi_project/experiments/computenode_32_32/example.088.lightcone.npy"]
    #plot_lightcone_files(filenames)
    maps = [np.load(f) for f in filenames]
    for (i, c0, c1) in zip(range(len(maps[0])), maps[0], maps[1]):
        if c0 != c1:
            print(i, c0, c1)
        
    
    


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
        return np.fromfile(in_file, dtype=get_dark_type(), count=header['Ndark']) # Just dark matter (not gas or stars)
    

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
    
    if False:
        plt.plot(s_arr, t_arr)
        plt.plot(s_arr, z_arr)
        plt.show()
    
    
    for (s, t, z) in zip(s_arr, t_arr, z_arr):
        print(s,t,z)
    
def show_one_output_file_example():
    filename = "/share/splinter/ucapwhi/lfi_project/experiments/gpu_1024_1024_1000/example_output.txt"
    show_one_output_file(filename)
    
    
def build_z_values_file():

    
    directory = "/share/splinter/ucapwhi/lfi_project/experiments/gpu_512_4096_900"
    input_filename = directory + "/example_output.txt"
    output_filename = directory + "/z_values.txt"
    
    (s_arr, t_arr, z_arr) = read_one_output_file(input_filename)
    np.savetxt(output_filename, np.column_stack((s_arr, z_arr)), fmt=["%i", "%f6"], delimiter=",")
    
    
    

    
    
# ======================== End of code for reading PKDGRAV3 output ========================    



if __name__ == '__main__':
    
    #show_one_output_file_example()
    #show_one_shell_example()
    #match_points_between_boxes()
    #save_all_lightcone_files()
    #show_one_lightcone()
    #show_two_lightcones()
    #num_objects_in_lightcones()
    build_z_values_file()
    