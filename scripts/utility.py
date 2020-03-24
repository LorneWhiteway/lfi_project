#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
    Routines for analyzing PKDGRAV3 output.
    Author: Lorne Whiteway.
"""




# ======================== Start of code for reading lightcone files ========================


# basefilename will be something like "/share/splinter/ucapwhi/lfi_project/experiments/example.00001.hpb"
def read_one_healpix_map(basefilename, nside):

    import glob
    import numpy as np
    import healpy as hp
    
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
    


# filespec will be something like "/share/splinter/ucapwhi/lfi_project/experiments/example.{}.hpb"
def read_lightcone(filespec, time_indices, num_digits_for_time_index, nside):
    import numpy as np
    import glob
    import sys
    import healpy as hp
    import matplotlib.pyplot as plt
    
    showHealpixFiles = True
    
    total_num_objects = 0
    for t in sorted(time_indices):
        basefilename = filespec.format(str(t).zfill(num_digits_for_time_index))
        map_t = read_one_healpix_map(basefilename, nside)
        num_objects_t = np.sum(map_t)
        if showHealpixFiles and num_objects_t > 0 and t == 89:
            hp.mollview(map_t, title=str(t), xsize=400, badcolor="grey")
            hp.graticule(dpar=30.0)
            plt.show()
        print(t, num_objects_t)
        total_num_objects += num_objects_t
    print("======")
    print(total_num_objects)
    
    
def lightcone_example():
    import numpy as np
    filespec = "/share/splinter/ucapwhi/lfi_project/experiments/example.{}.hpb"
    time_indices = 1 + np.arange(100)
    num_digits_for_time_index = 5
    nside = 16
    read_lightcone(filespec, time_indices, num_digits_for_time_index, nside)

# ======================== End of code for reading lightcone files ========================







# ======================== Start of code for reading boxes ========================

def get_header_type():
    import numpy as np
    return np.dtype([('time','>f8'),('N','>i4'),('Dims','>i4'),('Ngas','>i4'),('Ndark', '>i4'),('Nstar','>i4'),('pad','>i4')])

def get_dark_type():
    import numpy as np
    return np.dtype([('mass','>f4'),('x','>f4'),('y','>f4'),('z','>f4'),('vx','>f4'),('vy','>f4'),('vz','>f4'),('eps','>f4'),('phi','>f4')])


# file_name will be something like "/share/splinter/ucapwhi/lfi_project/experiments/example.00100"
# This is an edited version the code in the readtipsy.py file distributed with PKDGRAV3.
# Returns an array with elements of type get_dark_type()
def read_one_box(file_name):

    import numpy as np
    with open(file_name, 'rb') as in_file:

        header = np.fromfile(in_file, dtype=get_header_type(), count=1)
        header = dict(zip(get_header_type().names, header[0]))
        return np.fromfile(in_file, dtype=get_dark_type(), count=header['Ndark']) # Just dark matter (not gas or stars)
    

# File_name will be something like "/share/splinter/ucapwhi/lfi_project/experiments/example.00100"
def show_one_shell(file_name, shell_low, shell_high, nside):
    import numpy as np
    import healpy as hp
    import matplotlib.pyplot as plt
    
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

    file_name = "/share/splinter/ucapwhi/lfi_project/experiments/example.00099"
    shell_low = 0.1137
    shell_high = 0.228
    nside = 16
    show_one_shell(file_name, shell_low, shell_high, nside)
    
    
def match_points_between_boxes():

    import numpy as np
    import matplotlib.pyplot as plt

    file_names = ["/share/splinter/ucapwhi/lfi_project/experiments/example.00001", "/share/splinter/ucapwhi/lfi_project/experiments/example.00100"]

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
    
    import numpy as np
    
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

    import matplotlib.pyplot as plt

    (s_arr, t_arr, z_arr) = read_one_output_file(filename)
    
    if False:
        plt.plot(s_arr, t_arr)
        plt.plot(s_arr, z_arr)
        plt.show()
    
    
    for (s, t, z) in zip(s_arr, t_arr, z_arr):
        print(s,t,z)
    
def show_one_output_file_example():
    filename = "/share/splinter/ucapwhi/lfi_project/experiments/output.txt"
    show_one_output_file(filename)
    
    
# ======================== End of code for reading PKDGRAV3 output ========================    



if __name__ == '__main__':
    
    #lightcone_example()
    #show_one_output_file_example()
    #show_one_shell_example()
    match_points_between_boxes()

