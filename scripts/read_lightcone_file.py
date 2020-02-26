#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
    Converts lightcone output from PKDGRAV3 to healpix map format.
    Author: Lorne Whiteway.
"""
    
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
    
    total_num_objects = 0
    for t in sorted(time_indices):
        basefilename = filespec.format(str(t).zfill(num_digits_for_time_index))
        map_t = read_one_healpix_map(basefilename, nside)
        num_objects_t = np.sum(map_t)
        print(t, num_objects_t)
        total_num_objects += num_objects_t
    print("======")
    print(total_num_objects)
    
    
def main():
    import numpy as np
    filespec = "/share/splinter/ucapwhi/lfi_project/experiments/example.{}.hpb"
    time_indices = 1 + np.arange(100)
    num_digits_for_time_index = 5
    nside = 16
    read_lightcone(filespec, time_indices, num_digits_for_time_index, nside)


if __name__ == '__main__':
    main()

