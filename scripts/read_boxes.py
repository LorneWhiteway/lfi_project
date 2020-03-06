#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
    Reads simulatioes output by PKDGRAV3.
    Author: Lorne Whiteway.
"""
    

def get_header_type():
    import numpy as np
    return np.dtype([('time','>f8'),('N','>i4'),('Dims','>i4'),('Ngas','>i4'),('Ndark', '>i4'),('Nstar','>i4'),('pad','>i4')])

def get_dark_type():
    import numpy as np
    return np.dtype([('mass','>f4'),('x','>f4'),('y','>f4'),('z','>f4'),('vx','>f4'),('vy','>f4'),('vz','>f4'),('eps','>f4'),('phi','>f4')])


# file_name will be something like "/share/splinter/ucapwhi/lfi_project/experiments/example.00100"
# This is an edited version the code in the readtipsy.py file distributed with PKDGRAV3.
def read_one_box(file_name):

    import numpy as np
    with open(file_name, 'rb') as in_file:

        header = np.fromfile(in_file, dtype=get_header_type(), count=1)
        header = dict(zip(get_header_type().names, header[0]))
        return np.fromfile(in_file, dtype=get_dark_type(), count=header['Ndark']) # Just dark matter (not gas or stars)
    
def main():
    import numpy as np
    import healpy as hp
    import matplotlib.pyplot as plt
    
    file_name = "/share/splinter/ucapwhi/lfi_project/experiments/example.00100"
    d = read_one_box(file_name)
    ra_list = []
    dec_list = []
    for start_x in [-0.5, 0.5]:
        for start_y in [-0.5, 0.5]:
            for start_z in [-0.5, 0.5]:
                dist = np.sqrt((d['x']-start_x)**2 + (d['y']-start_y)**2 + (d['z']-start_z)**2)
                for (dd, xx, yy, zz) in zip(dist, -(start_x-d['x']), -(start_y-d['y']), -(start_z-d['z'])):
                    if dd < 0.1137:
                        ra_list.append(np.degrees(np.arctan2(yy, xx)))
                        dec_list.append(np.degrees(np.arcsin(zz/dd)))
    ra = np.array(ra_list)
    dec = np.array(dec_list)
    print(ra, dec)
    
    nside = 16
    num_healpixels = hp.nside2npix(nside)
    map = np.zeros(num_healpixels)
    ids = hp.ang2pix(nside, ra, dec, False, lonlat=True)
    for id in ids:
        map[id] += 1.0
    hp.mollview(map, title="", xsize=400, badcolor="grey")
    hp.graticule(dpar=30.0)
    plt.show()

    
    


if __name__ == '__main__':
    main()

