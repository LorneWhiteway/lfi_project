#!/usr/bin/env python

import h5py
import numpy as np


# See https://stackoverflow.com/questions/70161692/how-to-edit-part-of-an-hdf5-file

# To install h5py on tursa I did:
# cd /mnt/lustre/tursafs1/home/dp327/dp327/shared/lfi_project/env
# pip install h5py


def perturb_delta_metric():

    filename = "/mnt/lustre/tursafs1/home/dp327/dp327/shared/lfi_project/runsV14/hdf5/class_processed_gs2_batch3_V14_012.hdf5"
    f = h5py.File(filename,'r+')
    a = f['perturbations']['delta_metric'][()]
    a *= 1e-4
    f['perturbations']['delta_metric'][()] = a
    f.close()
    
    
    
if __name__ == '__main__':
    #perturb_delta_metric()
