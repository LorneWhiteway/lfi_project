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
    

def transfer_function_from_hdf5(hdf5_filename, transfer_function_filename):
    f_in = h5py.File(hdf5_filename)
    k = f_in['perturbations']['k'][()]
    print("k")
    print(k)
    
    rho_cdm_b = f_in['background']['rho_cdm+b'][-1] # -1 for last element i.e. z=0
    print("rho_cdm_b")
    print(rho_cdm_b)
    
    at_z_equals_zero = -1
    
    rho_ncdm_0 = f_in['background']['rho_ncdm[0]'][at_z_equals_zero]
    rho_ncdm_1 = f_in['background']['rho_ncdm[1]'][at_z_equals_zero]
    rho_ncdm_2 = f_in['background']['rho_ncdm[2]'][at_z_equals_zero]
    

    delta_cdm_b = f_in['perturbations']['delta_cdm+b'][at_z_equals_zero,:]
    delta_ncdm_0 = f_in['perturbations']['delta_ncdm[0]'][at_z_equals_zero,:]
    delta_ncdm_1 = f_in['perturbations']['delta_ncdm[1]'][at_z_equals_zero,:]
    delta_ncdm_2 = f_in['perturbations']['delta_ncdm[2]'][at_z_equals_zero,:]
    
    print("Here 1")
    print(k.shape, rho_cdm_b.shape, delta_cdm_b.shape,rho_ncdm_0.shape, delta_ncdm_0.shape, rho_ncdm_1.shape, delta_ncdm_1.shape, rho_ncdm_2.shape, delta_ncdm_2.shape)
    
    transfer_fn = rho_cdm_b * delta_cdm_b + rho_ncdm_0 * delta_ncdm_0 + rho_ncdm_1 * delta_ncdm_1 + rho_ncdm_2 * delta_ncdm_2 / (rho_cdm_b + rho_ncdm_0 + rho_ncdm_1 + rho_ncdm_2)
    transfer_fn *= k**-2
    transfer_fn /= transfer_fn[0]
    transfer_fn = np.stack((k, transfer_fn), axis=-1)

    print(transfer_fn[0,:])
    print(transfer_fn[-1,:])
    
    np.savetxt(transfer_function_filename, transfer_fn)
    
def transfer_function_from_hdf5_test_harness():
    hdf5_filename = "/mnt/lustre/tursafs1/home/dp327/dp327/shared/lfi_project/runsV14/hdf5/class_processed_flagship_test2.hdf5"
    print("hdf5_filename = {}".format(hdf5_filename))
    transfer_function_filename = "/mnt/lustre/tursafs1/home/dp327/dp327/shared/lfi_project/runsV14/run021/transfer_function_V14_021.txt"
    print("transfer_function_filename = {}".format(transfer_function_filename))
    
    
    transfer_function_from_hdf5(hdf5_filename, transfer_function_filename)
    
    
    
if __name__ == '__main__':
    
    #perturb_delta_metric()
    transfer_function_from_hdf5_test_harness()
