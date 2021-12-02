#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
    Routines for working with nbodykit.
    Author: Lorne Whiteway.
"""

import numpy as np
import matplotlib.pyplot as plt
from nbodykit.lab import *


def hello_world():
    k = 0.00001
    t = cosmology.power.transfers.CLASS(cosmology.Planck15,0.0)(k)
    print(t) # Should be 1.0


def compare_with_euclid():

    file_name = "/share/splinter/ucapwhi/lfi_project/data/euclid_z0_transfer_combined.dat"
    d = np.loadtxt(file_name, delimiter = " ")
    k = d[:,0]
    t_euclid = d[:,1] 
    t_class_1 = cosmology.power.transfers.NoWiggleEisensteinHu(cosmology.Planck15,0.0)(k)
    t_class_2 = cosmology.power.transfers.EisensteinHu(cosmology.Planck15,0.0)(k)
    t_class_3 = cosmology.power.transfers.CLASS(cosmology.Planck15,0.0)(k)
    t_class_4 = cosmology.power.transfers.NoWiggleEisensteinHu(cosmology.WMAP5,0.0)(k)
    t_class_5 = cosmology.power.transfers.EisensteinHu(cosmology.WMAP5,0.0)(k)
    t_class_6 = cosmology.power.transfers.CLASS(cosmology.WMAP5,0.0)(k)
    t_amended = d = np.loadtxt("/share/splinter/ucapwhi/lfi_project/data/euclid_z0_transfer_combined_amended.dat", delimiter = " ")[:,1]
    
    if False:
        for c in [cosmology.Planck15, cosmology.WMAP5]:
            print("=======")
            cosmo_dict = dict(c)
            for y in cosmo_dict:
                print("{} = {}".format(y, cosmo_dict[y]))

    if True:
        for (i, clr) in zip([t_class_1, t_class_2, t_class_3, t_class_4, t_class_5, t_class_6], ['red', 'blue', 'green', 'orange', 'black', 'violet']):
            plt.plot(k, t_euclid/i, c=clr)
        plt.xscale('log')
        plt.xlabel('$k$')
        plt.ylabel('Euclid transfer fn / other transfer fn')
        plt.show()
    
    if False:
        plt.loglog(k, t_euclid)
        plt.loglog(k, t_amended)
        plt.show()
    
    if False:
        plt.plot(k, t_euclid)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        
def make_amended_euclid_transfer_function():
    file_name_orig = "/share/splinter/ucapwhi/lfi_project/data/euclid_z0_transfer_combined.dat"
    file_name_amend = "/share/splinter/ucapwhi/lfi_project/data/euclid_z0_transfer_combined_amended.dat"
    d_orig = np.loadtxt(file_name_orig, delimiter = " ")
    k = d_orig[:,0]
    d_amend = np.column_stack([k, d_orig[:,1]*(0.75 + 0.02*(np.log10(k)))])
    if False:
        print(d_orig)
        print("=======")
        print(d_amend)
    np.savetxt(file_name_amend, d_amend, delimiter = " ", fmt = "%10.5E")
    

def print_cosmology(cosmo, cosmo_name):
    print("======= Parameters for {} =======".format(cosmo_name))
    cosmo_dict = dict(cosmo)
    for key in cosmo_dict:
        print("{} = {}".format(key, cosmo_dict[key]))
    print("sigma8 = {}".format(cosmo.sigma8))
    
    
    
def trim_rows_containing_nan(a):
    # Credit: https://note.nkmk.me/en/python-numpy-nan-remove/
    return a[~np.isnan(a).any(axis=1), :]
    

def make_specific_cosmology_transfer_function(identifier, h, Omega0_b, Omega0_cdm, n_s, sigma8):

    # Load an old transfer function (just to get a list of wavenumbers).
    file_name_orig = "../data/euclid_z0_transfer_combined.dat"
    d_orig = np.loadtxt(file_name_orig, delimiter = " ")
    k = d_orig[:,0]
    
    new_name = "specific_cosmology_{}".format(identifier)
    new_cosmo = (cosmology.Planck15).clone(h=h, Omega0_b=Omega0_b, Omega0_cdm=Omega0_cdm, n_s=n_s).match(sigma8=sigma8)
    transfer_function = np.column_stack([k, cosmology.power.transfers.CLASS(new_cosmo, 0.0)(k)])
    transfer_function = trim_rows_containing_nan(transfer_function)
    
    file_name_new = "../data/" + new_name + "_transfer_function.dat"
    print("Saved transfer function in {}".format(file_name_new))
    print("Consider updating the README file with the following specifications for this cosmology.")
    print_cosmology(new_cosmo, new_name)
    np.savetxt(file_name_new, transfer_function, delimiter = " ", fmt = "%10.5E")
    

def make_specific_cosmology_transfer_function_caller():
    make_specific_cosmology_transfer_function(identifier="02", h=0.6736, Omega0_b=0.0493, Omega0_cdm=0.2107, n_s=0.9649, sigma8=0.84)

    
def compare_two_transfer_functions():
    file_name_1 = "/share/splinter/ucapwhi/lfi_project/data/euclid_z0_transfer_combined.dat"
    file_name_2 = "/share/splinter/ucapwhi/lfi_project/data/specific_cosmology_01_transfer_function.dat"
    d_1 = np.loadtxt(file_name_1, delimiter = " ")
    d_2 = np.loadtxt(file_name_2, delimiter = " ")
    # Here we naively assume that the x axis (i.e. first column) of file 2 is an initial subset of the x axis of file 1
    d_1 = d_1[:d_2.shape[0],:]
    plt.plot(d_1[:,0], d_1[:,1]/d_2[:,1])
    plt.xscale('log')
    plt.xlabel('$k$')
    plt.ylabel('Euclid transfer fn / new transfer fn')
    plt.show()
    
    

if __name__ == '__main__':
    
    #hello_world()
    #make_amended_euclid_transfer_function()
    #compare_with_euclid()
    #make_specific_cosmology_transfer_function()
    #compare_two_transfer_functions()
    make_specific_cosmology_transfer_function_caller()
    