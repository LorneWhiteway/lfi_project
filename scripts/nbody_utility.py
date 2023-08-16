#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
    Routines for working with nbodykit.
    Author: Lorne Whiteway.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from nbodykit.lab import *
from nbodykit.source.catalog import BinaryCatalog
from nbodykit.algorithms.fftpower import FFTPower
import utility
import datetime


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
    
def calculate_power_spectrum_of_simulation_box_old():

    simulation_box_filename = "/share/testde/ucapwhi/lfi_project/runsI/run016/run.00100"
    num_objects = 1259712000
    
    #simulation_box_filename = "/share/splinter/ucapwhi/lfi_project/experiments/gpu_512_256_1000/run.00100"
    #num_objects = 134217728
    
    binary_output_filename = simulation_box_filename + ".dat"
    
    if True:
        print("{} Reading file {}...".format(datetime.datetime.now().time(), simulation_box_filename))
        d = utility.read_one_box(simulation_box_filename)
        print("{} Finished reading file.".format(datetime.datetime.now().time()))
        
        e = np.column_stack((d['x']+0.5, d['y']+0.5, d['z']+0.5))
        print(e.shape[0])
        
        print("{} Writing file {}...".format(datetime.datetime.now().time(), binary_output_filename))
        # See https://nbodykit.readthedocs.io/en/latest/catalogs/reading.html#Binary-Data
        with open(binary_output_filename, 'wb') as ff:
            e.tofile(ff)
            ff.seek(0)
        print("{} Finished writing file.".format(datetime.datetime.now().time()))
    
    print("{} Reading file {}...".format(datetime.datetime.now().time(), binary_output_filename))
    f = BinaryCatalog(binary_output_filename, [('Position', ('f8', 3))], size=num_objects)
    print("{} Finished reading file.".format(datetime.datetime.now().time()))
    
    print(f)
    print("columns = ", f.columns) # default Weight,Selection also present
    print("total size = ", f.csize)
    
    print("{} About to create mesh".format(datetime.datetime.now().time()))
    mesh = f.to_mesh(Nmesh=32, BoxSize=1, dtype='f4')
    print(mesh)
    print("{} Finished creating mesh".format(datetime.datetime.now().time()))
    
    print("{} About to create FFTPower object".format(datetime.datetime.now().time()))
    r = FFTPower(mesh, mode='1d', dk=0.005, kmin=0.01)
    print("{} Finished creating FFTPower object".format(datetime.datetime.now().time()))
    
    Pk = r.power
    print(Pk)
    
# This is the function to call to calculate the PS of a pkdgrav3 z=0 simulation box.
# See e.g. my Slack messeages to NJ on 12 June 2022 and 16 August 2023.    
def PKDGrav3Example():

    simulation_box_filename = "/share/testde/ucapwhi/lfi_project/runsO/run029/run.00100"
    BoxSize = 1250 #Mpc/h
    
    #simulation_box_filename = "/share/splinter/ucapwhi/lfi_project/experiments/gpu_512_256_1000/run.00100"
    #BoxSize = 1000 #Mpc/h
    

    print("{} About to call utility.read_one_box".format(datetime.datetime.now().time()))
    d = utility.read_one_box(simulation_box_filename)
    print("{} Finished calling utility.read_one_box".format(datetime.datetime.now().time()))
    
    
    num_data = d['x'].shape[0]
    print("num_data = {}".format(num_data))
    
    data = numpy.empty(num_data, dtype=[('Position', (d['x'].dtype, 3))])
    
    data['Position'] = (np.column_stack([d['x'], d['y'], d['z']]) + 0.5) * BoxSize
    
    print("{} About to call ArrayCatalog".format(datetime.datetime.now().time()))
    cat = ArrayCatalog(data, BoxSize=BoxSize, Nmesh=1024)
    print("{} Finished calling ArrayCatalog".format(datetime.datetime.now().time()))
    
    print(cat)
    
    AnalyzeCatalog(cat, simulation_box_filename + ".ps_results")
    


def LogNormalCatalogExample():
    print("From https://nbodykit.readthedocs.io/en/latest/cookbook/fftpower.html")
    redshift = 0.55
    cosmo = cosmology.Planck15
    Plin = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')
    b1 = 2.0

    cat = LogNormalCatalog(Plin=Plin, nbar=3e-4, BoxSize=1380., Nmesh=256, bias=b1, seed=42)
    print(cat)
    
    AnalyzeCatalog(cat, None)
    
    
def ArrayCatalogExample():
    from nbodykit.source.catalog import ArrayCatalog
    
    print("https://nbodykit.readthedocs.io/en/latest/catalogs/reading.html#array-data")

    # generate random data
    num_data = 4096*4096
    BoxSize = 1380
    data = numpy.empty(num_data, dtype=[('Position', ('f8', 3))])
    data['Position'] = numpy.random.uniform(size=(num_data, 3)) * BoxSize

    # save to a npy file
    numpy.save("npy-example.npy", data)

    data = numpy.load("npy-example.npy")

    # initialize the catalog
    cat = ArrayCatalog(data, BoxSize=BoxSize, Nmesh=128)
    print(cat)
    
    AnalyzeCatalog(cat, None)
    
    
    
    
def AnalyzeCatalog(cat, file_to_save_results):

    print("{} About to call cat.to_mesh".format(datetime.datetime.now().time()))
    mesh = cat.to_mesh(resampler='tsc', compensated=True)
    print("{} Finished calling cat.to_mesh".format(datetime.datetime.now().time()))
    
    kmin = 0.005
    
    print("{} About to call FFTPower".format(datetime.datetime.now().time()))
    r = FFTPower(mesh, mode='1d', dk=0.005, kmin=kmin)
    print("{} Finished calling FFTPower".format(datetime.datetime.now().time()))
    Pk = r.power
    print(Pk)
    
    print(Pk.coords)
    for k in Pk.attrs:
        print("%s = %s" %(k, str(Pk.attrs[k])))
        
    if file_to_save_results:
        r.save(file_to_save_results)
        np.save(file_to_save_results + ".npy", np.array([Pk['k'], Pk['power'].real - Pk.attrs['shotnoise']]))
        
    plt.switch_backend('Qt5Agg')

    plt.loglog(Pk['k'], Pk['power'].real - Pk.attrs['shotnoise'])

    # format the axes
    plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    plt.ylabel(r"$P(k)$ [$h^{-3}\mathrm{Mpc}^3$]")
    
    # kmax will be pi*NMesh/BoxLength...
    #plt.xlim(kmin, 0.6)
    
    plt.show()
    
 

    
    

if __name__ == '__main__':
    
    #hello_world()
    #make_amended_euclid_transfer_function()
    #compare_with_euclid()
    #make_specific_cosmology_transfer_function()
    #compare_two_transfer_functions()
    #make_specific_cosmology_transfer_function_caller()
    #calculate_power_spectrum_of_simulation_box()
    #LogNormalCatalogExample()
    #ArrayCatalogExample()
    PKDGrav3Example()
    
    
