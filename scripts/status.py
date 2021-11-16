#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
    Status report on a PKDGRAV3 output directory.
    Author: Lorne Whiteway.
"""

import sys
import traceback
import glob
import numpy as np
import os
import datetime


# ======================== Start of code for reporting on the status of an 'experiments' directory ========================

# This functionality is exposed through 'status.py'.

# Helper functions
def report_whether_file_exists(file_description, file_name):
    print("File {} {} '{}'".format(("exists:" if os.path.isfile(file_name) else "DOES NOT exist: no"), file_description, file_name))
    
# Returns True if there are file, else False
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
    report_whether_file_exists("control file", control_file_name)
    report_whether_file_exists("old log file", os.path.abspath(os.path.join(directory, "example.log")))
    report_whether_file_exists("log file", os.path.abspath(os.path.join(directory, "run.log")))
    report_whether_file_exists("old output file", os.path.abspath(os.path.join(directory, "example_output.txt")))
    report_whether_file_exists("output file", os.path.abspath(os.path.join(directory, "output.txt")))
    report_whether_several_files_exist("partial lightcone", os.path.join(directory, "*.hpb*"))
    if report_whether_several_files_exist("full lightcone", os.path.join(directory, "*.npy")):
        print("   Type of full lightcone files is {}".format(npy_file_data_type(glob.glob(os.path.join(directory, "*.npy"))[0])))
    report_whether_several_files_exist("lightcone image (orthview)", os.path.join(directory, "*.lightcone.png"))
    report_whether_several_files_exist("lightcone image (mollview)", os.path.join(directory, "*.lightcone.mollview.png"))
    report_whether_file_exists("z_values file", os.path.abspath(os.path.join(directory, "z_values.txt")))


# ======================== End of code for reporting on the status of an 'experiments' directory ========================





if __name__ == '__main__':

    try:
    
        if len(sys.argv) != 2:
            raise SystemError("Provide directory name as command line argument.")
        status(sys.argv[1])
        
        
    except Exception as err:
        print(traceback.format_exc())
        sys.exit(1)
        




    
    
