#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
    Post-processing of a PKDGRAV3 output directory.
    For help, see the README file.
    Author: Lorne Whiteway.
"""

import utility
import traceback
import sys
import os
import numpy as np


def item_is_in_command_line(command_line_array, short_option, long_option):
    for item in command_line_array:
        if item == short_option or item == long_option:
            return True
    return False


def show_help():
    print("Usage: {} [Options] directory".format(os.path.basename(__file__)))
    print("Run this after having run PKDGRAV3 (which will have created raw, partial lightcone files)")
    print(" in order to create derived files such as full lightcones and images.")
    print("Options:")
    print("-l or --lightcone  create full lightcone files")
    print("-m or --mollview   create image files in mollview format")
    print("-o or --orthview   create image files in ortho format")
    print("-z or --zfile      create a text file specifying the redshift ranges for the lightcones")
    print("-s or --status     print a summary of the files in the directory")
    print("-f or --force      create output files even if they are present already")
    print("-a or --all        all of the above")
    print("-h or --help       print this help message, then exit")
    
    


if __name__ == '__main__':

    try:
    
        command_line_array = sys.argv
    
        if len(command_line_array) == 1 or item_is_in_command_line(command_line_array, "-h", "--help"):
            show_help()
            sys.exit(0)
            
        if command_line_array[-1][0] == "-":
            # Final argument should be directory; if this starts with a hyphe then probably the user
            # forgot to supply it.
            show_help()
            print("\nDid you forget to supply the final command-line argument (directory name)?")
            sys.exit(0)
            
            
        do_all = item_is_in_command_line(command_line_array, "-a", "--all")
        do_mollview_images = do_all or item_is_in_command_line(command_line_array, "-m", "--mollview")
        do_ortho_images = item_is_in_command_line(command_line_array, "-o", "--orthview")
        do_lightcone_files = do_mollview_images or do_ortho_images or do_all or item_is_in_command_line(command_line_array, "-l", "--lightcones")
        do_z_file = do_all or item_is_in_command_line(command_line_array, "-z", "--zfile")
        do_status = do_all or item_is_in_command_line(command_line_array, "-s", "--status")
        do_force = item_is_in_command_line(command_line_array, "-f", "--force")
        
        directory = command_line_array[-1]
        
            
        utility.pkdgrav3_postprocess(directory, do_lightcone_files, do_mollview_images, do_ortho_images, do_z_file, do_status, do_force)
        
    except Exception as err:
        print(traceback.format_exc())
        sys.exit(1)
        




    
    
