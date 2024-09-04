#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
    Status report on a directory hosting several PKDGRAV runs.
    For help, see the README file.
    Author: Lorne Whiteway.
"""

import utility
import traceback
import sys
import os


def item_is_in_command_line(command_line_array, short_option, long_option):
    for item in command_line_array:
        if item == short_option or item == long_option:
            return True
    return False


def show_help():
    print("Usage: {} runs_letter".format(os.path.basename(__file__)))




if __name__ == '__main__':

    try:
    
        command_line_array = sys.argv
    
        if len(command_line_array) != 2 or item_is_in_command_line(command_line_array, "-h", "--help"):
            show_help()
            sys.exit(0)
        
        runs_letter = command_line_array[1]

        utility.runs_directory_status(runs_letter)
        
    except Exception as err:
        print(traceback.format_exc())
        sys.exit(1)
        




    
    
