#!/usr/bin/env python

""" 
    Author: Lorne Whiteway.
"""

import utility
import traceback
import sys
import os
    
    
def expand_shell_script(original_shell_script_file_name, new_shell_script_file_name, list_of_jobs_string):

    with open(new_shell_script_file_name, 'w') as out_file:
        for job in decode_list_of_jobs_string(list_of_jobs_string):
            with open(original_shell_script_file_name, 'r') as in_file:
                for line in in_file:
                    out_file.write(line.replace("{}", str(job).zfill(3)))
    make_file_executable(new_shell_script_file_name)
        
    

def expand_shell_script_caller():
    
    expand_shell_script("./sample.sh", "./new_sample.sh", "1-16 x3 x5 x7-8")


def show_help():
    print("Usage: {} original_shell_script_file_name new_shell_script_file_name list_of_jobs...".format(os.path.basename(__file__)))
    print("Builds a new script file (called new_shell_script_file_name) based on ")
    print(" original_shell_script_file_name as a template. The new file contains multiple repeats")
    print(" of the original file, with '{}' strings replaced with integers (zero padded to be of")
    print(" length 3). The integers are specified last on the command line; use a syntax of which the")
    print(" following is an example: '1-20 22,23 25 x19 x12-17 14'; this example would yield integers")
    print(" 1,2,3,4,5,6,7,8,9,10,11,14,18,20,22,23,25")



if __name__ == '__main__':
     
    try:
    
        command_line_array = sys.argv
    
        if len(command_line_array) < 4 or "-h" in command_line_array or "--help" in command_line_array:
            show_help()
        else:
            original_shell_script_file_name = command_line_array[1]
            new_shell_script_file_name = command_line_array[2]
            list_of_jobs_string = " ".join(command_line_array[3:])
            
            utility.expand_shell_script(original_shell_script_file_name, new_shell_script_file_name, list_of_jobs_string)
        
    except Exception as err:
        print(traceback.format_exc())
        sys.exit(1)

    
