#!/usr/bin/env python

""" 
    Monitors a running PKDGRAV3 instance, deleting a dummy output file as soon as
    the corresponding real file(s) start to appear.
    Author: Lorne Whiteway.
"""

import glob
import os
import time
import utility
import contextlib
import sys


# Assumes there may be dummy (placeholding) output files, with names such as dummy.00076.npy
# Then as soon as example.00076.hpb.0 gets created, dummy.00076.npy will get deleted (but
# no error if it wasn't there).


def sorted_list_of_steps_for_which_there_are_partial_files(directory):
    list_of_steps = [int(utility.string_between_strings(f, "example.", ".hpb")) for f in glob.glob(directory + "/example.*.hpb.*")]
    list_of_steps = list(set(list_of_steps)) # Remove duplicates
    list_of_steps.sort()
    return list_of_steps
    

# If be_conservative then we don't process a step until the files in the next step are being written.
def monitor(directory, be_conservative, time_in_seconds_to_wait_until_next_pass):
    
    control_file_name = directory + "/control.par"
    nside = int(utility.get_float_from_control_file(control_file_name, "nSideHealpix"))
    
    print("Monitoring directory = {} with be_conservative = {} and seconds_to_wait = {}".format(directory, be_conservative, time_in_seconds_to_wait_until_next_pass))
    print("nside = {}".format(nside))
    
    while True:
    
        active_steps = sorted_list_of_steps_for_which_there_are_partial_files(directory)
        print("Steps for which there are partial files: {}".format(active_steps))

        if len(active_steps) >= (2 if be_conservative else 1):
            step_to_process = active_steps[0]
            step_to_process_as_str = str(step_to_process).zfill(5)
            print("Processing step {}".format(step_to_process))
            
            filespec = directory + "/example.{}.hpb".format(step_to_process_as_str)
            utility.save_all_lightcone_files_caller_core(filespec, nside, nside, delete_hpb_files_when_done=True, save_image_files=True)

            # Delete any corresponding dummy file
            dummy_filename = directory + "/dummy.{}.npy".format(step_to_process_as_str)
            if os.path.isfile(dummy_filename):
                print("Deleting {}...".format(dummy_filename))
                os.remove(dummy_filename)
    
        print("Waiting {} seconds...".format(time_in_seconds_to_wait_until_next_pass))
        time.sleep(time_in_seconds_to_wait_until_next_pass)
    
    


if __name__ == '__main__':

    if len(sys.argv) < 4:
        print("Usage: monitor directory be_conservative seconds_to_wait")
    else:
        directory = sys.argv[1]
        be_conservative = bool(sys.argv[2].lower() == "true")
        time_in_seconds_to_wait_until_next_pass = int(sys.argv[3])
        
        monitor(directory, be_conservative, time_in_seconds_to_wait_until_next_pass)
    