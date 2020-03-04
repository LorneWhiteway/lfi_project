#!/usr/bin/env python

""" 
    Reads data from file produced by piping output from PKDGRAV3 to text file.
    Author: Lorne Whiteway.
"""
    

def read_one_output_file(filename):
    
    import numpy as np
    
    step_list = []
    time_list = []
    z_list = []
    
    with open(filename, 'r') as infile:
        for el in infile:
            step_string = "Writing output for step "
            if step_string in el:
                step = int(el.split(step_string)[1])
                step_list.append(step)
            if "Time:" in el and "Redshift:" in el and not "Expansion factor" in el:
                t = float(el.split()[0].split(":")[1]) # Time
                time_list.append(t)
                z = float(el.split()[1].split(":")[1]) # redshift
                z_list.append(z)
                
    return (np.array(step_list), np.array(time_list), np.array(z_list))


            
def main():

    import matplotlib.pyplot as plt

    filename = "/share/splinter/ucapwhi/lfi_project/experiments/output.txt"
    (s_arr, t_arr, z_arr) = read_one_output_file(filename)
    
    if False:
        plt.plot(s_arr, t_arr)
        plt.plot(s_arr, z_arr)
        plt.show()
    
    
    for (s, t, z) in zip(s_arr, t_arr, z_arr):
        print(s,t,z)
    
    
    
    
    

if __name__ == '__main__':
    main()
    

