#!/usr/bin/env python

     
        
def create_supercmake(top_directory):
    import fnmatch
    import os
    
    anything_found = False
    
    abs_top_directory = os.path.abspath(top_directory)
    
    output_filename = abs_top_directory + '/super_cmake.cmake'

    with open(output_filename, 'w') as outfile:
        for root, dirnames, filenames in os.walk(abs_top_directory):
            for filename in (fnmatch.filter(filenames, 'CMakeLists.txt') + fnmatch.filter(filenames, 'Find*.cmake')):
                anything_found = True
                full_file_name = os.path.join(root, filename)
                print(full_file_name)
                outfile.write("\n")
                outfile.write("#---------------------------------------------------------------------------------------------------" + "\n")
                outfile.write(full_file_name + "\n")
                outfile.write("#---------------------------------------------------------------------------------------------------" + "\n")
                outfile.write("\n")
                with open(full_file_name) as infile:
                    outfile.write(infile.read())
                    
    if not anything_found:
        print("No cmake files found")
    else:
        print("Output written to " + output_filename)
        
    
    
if __name__ == '__main__':

    import sys
    if len(sys.argv) != 2:
        print("Usage: " + __file__ + " <startdirectory>")
        print("Creates file supercmake.txt in <start directory>.")
    else:
        create_supercmake(sys.argv[1])
  
    
    
    
