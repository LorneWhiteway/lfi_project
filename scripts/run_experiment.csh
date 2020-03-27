#!/usr/bin/tcsh
echo Start
echo $argv[1]
cd $argv[1]
pwd
/share/splinter/ucapwhi/lfi_project/pkdgrav3/build/pkdgrav3 ./control.par > ./example_output.txt

