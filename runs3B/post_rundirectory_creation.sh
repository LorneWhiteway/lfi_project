#!/usr/bin/env bash

# $1 = this_run_directory e.g. /mnt/lustre/tursafs1/home/dp327/dp327/shared/lfi_project/runs3B/run007
# $2 = run_string e.g. 007


sed -e 's/class_processed_runsA/class_processed_runsB/g' -e 's/ # calculated from sigma_8 = 0.7622847978449322//g' $1/../marco_files/run$2/control.par > $1/control.par
if [ $? -ne 0 ]; then
    exit 1
fi
cp $1/../marco_files/class-param_runsA_w0wa_run$2 $1/class-param_runsB_w0wa_run$2
if [ $? -ne 0 ]; then
    exit 2
fi
cp $1/../marco_files/class_processed_runsA_w0wa_run$2.hdf5 $1/class_processed_runsB_w0wa_run$2.hdf5
if [ $? -ne 0 ]; then
    exit 3
fi

