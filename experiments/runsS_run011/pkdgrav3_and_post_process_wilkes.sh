#!/usr/bin/env bash
module load python/3.8
source /rds/project/dirac_vol5/rds-dirac-dp153/lfi_project/env/bin/activate
cd /rds/project/dirac_vol5/rds-dirac-dp153/lfi_project/runsS/run011/
rm -f /rds/project/dirac_vol5/rds-dirac-dp153/lfi_project/runsS/run011/monitor_stop.txt
/rds/project/dirac_vol5/rds-dirac-dp153/lfi_project/scripts/monitor.py /rds/project/dirac_vol5/rds-dirac-dp153/lfi_project/runsS/run011/ > /rds/project/dirac_vol5/rds-dirac-dp153/lfi_project/runsS/run011/monitor_output.txt &
/rds/project/dirac_vol5/rds-dirac-dp153/lfi_project/pkdgrav3/build_wilkes/pkdgrav3 ./control.par > ./output.txt
echo stop > /rds/project/dirac_vol5/rds-dirac-dp153/lfi_project/runsS/run011/monitor_stop.txt
python3 /rds/project/dirac_vol5/rds-dirac-dp153/lfi_project/scripts/pkdgrav3_postprocess.py -l -d -z -f . >> ./output.txt
cd /rds/project/dirac_vol5/rds-dirac-dp153/lfi_project/runsS/
tar czvf run011.tar.gz ./run011/
test -f ./run011.tar.gz && rm ./run011/run*
