#!/usr/bin/env bash
module load python/3.8
source /rds/user/dc-whit2/rds-dirac-dp153/lfi_project/env/bin/activate
cd /rds/user/dc-whit2/rds-dirac-dp153/lfi_project/experiments/run999/
/rds/user/dc-whit2/rds-dirac-dp153/lfi_project/pkdgrav3/build_wilkes/pkdgrav3 ./control.par > ./output.txt
python3 /rds/user/dc-whit2/rds-dirac-dp153/lfi_project/scripts/pkdgrav3_postprocess.py -l -d -f . >> ./output.txt
cd /rds/user/dc-whit2/rds-dirac-dp153/lfi_project/experiments/
tar czvf run999.tar.gz ./run999/
test -f ./run999.tar.gz && rm ./run999/run*
