#!/usr/bin/env bash
source /mnt/lustre/tursafs1/home/dp327/dp327/shared/lfi_project/set_environment_tursa.sh
cd /mnt/lustre/tursafs1/home/dp327/dp327/shared/lfi_project/runsS/run011/
rm -f /mnt/lustre/tursafs1/home/dp327/dp327/shared/lfi_project/runsS/run011/monitor_stop.txt
/mnt/lustre/tursafs1/home/dp327/dp327/shared/lfi_project/scripts/monitor.py /mnt/lustre/tursafs1/home/dp327/dp327/shared/lfi_project/runsS/run011/ > /mnt/lustre/tursafs1/home/dp327/dp327/shared/lfi_project/runsS/run011/monitor_output.txt &
/mnt/lustre/tursafs1/home/dp327/dp327/shared/lfi_project/pkdgrav3/build_tursa/pkdgrav3 ./control.par > ./output.txt
echo stop > /mnt/lustre/tursafs1/home/dp327/dp327/shared/lfi_project/runsS/run011/monitor_stop.txt
python3 /mnt/lustre/tursafs1/home/dp327/dp327/shared/lfi_project/scripts/pkdgrav3_postprocess.py -l -d -z -f . >> ./output.txt
cd /mnt/lustre/tursafs1/home/dp327/dp327/shared/lfi_project/runsS/
tar czvf run011.tar.gz ./run011/
test -f ./run011.tar.gz && rm ./run011/run*
