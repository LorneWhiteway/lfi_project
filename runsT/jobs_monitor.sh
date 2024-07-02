echo "JOBS FAILED DUE TO TIME LIMIT:"
grep -r 'DUE TO TIME LIMIT' --include *.out ./ | sort -k 1
echo
echo "ANY OF THE ABOVE THAT ARE ALSO IN THE FOLLOWING LIST MAY HAVE FAILED DURING COMPRESSION AND HENCE MAY BE RECOVERABLE:"
find . -name "run.00100.lightcone.npy" -print | sort -k 1
echo
echo "JOBS FAILED DUE TO DISK SPACE:"
grep -r 'Disk quota exceeded' --include slurm* ./ | sort -k 1
echo
echo "JOBS FAILED DUE TO OUT OF MEMORY:"
grep -r 'Cannot allocate memory' --include slurm* ./ | sort -k 1
echo
echo "FINISHED JOBS THAT HAVE BEEN COPIED TO THE TAPE ARCHIVE STAGING AREA:"
ls -l /mnt/lustre/tursafs1/archive/dp327/runsT/run???.tar.gz
echo
echo "FINISHED JOBS NOT YET ARCHIVED (SORTED BY NAME):"
ls -l run???.tar.gz
echo
echo "FINISHED JOBS NOT YET ARCHIVED (SORTED BY DATE):"
ls -ltr run???.tar.gz
echo
echo "RUNNING AND QUEUED JOBS:"
squeue | grep p14_ | sort -k 3
echo
echo "DISK USAGE FOR PROJECT:"
lfs quota -hp $(id -g) .

