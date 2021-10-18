# cuda_test

Contains source code for a simple program that uses the GPU.

helper_cuda.h and helper_string.h are copies of files distributed at the Oxford GPU course. I edited helper_cuda.h to write to stdout instead of stderr.

To build:
- Go to one of the GPUs e.g. via `srun -p GPU --gres=gpu:v100:1 --pty tcsh` on splinter (could change v100 to k80).
- `source ../set_environment.csh BUILD`
- `make clean`
- `make`

To run:
`sbatch ./cuda_job_script`

