# Multi-GPU Performance Test

This experiment compares 1-GPU vs 4-GPU performance for PKDGRAV3 using the run140 configuration.

## Setup (run on Tursa)

```bash
cd /mnt/lustre/tursafs1/home/dp327/dp327/shared/lfi_project/experiments/multi_gpu_test/
chmod +x setup_symlinks.sh
./setup_symlinks.sh
```

## Run Tests

### 4-GPU Test
```bash
sbatch cuda_job_script_tursa_4gpu
```

### 1-GPU Baseline Test
```bash
sbatch cuda_job_script_tursa_1gpu
```

## Compare Results

After both jobs complete, compare the wallclock times:
```bash
grep -i "wallclock\|total\|elapsed" output_1gpu.txt output_4gpu.txt
```

## Configuration

| Parameter | 1-GPU | 4-GPU |
|-----------|-------|-------|
| MPI tasks | 1 | 4 |
| GPUs | 1 | 4 |
| CPUs per task | 48 | 12 |
| Total CPUs | 48 | 48 |

Both tests use the same control.par from run140 (via symlink).

