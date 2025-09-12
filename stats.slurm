#!/bin/bash
#SBATCH --job-name=stats
#SBATCH -N 1
#SBATCH --cpus-per-task=1  # Request 1 CPUs per array task
#SBATCH --time=168:00:00
#SBATCH --mem-per-cpu=32G   # 8GB per CPU = 32GB total per array task
#SBATCH --partition=serc
#SBATCH --mail-type=ALL
#SBATCH --output=stats.out

# # Set OpenMP environment variables
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1

# conda activate pycascade
export PYTHONUNBUFFERED=True
python stats_from_maxima.py