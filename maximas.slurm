#!/bin/bash
#SBATCH --job-name=mc_maximas
#SBATCH -N 1
#SBATCH --cpus-per-task=1  # Request 4 CPUs per array task
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=8G   # 8GB per CPU = 32GB total per array task
#SBATCH --partition=serc
#SBATCH --mail-type=ALL
#SBATCH --array=0-3

# Set OpenMP environment variables
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# conda activate pycascade
export PYTHONUNBUFFERED=True
python gen_block_maxima.py --series_type=$SLURM_ARRAY_TASK_ID