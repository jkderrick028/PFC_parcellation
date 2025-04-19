#!/bin/bash -l
#SBATCH --job-name=exploration5_smooth_decomposition_work_3
#SBATCH --account=def-mmur
#SBATCH --time=0-10:59
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-user=jkderrick.jobscheduler@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=slurm_outputs/%x_%A.out

module load StdEnv/2020  gcc/9.3.0
module load fsl/6.0.4

export PATH=$PATH:~/scratch/toolkits/workbench/bin_rh_linux64/

source ~/tcompo/bin/activate

python exploration5_smooth_decomposition_work_3.py
