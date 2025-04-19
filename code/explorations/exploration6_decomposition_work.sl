#!/bin/bash -l
#SBATCH --job-name=exploration6_decomposition_work
#SBATCH --account=def-mmur
#SBATCH --time=0-1:59
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-user=jkderrick.jobscheduler@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=slurm_outputs/%x_%A.out

module load StdEnv/2020  gcc/9.3.0
module load fsl/6.0.4

export PATH=$PATH:~/projects/def-mmur/jxiang27/toolkits/workbench/bin_rh_linux64/

source ~/tcompo/bin/activate

python exploration6_decomposition_work.py
