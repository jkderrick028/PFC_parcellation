#!/bin/bash -l
#SBATCH --job-name=START_A4_DCBC_pairwise
#SBATCH --account=def-hallett-ab
#SBATCH --time=0-5:59
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G 
#SBATCH --mail-user=jkderrick.jobscheduler@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=slurm_outputs/%x_%A.out

source ~/pfc_parcellation/bin/activate

python START_A3_bayes_parcellation.py
python START_A4_DCBC_pairwise.py 
