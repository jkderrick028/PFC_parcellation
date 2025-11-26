#!/bin/bash -l
#SBATCH --job-name=START_A2_gse_decomposition
#SBATCH --account=def-mmur
#SBATCH --time=0-10:59
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G 
#SBATCH --mail-user=jkderrick.jobscheduler@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=slurm_outputs/%x_%A.out

module load StdEnv/2020  gcc/9.3.0

source ~/pfc_parcellation/bin/activate

python START_A2_gse_decomposition.py 'Demand'
python START_A2_gse_decomposition.py 'Language'
python START_A2_gse_decomposition.py 'MDTB'
python START_A2_gse_decomposition.py 'Nishimoto'
python START_A2_gse_decomposition.py 'HCPur100'

