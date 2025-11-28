#!/bin/bash -l
#SBATCH --job-name=START_A7_spatialACF_map_LH
#SBATCH --account=def-hallett-ab
#SBATCH --time=0-3:59
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-user=jkderrick.jobscheduler@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=slurm_outputs/%x_%A.out

module load StdEnv/2020  gcc/9.3.0
module load fsl/6.0.4

export PATH=$PATH:~/projects/def-mmur/jxiang27/toolkits/workbench/bin_rh_linux64/

source ~/pfc_parcellation/bin/activate

python START_A7_spatialACF_map_LH.py 'MDTB'
