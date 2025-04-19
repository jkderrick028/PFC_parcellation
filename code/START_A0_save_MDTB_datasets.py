import os.path, pickle
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
from py_util_dx.py_utils import setProjectPath

"""

This script loads MDTB dataset and saves as pkl files. 

"""

projectPath, mainResultsPath = setProjectPath()
# base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
base_dir = '/cifs/diedrichsen/data/FunctionalFusion'
# base_dir = '/Users/jkderrick028/jxiang27_graham/scratch/7T_exploration/data/FunctionalFusion'
surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

resultsPath = os.path.join(projectPath, 'data')
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

# Get the atlas
atlas_str = 'fs32k'
atlas, ainf = am.get_atlas(atlas_str)

# # loading group data condAll
# X_group, info_group, dataset_obj_group = ds.get_dataset(base_dir,
#                                                         dataset='MDTB',
#                                                         atlas='fs32k',
#                                                         subj='group',
#                                                         sess='all',
#                                                         type='CondAll')

# loading individual data condAll

dataset_name = 'MDTB'
X_individuals, info_individuals, dataset_obj_individuals = ds.get_dataset(base_dir,
                                                                          dataset=dataset_name,
                                                                          atlas='fs32k',
                                                                          subj=None,
                                                                          sess='all',
                                                                          type='CondAll')

data = {'X_individuals': X_individuals, 'info_individuals': info_individuals, 'dataset_obj_individuals': dataset_obj_individuals}

PKL_data = os.path.join(resultsPath, f'{dataset_name}_Cond_All.pkl')
with open(PKL_data, 'wb') as pf:
    pickle.dump(data, pf)

